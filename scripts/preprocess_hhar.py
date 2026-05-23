import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch


DEFAULT_KEEP_LABELS = ("bike", "sit", "stand", "walk", "stairsup", "stairsdown")
WOODS_DEFAULT_USERS = ("a", "b", "c", "d", "e")
WOODS_DEFAULT_MODELS = ("nexus4", "s3", "s3mini", "lgwatch", "gear")


def _normalize_label(raw: str) -> str:
    return str(raw).strip().lower()


def _split_csv_arg(raw: str) -> List[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _parse_csv_row(row: Dict[str, str]) -> Optional[Tuple[float, float, float, str, str, str, str]]:
    try:
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        user = str(row["User"]).strip().lower()
        model = str(row["Model"]).strip().lower()
        device = str(row["Device"]).strip().lower()
        gt = _normalize_label(row["gt"])
    except Exception:
        return None
    if not user or not model or not device or not gt:
        return None
    return x, y, z, user, model, device, gt


def _parse_csv_row_with_time(
    row: Dict[str, str],
) -> Optional[Tuple[int, float, float, float, str, str, str, str]]:
    try:
        creation = row.get("Creation_Time")
        arrival = row.get("Arrival_Time")
        ts_raw = creation if creation not in (None, "") else arrival
        if ts_raw in (None, ""):
            return None
        ts = int(float(str(ts_raw).strip()))
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        user = str(row["User"]).strip().lower()
        model = str(row["Model"]).strip().lower()
        device = str(row["Device"]).strip().lower()
        gt = _normalize_label(row["gt"])
    except Exception:
        return None
    if not user or not model or not device or not gt:
        return None
    return ts, x, y, z, user, model, device, gt


def _emit_windows(
    seq_xyz: List[Tuple[float, float, float]],
    *,
    label_idx: int,
    domain_id: int,
    split: str,
    split_storage: Dict[str, Dict[str, List[torch.Tensor]]],
    window_size: int,
    stride: int,
    max_windows_per_segment: int,
) -> int:
    if len(seq_xyz) < window_size:
        return 0
    start = 0
    emitted = 0
    total = len(seq_xyz)
    while start + window_size <= total:
        segment = seq_xyz[start : start + window_size]
        win = torch.tensor(segment, dtype=torch.float32).transpose(0, 1).contiguous()
        split_storage[split]["data"].append(win)
        split_storage[split]["labels"].append(torch.tensor(label_idx, dtype=torch.long))
        split_storage[split]["lengths"].append(torch.tensor(window_size, dtype=torch.long))
        split_storage[split]["domain_ids"].append(torch.tensor(domain_id, dtype=torch.long))
        emitted += 1
        if max_windows_per_segment > 0 and emitted >= max_windows_per_segment:
            break
        start += stride
    return emitted


def _build_split_payload(
    *,
    split: str,
    data: List[torch.Tensor],
    labels: List[torch.Tensor],
    lengths: List[torch.Tensor],
    domain_ids: List[torch.Tensor],
    class_labels: List[str],
    domain_labels: List[str],
    meta: Dict[str, object],
) -> Dict[str, object]:
    if not data:
        raise RuntimeError(f"No windows were generated for split='{split}'.")
    return {
        "version": 2,
        "dataset": "HHAR",
        "split": split,
        "data": torch.stack(data).to(dtype=torch.float32),
        "labels": torch.stack(labels).to(dtype=torch.long),
        "lengths": torch.stack(lengths).to(dtype=torch.long),
        "domain_ids": torch.stack(domain_ids).to(dtype=torch.long),
        "class_labels": class_labels,
        "domain_labels": domain_labels,
        "meta": meta,
    }


def _resolve_domain_token(args, source: str, user: str, model: str, device: str) -> str:
    if args.domain_by == "user":
        return f"user:{user}"
    if args.domain_by == "device":
        return f"device:{device}"
    if args.domain_by == "source":
        return f"source:{source}"
    return f"model:{model}"


def _process_streaming_csv(
    *,
    csv_path: Path,
    source_name: str,
    resolve_split,
    keep_labels: Dict[str, int],
    args,
    keep_users: Optional[Set[str]] = None,
    keep_models: Optional[Set[str]] = None,
) -> Dict[str, object]:
    split_storage: Dict[str, Dict[str, List[torch.Tensor]]] = {
        "train": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
        "test": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
    }
    domain_to_id: Dict[str, int] = {}
    domain_counts = defaultdict(int)
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    rows_seen = defaultdict(int)
    rows_kept = defaultdict(int)
    windows_emitted = defaultdict(int)

    target_split = resolve_split(source_name, None)
    if target_split is None and args.protocol == "natural":
        print(f"skip_source={source_name} reason=not_in_train_or_test")
        return {
            "split_storage": split_storage,
            "domain_to_id": domain_to_id,
            "domain_counts": domain_counts,
            "label_counts": label_counts,
            "source_counts": source_counts,
            "rows_seen": rows_seen,
            "rows_kept": rows_kept,
            "windows_emitted": windows_emitted,
        }

    print(f"processing_source={source_name} csv={csv_path}")
    current_key: Optional[Tuple[str, str, str, str, str]] = None
    current_domain_id: Optional[int] = None
    current_label_idx: Optional[int] = None
    current_seq_xyz: List[Tuple[float, float, float]] = []

    def _flush_current() -> None:
        nonlocal current_key, current_domain_id, current_label_idx, current_seq_xyz
        if current_key is None or current_domain_id is None or current_label_idx is None:
            current_key = None
            current_domain_id = None
            current_label_idx = None
            current_seq_xyz = []
            return
        emitted = _emit_windows(
            current_seq_xyz,
            label_idx=current_label_idx,
            domain_id=current_domain_id,
            split=current_key[0],
            split_storage=split_storage,
            window_size=args.window_size,
            stride=args.stride,
            max_windows_per_segment=args.max_windows_per_segment,
        )
        windows_emitted[current_key[0]] += emitted
        current_key = None
        current_domain_id = None
        current_label_idx = None
        current_seq_xyz = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            rows_seen[source_name] += 1
            if args.max_rows_per_file > 0 and row_idx > args.max_rows_per_file:
                break
            parsed = _parse_csv_row(row)
            if parsed is None:
                _flush_current()
                continue
            x, y, z, user, model, device, gt = parsed
            split = resolve_split(source_name, model)
            if split is None:
                _flush_current()
                continue

            if keep_users is not None and user not in keep_users:
                _flush_current()
                continue
            if keep_models is not None and model not in keep_models:
                _flush_current()
                continue
            if args.drop_null and gt == "null":
                _flush_current()
                continue
            label_idx = keep_labels.get(gt)
            if label_idx is None:
                _flush_current()
                continue
            rows_kept[source_name] += 1

            domain_token = _resolve_domain_token(args, source_name, user, model, device)
            if domain_token not in domain_to_id:
                domain_to_id[domain_token] = len(domain_to_id)
            domain_id = domain_to_id[domain_token]

            seg_key = (split, user, model, device, gt)
            if current_key is None:
                current_key = seg_key
                current_domain_id = domain_id
                current_label_idx = label_idx
            elif seg_key != current_key:
                _flush_current()
                current_key = seg_key
                current_domain_id = domain_id
                current_label_idx = label_idx

            current_seq_xyz.append((x, y, z))
            domain_counts[domain_token] += 1
            label_counts[gt] += 1
            source_counts[source_name] += 1

            if args.report_every_rows > 0 and (row_idx % args.report_every_rows == 0):
                print(
                    f"progress source={source_name} rows_seen={row_idx} "
                    f"rows_kept={rows_kept[source_name]} windows_train={windows_emitted['train']} "
                    f"windows_test={windows_emitted['test']}"
                )

    _flush_current()
    return {
        "split_storage": split_storage,
        "domain_to_id": domain_to_id,
        "domain_counts": domain_counts,
        "label_counts": label_counts,
        "source_counts": source_counts,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "windows_emitted": windows_emitted,
    }


def _collect_sensor_windows_by_group(
    *,
    csv_path: Path,
    source_name: str,
    sensor_tag: str,
    resolve_split,
    keep_labels: Set[str],
    args,
    keep_users: Optional[Set[str]] = None,
    keep_models: Optional[Set[str]] = None,
) -> Dict[str, object]:
    groups = defaultdict(list)  # key -> List[(start_ts, window_tensor(3,T))]
    rows_seen = 0
    rows_kept = 0
    segment_count = 0
    windows_count = 0

    current_key: Optional[Tuple[str, str, str, str, str, str]] = None
    current_seq: List[Tuple[int, float, float, float]] = []  # (ts, x, y, z)

    def _flush_segment() -> None:
        nonlocal current_key, current_seq, segment_count, windows_count
        if current_key is None:
            current_seq = []
            return
        if len(current_seq) >= args.window_size:
            start = 0
            emitted = 0
            total = len(current_seq)
            while start + args.window_size <= total:
                seg = current_seq[start : start + args.window_size]
                start_ts = int(seg[0][0])
                xyz = [(sx, sy, sz) for _, sx, sy, sz in seg]
                win = torch.tensor(xyz, dtype=torch.float32).transpose(0, 1).contiguous()
                groups[current_key].append((start_ts, win))
                emitted += 1
                if args.max_windows_per_segment > 0 and emitted >= args.max_windows_per_segment:
                    break
                start += args.stride
            windows_count += emitted
            segment_count += 1
        current_key = None
        current_seq = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            rows_seen += 1
            if args.max_rows_per_file > 0 and row_idx > args.max_rows_per_file:
                break
            parsed = _parse_csv_row_with_time(row)
            if parsed is None:
                _flush_segment()
                continue
            ts, x, y, z, user, model, device, gt = parsed
            split = resolve_split(source_name, model)
            if split is None:
                _flush_segment()
                continue
            if keep_users is not None and user not in keep_users:
                _flush_segment()
                continue
            if keep_models is not None and model not in keep_models:
                _flush_segment()
                continue
            if args.drop_null and gt == "null":
                _flush_segment()
                continue
            if gt not in keep_labels:
                _flush_segment()
                continue

            rows_kept += 1
            seg_key = (split, source_name, user, model, device, gt)
            if current_key is None:
                current_key = seg_key
            elif seg_key != current_key:
                _flush_segment()
                current_key = seg_key
            current_seq.append((ts, x, y, z))

            if args.report_every_rows > 0 and (row_idx % args.report_every_rows == 0):
                print(
                    f"progress sensor={sensor_tag} source={source_name} rows_seen={row_idx} "
                    f"rows_kept={rows_kept} segment_windows={windows_count}"
                )

    _flush_segment()
    return {
        "groups": groups,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "segments": segment_count,
        "windows": windows_count,
    }


def _fuse_acc_gyro_groups(
    *,
    acc_groups: Dict[Tuple[str, str, str, str, str, str], List[Tuple[int, torch.Tensor]]],
    gyro_groups: Dict[Tuple[str, str, str, str, str, str], List[Tuple[int, torch.Tensor]]],
    label_to_idx: Dict[str, int],
    args,
    split_storage: Dict[str, Dict[str, List[torch.Tensor]]],
    domain_to_id: Dict[str, int],
    domain_counts: Dict[str, int],
    label_counts: Dict[str, int],
    source_counts: Dict[str, int],
    windows_emitted: Dict[str, int],
) -> Dict[str, int]:
    paired_windows = 0
    dropped_acc_only = 0
    dropped_gyro_only = 0

    for key, acc_list in acc_groups.items():
        gyro_list = gyro_groups.get(key)
        if not gyro_list:
            dropped_acc_only += len(acc_list)
            continue
        acc_sorted = sorted(acc_list, key=lambda x: x[0])
        gyro_sorted = sorted(gyro_list, key=lambda x: x[0])
        n = min(len(acc_sorted), len(gyro_sorted))
        if len(acc_sorted) > n:
            dropped_acc_only += len(acc_sorted) - n
        if len(gyro_sorted) > n:
            dropped_gyro_only += len(gyro_sorted) - n

        split, source_name, user, model, device, gt = key
        label_idx = label_to_idx[gt]
        domain_token = _resolve_domain_token(args, source_name, user, model, device)
        if domain_token not in domain_to_id:
            domain_to_id[domain_token] = len(domain_to_id)
        domain_id = domain_to_id[domain_token]

        for idx in range(n):
            acc_win = acc_sorted[idx][1]
            gyro_win = gyro_sorted[idx][1]
            fused_win = torch.cat([acc_win, gyro_win], dim=0)
            split_storage[split]["data"].append(fused_win)
            split_storage[split]["labels"].append(torch.tensor(label_idx, dtype=torch.long))
            split_storage[split]["lengths"].append(torch.tensor(args.window_size, dtype=torch.long))
            split_storage[split]["domain_ids"].append(torch.tensor(domain_id, dtype=torch.long))
            paired_windows += 1
            windows_emitted[split] += 1
            domain_counts[domain_token] += 1
            label_counts[gt] += 1
            source_counts[source_name] += 1

    for key, gyro_list in gyro_groups.items():
        if key not in acc_groups:
            dropped_gyro_only += len(gyro_list)

    return {
        "paired_windows": paired_windows,
        "dropped_acc_only": dropped_acc_only,
        "dropped_gyro_only": dropped_gyro_only,
    }


def _merge_stats(dst: Dict[str, object], src: Dict[str, object]) -> None:
    local_to_global: Dict[int, int] = {}
    for domain_token, local_domain_id in src["domain_to_id"].items():
        if domain_token not in dst["domain_to_id"]:
            dst["domain_to_id"][domain_token] = len(dst["domain_to_id"])
        local_to_global[int(local_domain_id)] = int(dst["domain_to_id"][domain_token])

    for split in ("train", "test"):
        for key in ("data", "labels", "lengths"):
            dst["split_storage"][split][key].extend(src["split_storage"][split][key])
        for domain_id_tensor in src["split_storage"][split]["domain_ids"]:
            local_domain_id = int(domain_id_tensor.item())
            global_domain_id = local_to_global[local_domain_id]
            dst["split_storage"][split]["domain_ids"].append(torch.tensor(global_domain_id, dtype=torch.long))

    for name in ("domain_counts", "label_counts", "source_counts", "rows_seen", "rows_kept", "windows_emitted"):
        for key, value in src[name].items():
            dst[name][key] += value


def _build_common_accumulators() -> Dict[str, object]:
    return {
        "split_storage": {
            "train": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
            "test": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
        },
        "domain_to_id": {},
        "domain_counts": defaultdict(int),
        "label_counts": defaultdict(int),
        "source_counts": defaultdict(int),
        "rows_seen": defaultdict(int),
        "rows_kept": defaultdict(int),
        "windows_emitted": defaultdict(int),
    }


def _run_natural(args, keep_labels: List[str], label_to_idx: Dict[str, int], file_map: Dict[str, Path]) -> Dict[str, object]:
    train_sources = set(_split_csv_arg(args.train_sources))
    test_sources = set(_split_csv_arg(args.test_sources))
    if not train_sources or not test_sources:
        raise ValueError("--train-sources and --test-sources must be non-empty.")
    if train_sources & test_sources:
        raise ValueError("train_sources and test_sources must be disjoint.")

    def resolve_split(source_name: str, _model: Optional[str]) -> Optional[str]:
        if source_name in train_sources:
            return "train"
        if source_name in test_sources:
            return "test"
        return None

    accum = _build_common_accumulators()
    for source_name, csv_path in file_map.items():
        stats = _process_streaming_csv(
            csv_path=csv_path,
            source_name=source_name,
            resolve_split=resolve_split,
            keep_labels=label_to_idx,
            args=args,
        )
        _merge_stats(accum, stats)

    domain_labels = [token for token, _ in sorted(accum["domain_to_id"].items(), key=lambda kv: kv[1])]
    meta = {
        "protocol": "natural",
        "window_size": args.window_size,
        "stride": args.stride,
        "sensor": args.sensor,
        "drop_null": bool(args.drop_null),
        "keep_labels": keep_labels,
        "domain_by": args.domain_by,
        "train_sources": sorted(train_sources),
        "test_sources": sorted(test_sources),
        "rows_seen": dict(accum["rows_seen"]),
        "rows_kept": dict(accum["rows_kept"]),
        "source_kept_counts": dict(accum["source_counts"]),
        "label_counts": dict(accum["label_counts"]),
        "domain_counts": dict(accum["domain_counts"]),
        "num_domains": len(domain_labels),
    }
    return {
        "meta": meta,
        "domain_labels": domain_labels,
        "split_storage": accum["split_storage"],
    }


def _run_woods(args, keep_labels: List[str], label_to_idx: Dict[str, int], file_map: Dict[str, Path]) -> Dict[str, object]:
    woods_users = set(_split_csv_arg(args.woods_users))
    woods_models = set(_split_csv_arg(args.woods_models))
    if not woods_users:
        raise ValueError("--woods-users must be non-empty.")
    if not woods_models:
        raise ValueError("--woods-models must be non-empty.")
    target_model = str(args.woods_target_model).strip().lower()
    if not target_model:
        raise ValueError("--woods-target-model is required when --protocol woods.")
    if target_model not in woods_models:
        raise ValueError(f"--woods-target-model '{target_model}' must be included in --woods-models.")

    source_models = sorted([model for model in woods_models if model != target_model])
    if not source_models:
        raise ValueError("WOODS split needs at least one source model.")
    print(
        "woods_protocol=1 "
        f"users={','.join(sorted(woods_users))} "
        f"models={','.join(sorted(woods_models))} "
        f"target_model={target_model}"
    )

    def resolve_split(_source_name: str, model: Optional[str]) -> Optional[str]:
        if model is None or model not in woods_models:
            return None
        if model == target_model:
            return "test"
        return "train"

    accum = _build_common_accumulators()
    for source_name, csv_path in file_map.items():
        stats = _process_streaming_csv(
            csv_path=csv_path,
            source_name=source_name,
            resolve_split=resolve_split,
            keep_labels=label_to_idx,
            args=args,
            keep_users=woods_users,
            keep_models=woods_models,
        )
        _merge_stats(accum, stats)

    domain_labels = [token for token, _ in sorted(accum["domain_to_id"].items(), key=lambda kv: kv[1])]
    source_domains = sorted([token for token in domain_labels if token != f"model:{target_model}"])
    meta = {
        "protocol": "woods",
        "window_size": args.window_size,
        "stride": args.stride,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "sample_rate_hz": args.sample_rate,
        "sensor": args.sensor,
        "drop_null": bool(args.drop_null),
        "keep_labels": keep_labels,
        "domain_by": args.domain_by,
        "woods_users": sorted(woods_users),
        "woods_models": sorted(woods_models),
        "woods_source_models": source_models,
        "woods_target_model": target_model,
        "rows_seen": dict(accum["rows_seen"]),
        "rows_kept": dict(accum["rows_kept"]),
        "source_kept_counts": dict(accum["source_counts"]),
        "label_counts": dict(accum["label_counts"]),
        "domain_counts": dict(accum["domain_counts"]),
        "num_domains": len(domain_labels),
        "num_source_domains": len(source_domains),
    }
    return {
        "meta": meta,
        "domain_labels": domain_labels,
        "split_storage": accum["split_storage"],
    }


def _run_natural_fused(
    args,
    keep_labels: List[str],
    label_to_idx: Dict[str, int],
    file_map_acc: Dict[str, Path],
    file_map_gyro: Dict[str, Path],
) -> Dict[str, object]:
    train_sources = set(_split_csv_arg(args.train_sources))
    test_sources = set(_split_csv_arg(args.test_sources))
    if not train_sources or not test_sources:
        raise ValueError("--train-sources and --test-sources must be non-empty.")
    if train_sources & test_sources:
        raise ValueError("train_sources and test_sources must be disjoint.")

    def resolve_split(source_name: str, _model: Optional[str]) -> Optional[str]:
        if source_name in train_sources:
            return "train"
        if source_name in test_sources:
            return "test"
        return None

    split_storage: Dict[str, Dict[str, List[torch.Tensor]]] = {
        "train": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
        "test": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
    }
    domain_to_id: Dict[str, int] = {}
    domain_counts = defaultdict(int)
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    windows_emitted = defaultdict(int)
    rows_seen = defaultdict(int)
    rows_kept = defaultdict(int)
    sensor_windows = defaultdict(int)
    align_stats = defaultdict(int)

    keep_label_set = set(keep_labels)
    for source_name in ("phone", "watch"):
        if source_name not in file_map_acc or source_name not in file_map_gyro:
            continue
        acc_stats = _collect_sensor_windows_by_group(
            csv_path=file_map_acc[source_name],
            source_name=source_name,
            sensor_tag="acc",
            resolve_split=resolve_split,
            keep_labels=keep_label_set,
            args=args,
        )
        gyro_stats = _collect_sensor_windows_by_group(
            csv_path=file_map_gyro[source_name],
            source_name=source_name,
            sensor_tag="gyro",
            resolve_split=resolve_split,
            keep_labels=keep_label_set,
            args=args,
        )
        rows_seen[f"{source_name}_acc"] += int(acc_stats["rows_seen"])
        rows_kept[f"{source_name}_acc"] += int(acc_stats["rows_kept"])
        rows_seen[f"{source_name}_gyro"] += int(gyro_stats["rows_seen"])
        rows_kept[f"{source_name}_gyro"] += int(gyro_stats["rows_kept"])
        sensor_windows[f"{source_name}_acc"] += int(acc_stats["windows"])
        sensor_windows[f"{source_name}_gyro"] += int(gyro_stats["windows"])

        fused = _fuse_acc_gyro_groups(
            acc_groups=acc_stats["groups"],
            gyro_groups=gyro_stats["groups"],
            label_to_idx=label_to_idx,
            args=args,
            split_storage=split_storage,
            domain_to_id=domain_to_id,
            domain_counts=domain_counts,
            label_counts=label_counts,
            source_counts=source_counts,
            windows_emitted=windows_emitted,
        )
        for key, value in fused.items():
            align_stats[key] += int(value)

    domain_labels = [token for token, _ in sorted(domain_to_id.items(), key=lambda kv: kv[1])]
    meta = {
        "protocol": "natural",
        "window_size": args.window_size,
        "stride": args.stride,
        "sensor": "both",
        "drop_null": bool(args.drop_null),
        "keep_labels": keep_labels,
        "domain_by": args.domain_by,
        "train_sources": sorted(train_sources),
        "test_sources": sorted(test_sources),
        "rows_seen": dict(rows_seen),
        "rows_kept": dict(rows_kept),
        "sensor_windows": dict(sensor_windows),
        "alignment": dict(align_stats),
        "source_kept_counts": dict(source_counts),
        "label_counts": dict(label_counts),
        "domain_counts": dict(domain_counts),
        "num_domains": len(domain_labels),
        "num_channels": 6,
    }
    return {
        "meta": meta,
        "domain_labels": domain_labels,
        "split_storage": split_storage,
    }


def _run_woods_fused(
    args,
    keep_labels: List[str],
    label_to_idx: Dict[str, int],
    file_map_acc: Dict[str, Path],
    file_map_gyro: Dict[str, Path],
) -> Dict[str, object]:
    woods_users = set(_split_csv_arg(args.woods_users))
    woods_models = set(_split_csv_arg(args.woods_models))
    if not woods_users:
        raise ValueError("--woods-users must be non-empty.")
    if not woods_models:
        raise ValueError("--woods-models must be non-empty.")
    target_model = str(args.woods_target_model).strip().lower()
    if not target_model:
        raise ValueError("--woods-target-model is required when --protocol woods.")
    if target_model not in woods_models:
        raise ValueError(f"--woods-target-model '{target_model}' must be included in --woods-models.")
    source_models = sorted([model for model in woods_models if model != target_model])
    if not source_models:
        raise ValueError("WOODS split needs at least one source model.")
    print(
        "woods_protocol=1 "
        f"users={','.join(sorted(woods_users))} "
        f"models={','.join(sorted(woods_models))} "
        f"target_model={target_model} sensor=both"
    )

    def resolve_split(_source_name: str, model: Optional[str]) -> Optional[str]:
        if model is None or model not in woods_models:
            return None
        if model == target_model:
            return "test"
        return "train"

    split_storage: Dict[str, Dict[str, List[torch.Tensor]]] = {
        "train": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
        "test": {"data": [], "labels": [], "lengths": [], "domain_ids": []},
    }
    domain_to_id: Dict[str, int] = {}
    domain_counts = defaultdict(int)
    label_counts = defaultdict(int)
    source_counts = defaultdict(int)
    windows_emitted = defaultdict(int)
    rows_seen = defaultdict(int)
    rows_kept = defaultdict(int)
    sensor_windows = defaultdict(int)
    align_stats = defaultdict(int)

    keep_label_set = set(keep_labels)
    for source_name in ("phone", "watch"):
        if source_name not in file_map_acc or source_name not in file_map_gyro:
            continue
        acc_stats = _collect_sensor_windows_by_group(
            csv_path=file_map_acc[source_name],
            source_name=source_name,
            sensor_tag="acc",
            resolve_split=resolve_split,
            keep_labels=keep_label_set,
            args=args,
            keep_users=woods_users,
            keep_models=woods_models,
        )
        gyro_stats = _collect_sensor_windows_by_group(
            csv_path=file_map_gyro[source_name],
            source_name=source_name,
            sensor_tag="gyro",
            resolve_split=resolve_split,
            keep_labels=keep_label_set,
            args=args,
            keep_users=woods_users,
            keep_models=woods_models,
        )
        rows_seen[f"{source_name}_acc"] += int(acc_stats["rows_seen"])
        rows_kept[f"{source_name}_acc"] += int(acc_stats["rows_kept"])
        rows_seen[f"{source_name}_gyro"] += int(gyro_stats["rows_seen"])
        rows_kept[f"{source_name}_gyro"] += int(gyro_stats["rows_kept"])
        sensor_windows[f"{source_name}_acc"] += int(acc_stats["windows"])
        sensor_windows[f"{source_name}_gyro"] += int(gyro_stats["windows"])

        fused = _fuse_acc_gyro_groups(
            acc_groups=acc_stats["groups"],
            gyro_groups=gyro_stats["groups"],
            label_to_idx=label_to_idx,
            args=args,
            split_storage=split_storage,
            domain_to_id=domain_to_id,
            domain_counts=domain_counts,
            label_counts=label_counts,
            source_counts=source_counts,
            windows_emitted=windows_emitted,
        )
        for key, value in fused.items():
            align_stats[key] += int(value)

    domain_labels = [token for token, _ in sorted(domain_to_id.items(), key=lambda kv: kv[1])]
    source_domains = sorted([token for token in domain_labels if token != f"model:{target_model}"])
    meta = {
        "protocol": "woods",
        "window_size": args.window_size,
        "stride": args.stride,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "sample_rate_hz": args.sample_rate,
        "sensor": "both",
        "drop_null": bool(args.drop_null),
        "keep_labels": keep_labels,
        "domain_by": args.domain_by,
        "woods_users": sorted(woods_users),
        "woods_models": sorted(woods_models),
        "woods_source_models": source_models,
        "woods_target_model": target_model,
        "rows_seen": dict(rows_seen),
        "rows_kept": dict(rows_kept),
        "sensor_windows": dict(sensor_windows),
        "alignment": dict(align_stats),
        "source_kept_counts": dict(source_counts),
        "label_counts": dict(label_counts),
        "domain_counts": dict(domain_counts),
        "num_domains": len(domain_labels),
        "num_source_domains": len(source_domains),
        "num_channels": 6,
    }
    return {
        "meta": meta,
        "domain_labels": domain_labels,
        "split_storage": split_storage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess HHAR Activity-recognition CSV into windowed train/test tensors. "
            "Supports natural split (phone->train/watch->test) and WOODS-style model leave-one-domain-out split."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(r"d:\TFproject\time-main\dataset\all_datasets\Heterogeneity Activity Recognition"),
        help="Root directory of HHAR raw dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"d:\TFproject\time-main\dataset\all_datasets\HHAR"),
        help="Output directory for HHAR_TRAIN.pt / HHAR_TEST.pt.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="natural",
        choices=["natural", "woods"],
        help="Preprocessing protocol: natural source split or WOODS-style model split.",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="accelerometer",
        choices=["accelerometer", "gyroscope", "both"],
        help="Sensor csv to preprocess. Use 'both' to align acc+gyro and emit 6-channel windows.",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Window size in samples (natural protocol).")
    parser.add_argument("--stride", type=int, default=64, help="Stride in samples (natural protocol).")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Window length in seconds (WOODS protocol).",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=5.0,
        help="Stride in seconds (WOODS protocol).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=100,
        help="Sampling rate used to convert WOODS window/stride seconds into samples.",
    )
    parser.add_argument(
        "--keep-labels",
        type=str,
        default=",".join(DEFAULT_KEEP_LABELS),
        help="Comma-separated labels to keep from gt column.",
    )
    null_group = parser.add_mutually_exclusive_group()
    null_group.add_argument(
        "--drop-null",
        dest="drop_null",
        action="store_true",
        default=True,
        help="Drop rows with gt=null.",
    )
    null_group.add_argument(
        "--keep-null",
        dest="drop_null",
        action="store_false",
        help="Keep rows with gt=null (only if included in --keep-labels).",
    )
    parser.add_argument(
        "--domain-by",
        type=str,
        default="model",
        choices=["model", "device", "user", "source"],
        help="Domain id granularity used for DG metadata.",
    )
    parser.add_argument(
        "--train-sources",
        type=str,
        default="phone",
        help="Comma-separated source names for train split in natural protocol (phone/watch).",
    )
    parser.add_argument(
        "--test-sources",
        type=str,
        default="watch",
        help="Comma-separated source names for test split in natural protocol (phone/watch).",
    )
    parser.add_argument(
        "--woods-users",
        type=str,
        default=",".join(WOODS_DEFAULT_USERS),
        help="WOODS protocol users subset.",
    )
    parser.add_argument(
        "--woods-models",
        type=str,
        default=",".join(WOODS_DEFAULT_MODELS),
        help="WOODS protocol device-model domains subset.",
    )
    parser.add_argument(
        "--woods-target-model",
        type=str,
        default="",
        help="WOODS protocol target model held out for test.",
    )
    parser.add_argument(
        "--max-windows-per-segment",
        type=int,
        default=0,
        help="Optional cap per contiguous segment (0 means unlimited).",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional debug cap for rows read from each file (0 means full file).",
    )
    parser.add_argument(
        "--report-every-rows",
        type=int,
        default=500000,
        help="Progress print interval while reading each csv.",
    )
    args = parser.parse_args()

    if args.max_windows_per_segment < 0:
        raise ValueError("--max-windows-per-segment must be >= 0.")
    if args.max_rows_per_file < 0:
        raise ValueError("--max-rows-per-file must be >= 0.")
    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.window_seconds <= 0.0:
        raise ValueError("--window-seconds must be positive.")
    if args.stride_seconds <= 0.0:
        raise ValueError("--stride-seconds must be positive.")
    if args.sample_rate < 1:
        raise ValueError("--sample-rate must be >= 1.")

    keep_labels = _split_csv_arg(args.keep_labels)
    if args.drop_null and "null" in keep_labels:
        keep_labels = [item for item in keep_labels if item != "null"]
    if not keep_labels:
        raise ValueError("No labels selected after parsing --keep-labels.")
    class_labels = sorted(set(keep_labels))
    label_to_idx = {name: idx for idx, name in enumerate(class_labels)}

    if args.protocol == "woods":
        args.window_size = int(round(args.window_seconds * float(args.sample_rate)))
        args.stride = int(round(args.stride_seconds * float(args.sample_rate)))
        if args.window_size < 2:
            raise ValueError("WOODS-derived --window-size must be >= 2.")
        if args.stride < 1:
            raise ValueError("WOODS-derived --stride must be >= 1.")

    activity_dir = args.raw_root / "Activity recognition exp"
    if args.sensor == "both":
        file_map_acc = {
            "phone": activity_dir / "Phones_accelerometer.csv",
            "watch": activity_dir / "Watch_accelerometer.csv",
        }
        file_map_gyro = {
            "phone": activity_dir / "Phones_gyroscope.csv",
            "watch": activity_dir / "Watch_gyroscope.csv",
        }
        for source, path in {**file_map_acc, **file_map_gyro}.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing HHAR csv for source='{source}': {path}")
        if args.protocol == "woods":
            out = _run_woods_fused(args, class_labels, label_to_idx, file_map_acc, file_map_gyro)
        else:
            out = _run_natural_fused(args, class_labels, label_to_idx, file_map_acc, file_map_gyro)
    else:
        file_map = {
            "phone": activity_dir / f"Phones_{args.sensor}.csv",
            "watch": activity_dir / f"Watch_{args.sensor}.csv",
        }
        for source, path in file_map.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing HHAR csv for source='{source}': {path}")
        if args.protocol == "woods":
            out = _run_woods(args, class_labels, label_to_idx, file_map)
        else:
            out = _run_natural(args, class_labels, label_to_idx, file_map)

    payload_train = _build_split_payload(
        split="train",
        data=out["split_storage"]["train"]["data"],
        labels=out["split_storage"]["train"]["labels"],
        lengths=out["split_storage"]["train"]["lengths"],
        domain_ids=out["split_storage"]["train"]["domain_ids"],
        class_labels=class_labels,
        domain_labels=out["domain_labels"],
        meta={**out["meta"], "split": "train"},
    )
    payload_test = _build_split_payload(
        split="test",
        data=out["split_storage"]["test"]["data"],
        labels=out["split_storage"]["test"]["labels"],
        lengths=out["split_storage"]["test"]["lengths"],
        domain_ids=out["split_storage"]["test"]["domain_ids"],
        class_labels=class_labels,
        domain_labels=out["domain_labels"],
        meta={**out["meta"], "split": "test"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "HHAR_TRAIN.pt"
    test_path = args.output_dir / "HHAR_TEST.pt"
    meta_path = args.output_dir / "HHAR_meta.json"
    torch.save(payload_train, train_path)
    torch.save(payload_test, test_path)
    meta_path.write_text(json.dumps(out["meta"], indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"saved_train={train_path}")
    print(f"saved_test={test_path}")
    print(f"saved_meta={meta_path}")
    print(
        f"summary protocol={args.protocol} train_windows={payload_train['data'].shape[0]} "
        f"test_windows={payload_test['data'].shape[0]} num_classes={len(class_labels)} "
        f"num_domains={len(out['domain_labels'])}"
    )


if __name__ == "__main__":
    main()
