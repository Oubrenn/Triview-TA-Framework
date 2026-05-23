import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_NFFT = {
    "Handwriting": [16, 32, 64],
    "UWaveGestureLibrary": [128, 256, 512],
    "HHAR": [32, 64, 128],
    "FaceDetection": [8, 16, 32],
}

VARIANT_ARGS: Dict[str, List[str]] = {
    "baseline": [
        "--lambda-md",
        "0.0",
        "--lambda-ta",
        "0.0",
        "--lambda-shift",
        "0.0",
        "--lambda-scale",
        "0.0",
        "--lambda-color",
        "0.0",
        "--lambda-attn",
        "0.0",
    ],
    "triview": [
        "--lambda-md",
        "1.0",
        "--lambda-ta",
        "0.0",
        "--lambda-shift",
        "0.0",
        "--lambda-scale",
        "0.0",
        "--lambda-color",
        "0.0",
        "--lambda-attn",
        "0.0",
    ],
    "full": [
        "--lambda-md",
        "1.0",
        "--lambda-ta",
        "1.0",
        "--lambda-shift",
        "1.0",
        "--lambda-scale",
        "1.0",
        "--lambda-color",
        "1.0",
        "--lambda-attn",
        "1.0",
    ],
}

PER_SEED_FIELDS = [
    "status",
    "dataset",
    "seed",
    "variant",
    "n_fft",
    "win_length",
    "hop_length",
    "aggregate_scope",
    "clean_acc",
    "avg_acc",
    "worst_acc",
    "drop_acc",
    "clean_mf1",
    "avg_mf1",
    "worst_mf1",
    "drop_mf1",
    "train_return_code",
    "sweep_return_code",
    "checkpoint_path",
    "train_log_path",
    "sweep_log_path",
    "summary_csv",
    "robustness_csv",
]

SUMMARY_FIELDS = [
    "dataset",
    "variant",
    "n_fft",
    "win_length",
    "hop_length",
    "aggregate_scope",
    "num_runs",
    "seed_list",
    "clean_acc_mean",
    "clean_acc_std",
    "avg_acc_mean",
    "avg_acc_std",
    "worst_acc_mean",
    "worst_acc_std",
    "drop_acc_mean",
    "drop_acc_std",
    "clean_mf1_mean",
    "clean_mf1_std",
    "avg_mf1_mean",
    "avg_mf1_std",
    "worst_mf1_mean",
    "worst_mf1_std",
    "drop_mf1_mean",
    "drop_mf1_std",
]


def _parse_csv_str(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_int(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_dataset_nfft_overrides(raw: str) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    if not raw.strip():
        return out
    for chunk in [c.strip() for c in raw.split(";") if c.strip()]:
        if ":" not in chunk:
            raise ValueError(f"Invalid --dataset-nfft-overrides chunk: {chunk}")
        dataset, values = chunk.split(":", maxsplit=1)
        vals = sorted(set(_parse_csv_int(values)))
        if not dataset.strip() or not vals:
            raise ValueError(f"Invalid --dataset-nfft-overrides chunk: {chunk}")
        out[dataset.strip()] = vals
    return out


def _split_extra_args(raw: str) -> List[str]:
    return shlex.split(raw, posix=False) if raw.strip() else []


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_mean(values: List[float]) -> float:
    good = [v for v in values if not math.isnan(v)]
    return float(sum(good) / len(good)) if good else float("nan")


def _safe_std(values: List[float]) -> float:
    good = [v for v in values if not math.isnan(v)]
    return float(statistics.stdev(good)) if len(good) > 1 else 0.0


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_command(cmd: List[str], log_path: Path) -> int:
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return int(completed.returncode)


def _find_best_checkpoint(save_dir: Path, run_name: str) -> Optional[Path]:
    candidates = sorted(save_dir.glob(f"{run_name}_ep*_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_sweep_paths(sweep_output_root: Path) -> Tuple[Path, Path]:
    meta_path = sweep_output_root / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Sweep meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    extra = meta.get("extra", {})
    summary = str(extra.get("summary_csv", "")).strip()
    robust = str(extra.get("robustness_csv", "")).strip()
    if not summary or not robust:
        raise ValueError(f"Missing summary/robustness path in: {meta_path}")
    return _resolve_path(summary), _resolve_path(robust)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_scope_metrics(rows: List[Dict[str, str]], checkpoint: str, scope: str) -> Dict[str, float]:
    rows_ckpt = [r for r in rows if str(r.get("checkpoint", "")) == checkpoint]
    rows_clean = [r for r in rows_ckpt if str(r.get("transform", "")) == "clean"]
    if scope == "all_perturb":
        rows_perturb = [r for r in rows_ckpt if str(r.get("transform", "")) != "clean"]
    else:
        rows_perturb = [r for r in rows_ckpt if str(r.get("transform", "")) == scope]
    clean_acc = _safe_mean([_safe_float(r.get("acc")) for r in rows_clean])
    clean_mf1 = _safe_mean([_safe_float(r.get("mf1")) for r in rows_clean])
    p_acc = [v for v in [_safe_float(r.get("acc")) for r in rows_perturb] if not math.isnan(v)]
    p_mf1 = [v for v in [_safe_float(r.get("mf1")) for r in rows_perturb] if not math.isnan(v)]
    avg_acc = _safe_mean(p_acc)
    avg_mf1 = _safe_mean(p_mf1)
    worst_acc = min(p_acc) if p_acc else float("nan")
    worst_mf1 = min(p_mf1) if p_mf1 else float("nan")
    drop_acc = clean_acc - avg_acc if not (math.isnan(clean_acc) or math.isnan(avg_acc)) else float("nan")
    drop_mf1 = clean_mf1 - avg_mf1 if not (math.isnan(clean_mf1) or math.isnan(avg_mf1)) else float("nan")
    return {
        "clean_acc": clean_acc,
        "avg_acc": avg_acc,
        "worst_acc": worst_acc,
        "drop_acc": drop_acc,
        "clean_mf1": clean_mf1,
        "avg_mf1": avg_mf1,
        "worst_mf1": worst_mf1,
        "drop_mf1": drop_mf1,
    }


def _aggregate_ok_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int, int, int, str], List[Dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        key = (
            str(row["dataset"]),
            str(row["variant"]),
            int(row["n_fft"]),
            int(row["win_length"]),
            int(row["hop_length"]),
            str(row["aggregate_scope"]),
        )
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, object]] = []
    metric_fields = ["clean_acc", "avg_acc", "worst_acc", "drop_acc", "clean_mf1", "avg_mf1", "worst_mf1", "drop_mf1"]
    for key, members in sorted(groups.items()):
        dataset, variant, n_fft, win_length, hop_length, scope = key
        seeds = sorted(int(m["seed"]) for m in members)
        row: Dict[str, object] = {
            "dataset": dataset,
            "variant": variant,
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "aggregate_scope": scope,
            "num_runs": len(members),
            "seed_list": ",".join(str(s) for s in seeds),
        }
        for field in metric_fields:
            vals = [_safe_float(m.get(field)) for m in members]
            row[f"{field}_mean"] = _safe_mean(vals)
            row[f"{field}_std"] = _safe_std(vals)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="STFT sensitivity runner: train + sweep + table aggregation.")
    parser.add_argument("--datasets", type=str, default="Handwriting,UWaveGestureLibrary,HHAR")
    parser.add_argument("--include-facedetection", action="store_true", default=False)
    parser.add_argument("--dataset-nfft-overrides", type=str, default="")
    parser.add_argument("--variants", type=str, default="baseline,triview,full")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=ROOT / "src" / "train_uea.py")
    parser.add_argument("--sweep-script", type=Path, default=ROOT / "scripts" / "sweep_transforms.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_46" / "stft_sensitivity")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sweep-device", type=str, default="")
    parser.add_argument("--dataset-profile", type=str, default="none")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--val-split-mode", type=str, default="auto")
    parser.add_argument("--checkpoint-metric", type=str, default="val_mf1")
    parser.add_argument("--save-top-k", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="all")
    parser.add_argument("--use-temporal-attn", action="store_true", default=False)
    parser.add_argument("--use-shared-qk-attn", action="store_true", default=False)
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
    parser.add_argument("--stft-window", type=str, default="hann", choices=["hann", "hamming"])
    parser.add_argument("--stft-center", action="store_true", default=True)
    parser.add_argument("--no-stft-center", dest="stft_center", action="store_false")
    parser.add_argument("--stft-magnitude-power", type=float, default=1.0)
    parser.add_argument("--sweep-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--severity-source", type=str, default="train", choices=["fixed", "train", "val"])
    parser.add_argument("--aggregate-scope", type=str, default="all_perturb", choices=["all_perturb", "shift", "scale", "color", "mixed_shift_color"])
    parser.add_argument("--shift-bins", type=str, default="-0.2,-0.1,0,0.1,0.2")
    parser.add_argument("--scale-ratios", type=str, default="0.9,1,1.1")
    parser.add_argument("--color-max-db", type=str, default="0,3,6")
    parser.add_argument("--enable-mixed", action="store_true", default=False)
    parser.add_argument("--enforce-same-budget", action="store_true", default=True)
    parser.add_argument("--no-enforce-same-budget", dest="enforce_same_budget", action="store_false")
    parser.add_argument("--train-common-args", type=str, default="")
    parser.add_argument("--sweep-common-args", type=str, default="")
    parser.add_argument("--resume-existing", action="store_true", default=True)
    parser.add_argument("--no-resume-existing", dest="resume_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    sweep_device = args.sweep_device.strip() or args.device
    datasets = _parse_csv_str(args.datasets)
    if args.include_facedetection and "FaceDetection" not in datasets:
        datasets.append("FaceDetection")
    variants = _parse_csv_str(args.variants)
    seeds = _parse_csv_int(args.seeds)
    if not datasets or not variants or not seeds:
        raise ValueError("--datasets/--variants/--seeds must be non-empty.")
    for v in variants:
        if v not in VARIANT_ARGS:
            raise ValueError(f"Unsupported variant: {v}")

    overrides = _parse_dataset_nfft_overrides(args.dataset_nfft_overrides)
    dataset_nfft: Dict[str, List[int]] = {}
    for dataset in datasets:
        values = overrides.get(dataset, DEFAULT_DATASET_NFFT.get(dataset))
        if not values:
            raise ValueError(f"No n_fft preset for dataset={dataset}")
        values = sorted(set(int(v) for v in values))
        for n_fft in values:
            if n_fft <= 0 or n_fft % 4 != 0:
                raise ValueError(f"Invalid n_fft for dataset={dataset}: {n_fft}")
        dataset_nfft[dataset] = values

    train_extra = _split_extra_args(args.train_common_args)
    sweep_extra = _split_extra_args(args.sweep_common_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for n_fft in dataset_nfft[dataset]:
            hop = n_fft // 4
            for seed in seeds:
                pending: Dict[str, Dict[str, object]] = {}
                for variant in variants:
                    run_name = f"stftsens_{dataset}_{variant}_nfft{n_fft}_seed{seed}"
                    save_dir = args.output_dir / "checkpoints" / dataset / f"nfft{n_fft}" / f"seed{seed}" / variant
                    train_log = args.output_dir / "logs" / "train" / dataset / f"nfft{n_fft}" / f"{variant}_seed{seed}.log"
                    sweep_output_root = args.output_dir / "sweeps" / dataset / f"nfft{n_fft}_hop{hop}" / f"seed{seed}" / variant
                    sweep_log = args.output_dir / "logs" / "sweep" / dataset / f"nfft{n_fft}_{variant}_seed{seed}.log"

                    if args.resume_existing and (sweep_output_root / "run_meta.json").exists():
                        try:
                            summary_csv_cached, robustness_csv_cached = _load_sweep_paths(sweep_output_root)
                            summary_rows_cached = _read_csv(summary_csv_cached)
                            m_cached = _extract_scope_metrics(summary_rows_cached, checkpoint=variant, scope=args.aggregate_scope)
                            cached_ckpt = _find_best_checkpoint(save_dir, run_name)
                            rows.append(
                                {
                                    "status": "ok",
                                    "dataset": dataset,
                                    "seed": seed,
                                    "variant": variant,
                                    "n_fft": n_fft,
                                    "win_length": n_fft,
                                    "hop_length": hop,
                                    "aggregate_scope": args.aggregate_scope,
                                    "clean_acc": m_cached["clean_acc"],
                                    "avg_acc": m_cached["avg_acc"],
                                    "worst_acc": m_cached["worst_acc"],
                                    "drop_acc": m_cached["drop_acc"],
                                    "clean_mf1": m_cached["clean_mf1"],
                                    "avg_mf1": m_cached["avg_mf1"],
                                    "worst_mf1": m_cached["worst_mf1"],
                                    "drop_mf1": m_cached["drop_mf1"],
                                    "train_return_code": 0,
                                    "sweep_return_code": 0,
                                    "checkpoint_path": str(cached_ckpt) if cached_ckpt is not None else "",
                                    "train_log_path": str(train_log),
                                    "sweep_log_path": str(sweep_log),
                                    "summary_csv": str(summary_csv_cached),
                                    "robustness_csv": str(robustness_csv_cached),
                                }
                            )
                            continue
                        except Exception:
                            pass

                    if args.resume_existing:
                        existing_ckpt = _find_best_checkpoint(save_dir, run_name)
                        if existing_ckpt is not None:
                            pending[variant] = {
                                "checkpoint_path": existing_ckpt,
                                "train_log_path": train_log,
                                "train_return_code": 0,
                            }
                            continue

                    train_cmd = [
                        args.python_exec,
                        str(args.train_script),
                        "--dataset",
                        dataset,
                        "--dataset-profile",
                        args.dataset_profile,
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--run-name",
                        run_name,
                        "--save-dir",
                        str(save_dir),
                        "--epochs",
                        str(args.epochs),
                        "--finetune-epochs",
                        str(args.finetune_epochs),
                        "--pretrain-epochs",
                        str(args.pretrain_epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                        "--eval-num-workers",
                        str(args.eval_num_workers),
                        "--val-split",
                        str(args.val_split),
                        "--val-split-mode",
                        args.val_split_mode,
                        "--checkpoint-metric",
                        args.checkpoint_metric,
                        "--save-top-k",
                        str(args.save_top_k),
                        "--backbone",
                        args.backbone,
                        "--stft-window",
                        args.stft_window,
                        "--stft-magnitude-power",
                        str(args.stft_magnitude_power),
                        "--n-fft",
                        str(n_fft),
                        "--hop-length",
                        str(hop),
                        "--stft-win-length",
                        str(n_fft),
                    ]
                    train_cmd.append("--stft-center" if args.stft_center else "--no-stft-center")
                    if args.use_temporal_attn:
                        train_cmd.append("--use-temporal-attn")
                    if args.use_shared_qk_attn:
                        train_cmd.extend(["--use-shared-qk-attn", "--shared-qk-heads", str(args.shared_qk_heads), "--shared-qk-dropout", str(args.shared_qk_dropout)])
                    train_cmd.extend(train_extra)
                    train_cmd.extend(VARIANT_ARGS[variant])
                    if args.dry_run:
                        print("[dry-run][train]", " ".join(train_cmd))
                        continue
                    rc = _run_command(train_cmd, train_log)
                    ckpt = _find_best_checkpoint(save_dir, run_name) if rc == 0 else None
                    if rc != 0 or ckpt is None:
                        rows.append(
                            {
                                "status": "train_failed" if rc != 0 else "checkpoint_missing",
                                "dataset": dataset,
                                "seed": seed,
                                "variant": variant,
                                "n_fft": n_fft,
                                "win_length": n_fft,
                                "hop_length": hop,
                                "aggregate_scope": args.aggregate_scope,
                                "clean_acc": float("nan"),
                                "avg_acc": float("nan"),
                                "worst_acc": float("nan"),
                                "drop_acc": float("nan"),
                                "clean_mf1": float("nan"),
                                "avg_mf1": float("nan"),
                                "worst_mf1": float("nan"),
                                "drop_mf1": float("nan"),
                                "train_return_code": rc,
                                "sweep_return_code": "",
                                "checkpoint_path": str(ckpt) if ckpt is not None else "",
                                "train_log_path": str(train_log),
                                "sweep_log_path": "",
                                "summary_csv": "",
                                "robustness_csv": "",
                            }
                        )
                    else:
                        pending[variant] = {"checkpoint_path": ckpt, "train_log_path": train_log, "train_return_code": rc}

                if args.dry_run:
                    continue
                for variant in variants:
                    if variant not in pending:
                        continue
                    info = pending[variant]
                    sweep_output_root = args.output_dir / "sweeps" / dataset / f"nfft{n_fft}_hop{hop}" / f"seed{seed}" / variant
                    sweep_log = args.output_dir / "logs" / "sweep" / dataset / f"nfft{n_fft}_{variant}_seed{seed}.log"
                    sweep_cmd = [
                        args.python_exec,
                        str(args.sweep_script),
                        "--checkpoints",
                        str(info["checkpoint_path"]),
                        "--labels",
                        variant,
                        "--dataset",
                        dataset,
                        "--split",
                        args.sweep_split,
                        "--severity-source",
                        args.severity_source,
                        "--device",
                        sweep_device,
                        "--seed",
                        str(seed),
                        "--n-fft",
                        str(n_fft),
                        "--hop-length",
                        str(hop),
                        "--stft-win-length",
                        str(n_fft),
                        "--stft-window",
                        args.stft_window,
                        "--stft-magnitude-power",
                        str(args.stft_magnitude_power),
                        f"--shift-bins={args.shift_bins}",
                        "--scale-ratios",
                        args.scale_ratios,
                        "--color-max-db",
                        args.color_max_db,
                        "--output-root",
                        str(sweep_output_root),
                    ]
                    sweep_cmd.append("--stft-center" if args.stft_center else "--no-stft-center")
                    if args.enable_mixed:
                        sweep_cmd.append("--enable-mixed")
                    sweep_cmd.append("--enforce-same-budget" if args.enforce_same_budget else "--no-enforce-same-budget")
                    sweep_cmd.extend(sweep_extra)
                    if args.dry_run:
                        print("[dry-run][sweep]", " ".join(sweep_cmd))
                        continue
                    sweep_rc = _run_command(sweep_cmd, sweep_log)
                    if sweep_rc != 0:
                        rows.append(
                            {
                                "status": "sweep_failed",
                                "dataset": dataset,
                                "seed": seed,
                                "variant": variant,
                                "n_fft": n_fft,
                                "win_length": n_fft,
                                "hop_length": hop,
                                "aggregate_scope": args.aggregate_scope,
                                "clean_acc": float("nan"),
                                "avg_acc": float("nan"),
                                "worst_acc": float("nan"),
                                "drop_acc": float("nan"),
                                "clean_mf1": float("nan"),
                                "avg_mf1": float("nan"),
                                "worst_mf1": float("nan"),
                                "drop_mf1": float("nan"),
                                "train_return_code": info["train_return_code"],
                                "sweep_return_code": sweep_rc,
                                "checkpoint_path": str(info["checkpoint_path"]),
                                "train_log_path": str(info["train_log_path"]),
                                "sweep_log_path": str(sweep_log),
                                "summary_csv": "",
                                "robustness_csv": "",
                            }
                        )
                        continue

                    summary_csv, robustness_csv = _load_sweep_paths(sweep_output_root)
                    summary_rows = _read_csv(summary_csv)
                    m = _extract_scope_metrics(summary_rows, checkpoint=variant, scope=args.aggregate_scope)
                    rows.append(
                        {
                            "status": "ok",
                            "dataset": dataset,
                            "seed": seed,
                            "variant": variant,
                            "n_fft": n_fft,
                            "win_length": n_fft,
                            "hop_length": hop,
                            "aggregate_scope": args.aggregate_scope,
                            "clean_acc": m["clean_acc"],
                            "avg_acc": m["avg_acc"],
                            "worst_acc": m["worst_acc"],
                            "drop_acc": m["drop_acc"],
                            "clean_mf1": m["clean_mf1"],
                            "avg_mf1": m["avg_mf1"],
                            "worst_mf1": m["worst_mf1"],
                            "drop_mf1": m["drop_mf1"],
                            "train_return_code": info["train_return_code"],
                            "sweep_return_code": sweep_rc,
                            "checkpoint_path": str(info["checkpoint_path"]),
                            "train_log_path": str(info["train_log_path"]),
                            "sweep_log_path": str(sweep_log),
                            "summary_csv": str(summary_csv),
                            "robustness_csv": str(robustness_csv),
                        }
                    )

    if args.dry_run:
        print("dry_run=1 no files written.")
        return

    per_seed_csv = args.output_dir / "tables" / "stft_sensitivity_per_seed.csv"
    summary_csv = args.output_dir / "tables" / "stft_sensitivity_summary.csv"
    protocol_json = args.output_dir / "tables" / "stft_sensitivity_protocol.json"
    _write_csv(per_seed_csv, rows, PER_SEED_FIELDS)
    _write_csv(summary_csv, _aggregate_ok_rows(rows), SUMMARY_FIELDS)
    protocol = {
        "datasets": datasets,
        "dataset_nfft": dataset_nfft,
        "variants": variants,
        "variant_args": {k: VARIANT_ARGS[k] for k in variants},
        "seeds": seeds,
        "stft_rule": {"win_length": "n_fft", "hop_length": "n_fft/4", "window": args.stft_window, "center": bool(args.stft_center)},
        "train_common_args": train_extra,
        "sweep_common_args": sweep_extra,
        "aggregate_scope": args.aggregate_scope,
    }
    protocol_json.parent.mkdir(parents=True, exist_ok=True)
    protocol_json.write_text(json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"saved_per_seed={per_seed_csv}")
    print(f"saved_summary={summary_csv}")
    print(f"saved_protocol={protocol_json}")


if __name__ == "__main__":
    main()
