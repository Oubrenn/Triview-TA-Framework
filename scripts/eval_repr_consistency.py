import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel  # noqa: E402
from output_utils import (  # noqa: E402
    build_tag,
    ensure_output_dirs,
    save_eval_records,
    stable_hash,
    write_csv,
    write_run_meta,
)
from train_uea import UEAClassifier, collate_fn  # noqa: E402
from transforms import (  # noqa: E402
    band_shift_time,
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
)


def _parse_float_list(raw: str) -> List[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _load_checkpoint(path: Path, device: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {path}")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing config/model_state: {path}")
    return config, state


def _build_classifier(config: Dict[str, object], input_dim: int, num_classes: int, device: str) -> UEAClassifier:
    return UEAClassifier(
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 64)),
        embed_dim=int(config.get("embed_dim", 128)),
        num_classes=num_classes,
        num_heads=int(config.get("num_heads", 4)),
        res_blocks=int(config.get("res_blocks", 2)),
        backbone=str(config.get("backbone", "all")),
        use_temporal_attn=bool(config.get("use_temporal_attn", False)),
        use_se=bool(config.get("use_se", False)),
        se_reduction=int(config.get("se_reduction", 16)),
        use_shared_qk_attn=bool(config.get("use_shared_qk_attn", False)),
        shared_qk_heads=int(config.get("shared_qk_heads", 4)),
        shared_qk_dropout=float(config.get("shared_qk_dropout", 0.0)),
        fuse_dropout=float(config.get("fuse_dropout", 0.0)),
        head_dropout=float(config.get("head_dropout", 0.0)),
    ).to(device)


def _extract_feature_and_pred(model: UEAClassifier, x: torch.Tensor, feature_space: str) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(x, return_intermediates=True)
    feat_key = "h" if feature_space == "h" else "z"
    feat = out[feat_key]
    pred = out["logits"].argmax(dim=1)
    return feat, pred


def _x_value(row: Dict[str, object], transform_name: str) -> float:
    if transform_name == "shift":
        return float(row["b_bins"])
    if transform_name == "scale":
        return float(row["rho"])
    if transform_name == "color":
        return float(row["g_db"])
    return float(row["severity_id"])


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _method_color(method: str, idx: int) -> str:
    key = method.lower().replace(" ", "").replace("-", "")
    if "backbone" in key:
        return "#1f77b4"  # blue
    if "triviewta" in key or ("triview" in key and "ta" in key):
        return "#2ca02c"  # green
    if "triview" in key:
        return "#ff7f0e"  # orange
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    return palette[idx % len(palette)]


def _preferred_distance(rows: List[Dict[str, object]]) -> float:
    mean_row = next((r for r in rows if str(r.get("trial", "")) == "mean"), None)
    if mean_row is not None:
        return float(mean_row["distance"])
    vals = [float(r["distance"]) for r in rows]
    return sum(vals) / max(1, len(vals))


def _aggregate_distance_curve(
    rows: List[Dict[str, object]],
    method: str,
    transform_name: str,
) -> Tuple[List[float], List[float], List[float], int]:
    selected = [
        r
        for r in rows
        if str(r.get("checkpoint", "")) == method and str(r.get("transform", "")) == transform_name
    ]
    if not selected:
        return [], [], [], 0

    by_run: Dict[str, List[Dict[str, object]]] = {}
    for row in selected:
        run_id = str(row.get("run_id", "run0"))
        by_run.setdefault(run_id, []).append(row)

    x_to_vals: Dict[float, List[float]] = {}
    for run_rows in by_run.values():
        per_x: Dict[float, List[Dict[str, object]]] = {}
        for row in run_rows:
            x_val = _x_value(row, transform_name)
            per_x.setdefault(x_val, []).append(row)
        for x_val, x_rows in per_x.items():
            x_to_vals.setdefault(x_val, []).append(_preferred_distance(x_rows))

    xs = sorted(x_to_vals.keys())
    means: List[float] = []
    stds: List[float] = []
    for x_val in xs:
        vals_t = torch.tensor(x_to_vals[x_val], dtype=torch.float32)
        means.append(float(vals_t.mean().item()))
        stds.append(float(vals_t.std(unbiased=False).item()) if vals_t.numel() > 1 else 0.0)
    return xs, means, stds, len(by_run)


def _evaluate_condition(
    model: UEAClassifier,
    loader: DataLoader,
    device: str,
    feature_space: str,
    transform_name: str,
    transform_fn=None,
    severity_id: int = 0,
    b_bins: float = 0.0,
    rho: float = 1.0,
    g_db: float = 0.0,
    trial: object = 0,
    method: str = "",
    collect_samples: bool = True,
    max_sample_rows: int = 100000,
) -> Tuple[float, List[Dict[str, object]]]:
    model.eval()
    total_sim = 0.0
    total = 0
    sample_rows: List[Dict[str, object]] = []
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_clean = batch["x_time"].to(device)
            y = batch["y"].to(device)
            x_aug = x_clean if transform_fn is None else transform_fn(x_clean, batch_idx)
            if isinstance(x_aug, tuple):
                x_aug = x_aug[0]

            feat_clean, pred_clean = _extract_feature_and_pred(model, x_clean, feature_space=feature_space)
            feat_aug, pred_aug = _extract_feature_and_pred(model, x_aug, feature_space=feature_space)
            sim = F.cosine_similarity(feat_clean, feat_aug, dim=-1)

            batch_size = int(sim.shape[0])
            total_sim += float(sim.sum().item())
            total += batch_size

            if collect_samples and len(sample_rows) < max_sample_rows:
                for i in range(batch_size):
                    if len(sample_rows) >= max_sample_rows:
                        break
                    sample_rows.append(
                        {
                            "sample_id": sample_offset + i,
                            "label": int(y[i].item()),
                            "pred_clean": int(pred_clean[i].item()),
                            "pred_aug": int(pred_aug[i].item()),
                            "correct_clean": int(pred_clean[i].item() == y[i].item()),
                            "correct_aug": int(pred_aug[i].item() == y[i].item()),
                            "method": method,
                            "factor_type": transform_name,
                            "split": "clean" if transform_name == "clean" else "transformed",
                            "severity_id": int(severity_id),
                            "b": float(b_bins),
                            "rho": float(rho),
                            "g": float(g_db),
                            "domain_id": -1,
                            "cosine": float(sim[i].item()),
                            "distance": float(1.0 - sim[i].item()),
                            "h_clean_norm": float(torch.norm(feat_clean[i], p=2).item()),
                            "h_aug_norm": float(torch.norm(feat_aug[i], p=2).item()),
                        }
                    )
            sample_offset += batch_size

    return total_sim / max(1, total), sample_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated classifier checkpoints.")
    parser.add_argument("--labels", type=str, default="", help="Comma-separated labels; defaults to checkpoint stems.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-space", type=str, default="h", choices=["h", "z"])
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--stft-win-length", type=int, default=None)
    parser.add_argument("--stft-window", type=str, default="hann", choices=["hann", "hamming"])
    parser.add_argument("--stft-center", action="store_true", default=True)
    parser.add_argument("--no-stft-center", dest="stft_center", action="store_false")
    parser.add_argument("--stft-magnitude-power", type=float, default=None)
    parser.add_argument("--tf-log1p", action="store_true", default=True)
    parser.add_argument("--no-tf-log1p", dest="tf_log1p", action="store_false")
    parser.add_argument("--tf-flatten", action="store_true", default=True)
    parser.add_argument("--no-tf-flatten", dest="tf_flatten", action="store_false")
    parser.add_argument("--normalize-mode", type=str, default="per_sample_channel", choices=["per_sample_channel"])
    parser.add_argument("--shift-bins", type=str, default="-0.5,-0.25,0,0.25,0.5")
    parser.add_argument("--shift-mode", type=str, default="stft", choices=["rfft", "stft"])
    parser.add_argument("--shift-fill", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--scale-ratios", type=str, default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--color-max-db", type=str, default="0,3,6,9")
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-trials", type=int, default=1)
    parser.add_argument("--enable-mixed", action="store_true", default=False)
    parser.add_argument("--mixed-shift-bins", type=str, default="")
    parser.add_argument("--mixed-color-max-db", type=str, default="")
    parser.add_argument("--mixed-trials", type=int, default=1)
    parser.add_argument("--save-sample-records", action="store_true", default=True)
    parser.add_argument("--no-save-sample-records", dest="save_sample_records", action="store_false")
    parser.add_argument("--max-sample-rows", type=int, default=200000)
    parser.add_argument("--plot-metric", type=str, default="distance", choices=["distance"])
    parser.add_argument("--output-root", type=str, default="outputs_new")
    args = parser.parse_args()

    ckpt_paths = [Path(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
    if not ckpt_paths:
        raise ValueError("No checkpoints provided.")
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if labels and len(labels) != len(ckpt_paths):
        raise ValueError("--labels count must match checkpoints count.")
    if not labels:
        labels = [p.stem for p in ckpt_paths]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    loaded = [_load_checkpoint(p, device) for p in ckpt_paths]
    dataset_name = args.dataset or str(loaded[0][0].get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset name missing; set --dataset.")

    cfg0 = loaded[0][0]
    view_config = ViewConfig(
        n_fft=args.n_fft if args.n_fft is not None else int(cfg0.get("n_fft", 256)),
        hop_length=args.hop_length if args.hop_length is not None else int(cfg0.get("hop_length", 64)),
        win_length=args.stft_win_length if args.stft_win_length is not None else cfg0.get("stft_win_length"),
        window_name=args.stft_window,
        center=bool(cfg0.get("stft_center", args.stft_center)),
        magnitude_power=(
            args.stft_magnitude_power if args.stft_magnitude_power is not None else float(cfg0.get("stft_magnitude_power", 1.0))
        ),
        tf_log1p=bool(cfg0.get("tf_log1p", args.tf_log1p)),
        tf_flatten=bool(cfg0.get("tf_flatten", args.tf_flatten)),
        normalize_mode=args.normalize_mode,
        shift_mode=args.shift_fill,
    )
    ds = UEATimeSeriesDataset(dataset_name, split=args.split, pad_to_max=args.pad_to_max, view_config=view_config, normalize=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    input_dim = ds.data[0].shape[0]
    num_classes = len(ds.class_labels)
    num_bins = ds.data[0].shape[-1] // 2 + 1

    shift_bins = _parse_float_list(args.shift_bins)
    scale_ratios = _parse_float_list(args.scale_ratios)
    color_levels = _parse_float_list(args.color_max_db)
    if not shift_bins or not scale_ratios or not color_levels:
        raise ValueError("shift/scale/color severities must be non-empty.")
    mixed_shift_bins = _parse_float_list(args.mixed_shift_bins) or shift_bins
    mixed_color_levels = _parse_float_list(args.mixed_color_max_db) or color_levels
    color_trials = max(1, int(args.color_trials))
    mixed_trials = max(1, int(args.mixed_trials))

    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for (config, state), ckpt_path, label in zip(loaded, ckpt_paths, labels):
        model = _build_classifier(config, input_dim=input_dim, num_classes=num_classes, device=device)
        model.load_state_dict(state, strict=True)

        clean_val, clean_samples = _evaluate_condition(
            model=model,
            loader=loader,
            device=device,
            feature_space=args.feature_space,
            transform_name="clean",
            transform_fn=None,
            method=label,
            collect_samples=args.save_sample_records,
            max_sample_rows=args.max_sample_rows - len(sample_rows),
        )
        rows.append(
            {
                "checkpoint": label,
                "transform": "clean",
                "severity_id": 0,
                "b_bins": 0.0,
                "rho": 1.0,
                "g_db": 0.0,
                "trial": 0,
                "feature_space": args.feature_space,
                "cosine": clean_val,
                "run_id": ckpt_path.stem,
            }
        )
        sample_rows.extend(clean_samples)

        for i, b in enumerate(shift_bins):
            def _shift(x: torch.Tensor, _batch_idx: int, bins: float = b):
                if args.shift_mode == "stft":
                    return apply_per_sample_channel(
                        x,
                        lambda s: band_shift_time_stft(
                            s,
                            bins,
                            n_fft=view_config.n_fft,
                            hop_length=view_config.hop_length,
                            win_length=view_config.win_length,
                            window_name=view_config.window_name,
                            center=view_config.center,
                            shift_mode=args.shift_fill,
                        ),
                    )
                return apply_per_sample_channel(x, lambda s: band_shift_time(s, bins, shift_mode=args.shift_fill))

            val, one_samples = _evaluate_condition(
                model=model,
                loader=loader,
                device=device,
                feature_space=args.feature_space,
                transform_name="shift",
                transform_fn=_shift,
                severity_id=i,
                b_bins=b,
                method=label,
                collect_samples=args.save_sample_records,
                max_sample_rows=args.max_sample_rows - len(sample_rows),
            )
            rows.append(
                {
                    "checkpoint": label,
                    "transform": "shift",
                    "severity_id": i,
                    "b_bins": b,
                    "rho": 1.0,
                    "g_db": 0.0,
                    "trial": 0,
                    "feature_space": args.feature_space,
                    "cosine": val,
                    "run_id": ckpt_path.stem,
                }
            )
            sample_rows.extend(one_samples)

        for i, rho in enumerate(scale_ratios):
            val, one_samples = _evaluate_condition(
                model=model,
                loader=loader,
                device=device,
                feature_space=args.feature_space,
                transform_name="scale",
                transform_fn=lambda x, _idx, r=rho: apply_per_sample_channel(x, lambda s: frequency_scale_time(s, r)),
                severity_id=i,
                rho=rho,
                method=label,
                collect_samples=args.save_sample_records,
                max_sample_rows=args.max_sample_rows - len(sample_rows),
            )
            rows.append(
                {
                    "checkpoint": label,
                    "transform": "scale",
                    "severity_id": i,
                    "b_bins": 0.0,
                    "rho": rho,
                    "g_db": 0.0,
                    "trial": 0,
                    "feature_space": args.feature_space,
                    "cosine": val,
                    "run_id": ckpt_path.stem,
                }
            )
            sample_rows.extend(one_samples)

        for i, g_db in enumerate(color_levels):
            vals = []
            for trial in range(color_trials):
                gains = make_coloring_gains(
                    num_bins=num_bins,
                    bands=args.color_bands,
                    max_gain_db=g_db,
                    generator=torch.Generator().manual_seed(args.seed + i * 1000 + trial),
                )
                val, one_samples = _evaluate_condition(
                    model=model,
                    loader=loader,
                    device=device,
                    feature_space=args.feature_space,
                    transform_name="color",
                    transform_fn=lambda x, _idx, g=gains: apply_per_sample_channel(x, lambda s: spectral_coloring(s, g)),
                    severity_id=i,
                    g_db=g_db,
                    trial=trial,
                    method=label,
                    collect_samples=args.save_sample_records,
                    max_sample_rows=args.max_sample_rows - len(sample_rows),
                )
                vals.append(val)
                rows.append(
                    {
                        "checkpoint": label,
                        "transform": "color",
                        "severity_id": i,
                        "b_bins": 0.0,
                        "rho": 1.0,
                        "g_db": g_db,
                        "trial": trial,
                        "feature_space": args.feature_space,
                        "cosine": val,
                        "run_id": ckpt_path.stem,
                    }
                )
                sample_rows.extend(one_samples)
            if color_trials > 1:
                rows.append(
                    {
                        "checkpoint": label,
                        "transform": "color",
                        "severity_id": i,
                        "b_bins": 0.0,
                        "rho": 1.0,
                        "g_db": g_db,
                        "trial": "mean",
                        "feature_space": args.feature_space,
                        "cosine": sum(vals) / len(vals),
                        "run_id": ckpt_path.stem,
                    }
                )

        if args.enable_mixed:
            for i, b in enumerate(mixed_shift_bins):
                for j, g_db in enumerate(mixed_color_levels):
                    sid = i * len(mixed_color_levels) + j
                    vals = []
                    for trial in range(mixed_trials):
                        gains = make_coloring_gains(
                            num_bins=num_bins,
                            bands=args.color_bands,
                            max_gain_db=g_db,
                            generator=torch.Generator().manual_seed(args.seed + sid * 10000 + trial),
                        )

                        def _mixed(x: torch.Tensor, _batch_idx: int, bins: float = b, g=gains):
                            if args.shift_mode == "stft":
                                x_shift = apply_per_sample_channel(
                                    x,
                                    lambda s: band_shift_time_stft(
                                        s,
                                        bins,
                                        n_fft=view_config.n_fft,
                                        hop_length=view_config.hop_length,
                                        win_length=view_config.win_length,
                                        window_name=view_config.window_name,
                                        center=view_config.center,
                                        shift_mode=args.shift_fill,
                                    ),
                                )
                            else:
                                x_shift = apply_per_sample_channel(x, lambda s: band_shift_time(s, bins, shift_mode=args.shift_fill))
                            return apply_per_sample_channel(x_shift, lambda s: spectral_coloring(s, g))

                        val, one_samples = _evaluate_condition(
                            model=model,
                            loader=loader,
                            device=device,
                            feature_space=args.feature_space,
                            transform_name="mixed_shift_color",
                            transform_fn=_mixed,
                            severity_id=sid,
                            b_bins=b,
                            g_db=g_db,
                            trial=trial,
                            method=label,
                            collect_samples=args.save_sample_records,
                            max_sample_rows=args.max_sample_rows - len(sample_rows),
                        )
                        vals.append(val)
                        rows.append(
                            {
                                "checkpoint": label,
                                "transform": "mixed_shift_color",
                                "severity_id": sid,
                                "b_bins": b,
                                "rho": 1.0,
                                "g_db": g_db,
                                "trial": trial,
                                "feature_space": args.feature_space,
                                "cosine": val,
                                "run_id": ckpt_path.stem,
                            }
                        )
                        sample_rows.extend(one_samples)
                    if mixed_trials > 1:
                        rows.append(
                            {
                                "checkpoint": label,
                                "transform": "mixed_shift_color",
                                "severity_id": sid,
                                "b_bins": b,
                                "rho": 1.0,
                                "g_db": g_db,
                                "trial": "mean",
                                "feature_space": args.feature_space,
                                "cosine": sum(vals) / len(vals),
                                "run_id": ckpt_path.stem,
                            }
                        )

    for row in rows:
        cosine = float(row["cosine"])
        row["distance"] = 1.0 - cosine
        row["plot_metric"] = "distance"
        row["plot_value"] = float(row["distance"])
    for row in sample_rows:
        cosine = float(row["cosine"])
        row["distance"] = 1.0 - cosine

    transforms = ["shift", "scale", "color"] + (["mixed_shift_color"] if args.enable_mixed else [])
    if len(transforms) == 4:
        fig, axes_grid = plt.subplots(2, 2, figsize=(11.2, 8.2))
        axes = list(axes_grid.flat)
    else:
        ncols = 2 if len(transforms) > 1 else 1
        nrows = (len(transforms) + ncols - 1) // ncols
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.1 * nrows))
        axes = list(axes_grid.flat) if hasattr(axes_grid, "flat") else [axes_grid]

    methods = _ordered_unique(labels)
    title_map = {
        "shift": "(a) Shift",
        "scale": "(b) Scale",
        "color": "(c) Color",
        "mixed_shift_color": "(d) Shift+Color",
    }
    legend_handles = []
    for method_idx, method in enumerate(methods):
        color = _method_color(method, method_idx)
        legend_handles.append(Line2D([0], [0], color=color, lw=2.2, marker="o", label=method))

    for ax_idx, (ax, name) in enumerate(zip(axes, transforms)):
        y_lo, y_hi = float("inf"), float("-inf")
        for method_idx, method in enumerate(methods):
            color = _method_color(method, method_idx)
            xs, means, stds, n_runs = _aggregate_distance_curve(rows, method, name)
            if not xs:
                continue
            if len(xs) <= 1:
                ax.scatter(xs, means, marker="o", color=color, s=30)
            else:
                ax.plot(xs, means, marker="o", color=color, linewidth=2.0, markersize=5.5)
            if n_runs > 1 and len(xs) > 1:
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.16)
                y_lo = min(y_lo, min(lower))
                y_hi = max(y_hi, max(upper))
            else:
                y_lo = min(y_lo, min(means))
                y_hi = max(y_hi, max(means))

        ax.set_title(title_map.get(name, name.replace("_", " ").title()))
        ax.set_xlabel("Severity")
        if ax_idx % 2 == 0:
            ax.set_ylabel(f"Cosine distance (1-cos, {args.feature_space})")
        ax.grid(True, alpha=0.3)
        if y_hi > y_lo:
            pad = 0.12 * (y_hi - y_lo)
            lo = max(0.0, y_lo - pad)
            hi = y_hi + pad
            if hi <= lo:
                hi = lo + 1e-3
            ax.set_ylim(lo, hi)
        else:
            ax.set_ylim(max(0.0, y_lo - 0.02), y_hi + 0.02)

    for extra_ax in axes[len(transforms):]:
        extra_ax.axis("off")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(legend_handles)),
        frameon=False,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(out_root)
    stft_hash = view_config.to_preprocess_config().hash()
    sev_hash = stable_hash(
        {
            "shift_bins": shift_bins,
            "scale_ratios": scale_ratios,
            "color_levels": color_levels,
            "mixed_shift_bins": mixed_shift_bins if args.enable_mixed else [],
            "mixed_color_levels": mixed_color_levels if args.enable_mixed else [],
            "feature_space": args.feature_space,
            "plot_metric": args.plot_metric,
        }
    )
    mix_tag = f"-mx{len(mixed_shift_bins) * len(mixed_color_levels)}" if args.enable_mixed else ""
    stem = build_tag(
        "repr_consistency",
        dataset_name,
        args.split,
        f"seed{args.seed}",
        f"feat{args.feature_space}",
        f"metric{args.plot_metric}",
        f"stft{stft_hash}",
        f"sh{len(shift_bins)}-sc{len(scale_ratios)}-co{len(color_levels)}{mix_tag}",
        f"sev{sev_hash}",
    )
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    sample_csv = csv_dir / f"{stem}_sample_records.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, rows)
    if args.save_sample_records:
        save_eval_records(sample_rows, sample_csv)
    meta = write_run_meta(
        output_root=out_root,
        script_name="scripts/eval_repr_consistency.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "sample_csv": str(sample_csv) if args.save_sample_records else "",
            "labels": labels,
            "checkpoints": [str(p) for p in ckpt_paths],
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    if args.save_sample_records:
        print(f"saved_sample_csv={sample_csv}")
    print(f"saved_meta={meta}")


if __name__ == "__main__":
    main()
