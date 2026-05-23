import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel  # noqa: E402
from models import MultiViewModel  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, stable_hash, write_csv, write_run_meta  # noqa: E402
from preprocessing import build_triview_from_time  # noqa: E402
from train_uea import collate_fn  # noqa: E402
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
        return "#1f77b4"
    if "triviewta" in key or ("triview" in key and "ta" in key):
        return "#2ca02c"
    if "triview" in key:
        return "#ff7f0e"
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    return palette[idx % len(palette)]


def _load_checkpoint(path: Path, device: str) -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Invalid checkpoint: {path}")
    return ckpt


def _build_model(config: Dict[str, object], sample_views: Dict[str, torch.Tensor], device: str) -> MultiViewModel:
    return MultiViewModel(
        input_dim_time=sample_views["x_time"].shape[0],
        input_dim_freq=sample_views["x_freq"].shape[0],
        input_dim_tf=sample_views["x_tf"].shape[0],
        hidden_dim=int(config.get("hidden_dim", 64)),
        output_dim=int(config.get("embed_dim", 128)),
        num_heads=int(config.get("num_heads", 4)),
        res_blocks=int(config.get("res_blocks", 2)),
        backbone=str(config.get("backbone", "all")),
        use_se=bool(config.get("use_se", False)),
        se_reduction=int(config.get("se_reduction", 16)),
        use_temporal_attn=bool(config.get("use_temporal_attn", False)),
        use_shared_qk_attn=bool(config.get("use_shared_qk_attn", False)),
        shared_qk_heads=int(config.get("shared_qk_heads", 4)),
        shared_qk_dropout=float(config.get("shared_qk_dropout", 0.0)),
        fuse_dropout=float(config.get("fuse_dropout", 0.0)),
    ).to(device)


def _build_batch_views(x_time: torch.Tensor, preprocess_config) -> Tuple[torch.Tensor, torch.Tensor]:
    freq_list, tf_list = [], []
    for i in range(x_time.shape[0]):
        views = build_triview_from_time(x_time[i], preprocess_config)
        freq_list.append(views["x_freq"])
        tf_list.append(views["x_tf"])
    return torch.stack(freq_list, dim=0), torch.stack(tf_list, dim=0)


def _pair_sims(v_time: torch.Tensor, v_freq: torch.Tensor, v_tf: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "time_freq": F.cosine_similarity(v_time, v_freq, dim=-1),
        "time_tf": F.cosine_similarity(v_time, v_tf, dim=-1),
        "freq_tf": F.cosine_similarity(v_freq, v_tf, dim=-1),
    }


def _eval_condition(
    model: MultiViewModel,
    loader: DataLoader,
    device: str,
    preprocess_config,
    feature_space: str,
    transform_fn=None,
) -> Dict[str, float]:
    model.eval()
    sums = {"time_freq": 0.0, "time_tf": 0.0, "freq_tf": 0.0}
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_time = batch["x_time"].to(device)
            if transform_fn is not None:
                x_time = transform_fn(x_time, batch_idx)
            x_freq, x_tf = _build_batch_views(x_time, preprocess_config)
            out = model(x_time, x_freq, x_tf, return_intermediates=True)
            if feature_space == "h":
                sims = _pair_sims(out["h_time"], out["h_freq"], out["h_tf"])
            else:
                sims = _pair_sims(out["z_time"], out["z_freq"], out["z_tf"])
            bs = int(x_time.shape[0])
            total += bs
            for key in sums:
                sums[key] += float(sims[key].sum().item())
    return {k: sums[k] / max(1, total) for k in sums}


def _aggregate_method_condition_pair(
    rows: List[Dict[str, object]],
    methods: List[str],
    conditions: List[str],
    pairs: List[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for method in methods:
        for cond in conditions:
            for pair in pairs:
                vals = [
                    float(r["plot_value"])
                    for r in rows
                    if str(r["checkpoint"]) == method and str(r["condition"]) == cond and str(r["pair"]) == pair
                ]
                if not vals:
                    vals = [0.0]
                vals_t = torch.tensor(vals, dtype=torch.float32)
                mean_v = float(vals_t.mean().item())
                std_v = float(vals_t.std(unbiased=False).item()) if vals_t.numel() > 1 else 0.0
                out.append(
                    {
                        "method": method,
                        "condition": cond,
                        "pair": pair,
                        "mean_delta": mean_v,
                        "std_delta": std_v,
                        "n_runs": int(vals_t.numel()),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated pretrain checkpoints.")
    parser.add_argument("--labels", type=str, default="", help="Comma-separated labels; defaults to checkpoint stems.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-space", type=str, default="h", choices=["h", "z"])
    parser.add_argument(
        "--metric",
        type=str,
        default="delta_abs",
        choices=["distance_abs", "distance", "cosine", "raw", "delta", "delta_abs"],
        help=(
            "distance_abs: 1-|cosine|; distance: 1-cosine; cosine/raw: cosine directly; "
            "delta: cosine_clean-cosine_condition; delta_abs: (1-|cos|)_condition-(1-|cos|)_clean."
        ),
    )
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    parser.add_argument("--shift-mode", type=str, default="stft", choices=["rfft", "stft"])
    parser.add_argument("--shift-fill", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--safe-shift-bins", type=float, default=0.25)
    parser.add_argument("--safe-scale-ratio", type=float, default=1.0)
    parser.add_argument("--safe-color-max-db", type=float, default=3.0)
    parser.add_argument("--mixed-shift-bins", type=str, default="0.5,1.0")
    parser.add_argument("--mixed-scale-ratio", type=float, default=1.0)
    parser.add_argument("--mixed-color-max-db", type=str, default="3,6,9")
    parser.add_argument("--mixed-trials", type=int, default=1)
    parser.add_argument("--figure-role", type=str, default="appendix", choices=["appendix", "main"])
    parser.add_argument(
        "--appendix-plot",
        type=str,
        default="delta_grouped",
        choices=["delta_grouped", "triptych"],
        help="delta_grouped: grouped bars over pairs for Safe/Mixed delta-from-clean; triptych: keep Clean/Safe/Mixed panels.",
    )
    parser.add_argument("--appendix-ymin", type=float, default=None)
    parser.add_argument("--appendix-ymax", type=float, default=None)
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
    dataset_name = args.dataset or str(loaded[0]["config"].get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset name missing; set --dataset.")

    cfg0 = loaded[0]["config"]
    view_config = ViewConfig(
        n_fft=int(cfg0.get("n_fft", 256)),
        hop_length=int(cfg0.get("hop_length", 64)),
        win_length=cfg0.get("stft_win_length"),
        window_name=str(cfg0.get("stft_window", "hann")),
        center=bool(cfg0.get("stft_center", True)),
        magnitude_power=float(cfg0.get("stft_magnitude_power", 1.0)),
        tf_log1p=bool(cfg0.get("tf_log1p", True)),
        tf_flatten=bool(cfg0.get("tf_flatten", True)),
        normalize_mode=str(cfg0.get("normalize_mode", "per_sample_channel")),
        shift_mode=args.shift_fill,
    )
    preprocess_config = view_config.to_preprocess_config()
    ds = UEATimeSeriesDataset(dataset_name, split=args.split, pad_to_max=args.pad_to_max, view_config=view_config, normalize=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    sample_time = ds[0]["x_time"]
    sample_views = {"x_time": sample_time, **build_triview_from_time(sample_time, preprocess_config)}
    num_bins = sample_time.shape[-1] // 2 + 1

    safe_gains = make_coloring_gains(
        num_bins=num_bins,
        bands=int(cfg0.get("pretrain_color_bands", 8)),
        max_gain_db=args.safe_color_max_db,
        generator=torch.Generator().manual_seed(args.seed),
    )

    def _apply_transform(x: torch.Tensor, b: float, rho: float, gains: torch.Tensor) -> torch.Tensor:
        if args.shift_mode == "stft":
            x_shift = apply_per_sample_channel(
                x,
                lambda s: band_shift_time_stft(
                    s,
                    b,
                    n_fft=view_config.n_fft,
                    hop_length=view_config.hop_length,
                    win_length=view_config.win_length,
                    window_name=view_config.window_name,
                    center=view_config.center,
                    shift_mode=args.shift_fill,
                ),
            )
        else:
            x_shift = apply_per_sample_channel(x, lambda s: band_shift_time(s, b, shift_mode=args.shift_fill))
        x_scale = apply_per_sample_channel(x_shift, lambda s: frequency_scale_time(s, rho))
        return apply_per_sample_channel(x_scale, lambda s: spectral_coloring(s, gains))

    mixed_bs = _parse_float_list(args.mixed_shift_bins)
    mixed_gs = _parse_float_list(args.mixed_color_max_db)
    mixed_trials = max(1, int(args.mixed_trials))
    metric_mode = "cosine" if args.metric == "raw" else args.metric
    if args.figure_role == "appendix" and args.appendix_plot == "delta_grouped" and metric_mode not in {"delta", "delta_abs"}:
        print("warning=appendix delta_grouped expects a delta metric; overriding metric to delta_abs")
        metric_mode = "delta_abs"
    if args.figure_role == "main" and metric_mode != "delta_abs":
        print("warning=main figure expects delta_abs metric; overriding metric to delta_abs")
        metric_mode = "delta_abs"
    if args.figure_role == "main" and args.feature_space != "h":
        print("warning=main figure expects feature-space h; overriding feature-space to h")
        args.feature_space = "h"

    rows: List[Dict[str, object]] = []
    for ckpt, ckpt_path, label in zip(loaded, ckpt_paths, labels):
        model = _build_model(ckpt["config"], sample_views, device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        clean = _eval_condition(model, loader, device, preprocess_config, feature_space=args.feature_space, transform_fn=None)
        safe = _eval_condition(
            model,
            loader,
            device,
            preprocess_config,
            feature_space=args.feature_space,
            transform_fn=lambda x, _idx: _apply_transform(x, args.safe_shift_bins, args.safe_scale_ratio, safe_gains),
        )
        mixed_acc = {"time_freq": 0.0, "time_tf": 0.0, "freq_tf": 0.0}
        mixed_count = 0
        for i, b in enumerate(mixed_bs):
            for j, g_db in enumerate(mixed_gs):
                sid = i * max(1, len(mixed_gs)) + j
                for t in range(mixed_trials):
                    gains = make_coloring_gains(
                        num_bins=num_bins,
                        bands=int(ckpt["config"].get("pretrain_color_bands", 8)),
                        max_gain_db=g_db,
                        generator=torch.Generator().manual_seed(args.seed + sid * 1000 + t),
                    )
                    one = _eval_condition(
                        model,
                        loader,
                        device,
                        preprocess_config,
                        feature_space=args.feature_space,
                        transform_fn=lambda x, _idx, bb=b, gg=gains: _apply_transform(x, bb, args.mixed_scale_ratio, gg),
                    )
                    for k in mixed_acc:
                        mixed_acc[k] += one[k]
                    mixed_count += 1
        mixed = {k: mixed_acc[k] / max(1, mixed_count) for k in mixed_acc}

        clean_abs = {k: 1.0 - abs(float(v)) for k, v in clean.items()}
        for cond_name, metrics in [("clean", clean), ("safe", safe), ("mixed", mixed)]:
            for pair_name, raw_cos in metrics.items():
                raw_dist = 1.0 - raw_cos
                abs_dist = 1.0 - abs(raw_cos)
                if metric_mode == "cosine":
                    value = raw_cos
                elif metric_mode == "distance":
                    value = raw_dist
                elif metric_mode == "distance_abs":
                    value = abs_dist
                elif metric_mode == "delta_abs":
                    value = abs_dist - clean_abs[pair_name] if cond_name != "clean" else 0.0
                else:
                    value = clean[pair_name] - raw_cos if cond_name != "clean" else 0.0
                rows.append(
                    {
                        "checkpoint": label,
                        "checkpoint_path": str(ckpt_path),
                        "condition": cond_name,
                        "pair": pair_name,
                        "feature_space": args.feature_space,
                        "metric": metric_mode,
                        "raw_cosine": raw_cos,
                        "raw_distance": raw_dist,
                        "abs_distance": abs_dist,
                        "plot_value": value,
                    }
                )

    pairs = ["time_freq", "time_tf", "freq_tf"]
    pair_labels = ["time-freq", "time-tf", "freq-tf"]
    methods = _ordered_unique(labels)
    agg_plot_rows: List[Dict[str, object]] = []

    if args.figure_role == "main":
        conds_plot = ["safe", "mixed"]
        agg_plot_rows = _aggregate_method_condition_pair(rows, methods, conds_plot, pairs)
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
        x = torch.arange(len(pairs), dtype=torch.float32)
        n_methods = max(1, len(methods))
        width = 0.82 / n_methods

        y_candidates: List[float] = []
        for r in agg_plot_rows:
            mean_v = float(r["mean_delta"])
            std_v = float(r["std_delta"])
            y_candidates.extend([mean_v - std_v, mean_v + std_v])
        if args.appendix_ymin is not None and args.appendix_ymax is not None:
            y_lo = float(args.appendix_ymin)
            y_hi = float(args.appendix_ymax)
        else:
            y_min = min(y_candidates) if y_candidates else -0.02
            y_max = max(y_candidates) if y_candidates else 0.02
            span = y_max - y_min
            if span < 1e-6:
                span = 0.04
            pad = max(0.008, 0.18 * span)
            y_lo = y_min - pad
            y_hi = y_max + pad
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-3

        title_map = {"safe": "(a) Safe", "mixed": "(b) Mixed"}
        for cidx, cond in enumerate(conds_plot):
            ax = axes[cidx]
            for midx, method in enumerate(methods):
                means: List[float] = []
                stds: List[float] = []
                for pair in pairs:
                    one = next(
                        item
                        for item in agg_plot_rows
                        if item["method"] == method and item["condition"] == cond and item["pair"] == pair
                    )
                    means.append(float(one["mean_delta"]))
                    stds.append(float(one["std_delta"]))
                offset = (midx - 0.5 * (n_methods - 1)) * width
                bars = ax.bar(
                    x + offset,
                    means,
                    width=width,
                    color=_method_color(method, midx),
                    yerr=stds,
                    capsize=3.0,
                    label=method,
                )
                for bar, val in zip(bars, means):
                    y_shift = 0.012 * (y_hi - y_lo)
                    y_text = val + (y_shift if val >= 0 else -y_shift)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y_text,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=8,
                    )
            ax.axhline(0.0, color="#1a1a1a", linewidth=1.35, alpha=0.95)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_title(title_map[cond])
            if cidx == 0:
                ax.set_ylabel("Change in cross-view distance from clean")

        legend_handles = [Patch(color=_method_color(m, i), label=m) for i, m in enumerate(methods)]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=max(1, len(legend_handles)),
            frameon=False,
        )
        fig.text(
            0.5,
            0.008,
            "Clean is omitted because all clean-relative shifts are zero by definition.",
            ha="center",
            fontsize=9,
        )
        plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])
    elif args.figure_role == "appendix" and args.appendix_plot == "delta_grouped":
        conds_plot = ["safe", "mixed"]
        agg_plot_rows = _aggregate_method_condition_pair(rows, methods, conds_plot, pairs)
        ncols = max(1, len(methods))
        fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.5), sharey=True)
        axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
        x = torch.arange(len(pairs), dtype=torch.float32)
        cond_colors = {"safe": "#4C78A8", "mixed": "#F58518"}
        cond_labels = {"safe": "Safe", "mixed": "Mixed"}
        width = 0.34

        y_candidates: List[float] = []
        for r in agg_plot_rows:
            mean_v = float(r["mean_delta"])
            std_v = float(r["std_delta"])
            y_candidates.extend([mean_v - std_v, mean_v + std_v])
        if args.appendix_ymin is not None and args.appendix_ymax is not None:
            y_lo = float(args.appendix_ymin)
            y_hi = float(args.appendix_ymax)
        else:
            y_min = min(y_candidates) if y_candidates else -0.02
            y_max = max(y_candidates) if y_candidates else 0.02
            span = y_max - y_min
            if span < 1e-6:
                span = 0.04
            pad = max(0.008, 0.20 * span)
            y_lo = y_min - pad
            y_hi = y_max + pad
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-3

        for midx, method in enumerate(methods):
            ax = axes_list[midx]
            for cidx, cond in enumerate(conds_plot):
                ys: List[float] = []
                for pair in pairs:
                    one = next(
                        item
                        for item in agg_plot_rows
                        if item["method"] == method and item["condition"] == cond and item["pair"] == pair
                    )
                    ys.append(float(one["mean_delta"]))
                offset = (cidx - 0.5) * width
                bars = ax.bar(x + offset, ys, width=width, color=cond_colors[cond], label=cond_labels[cond])
                for bar, val in zip(bars, ys):
                    y_text = val + (0.012 * (y_hi - y_lo) if val >= 0 else -0.014 * (y_hi - y_lo))
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        y_text,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=8,
                    )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.65)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_title("(a) Delta from clean" if len(methods) == 1 else method)
            if midx == 0:
                ax.set_ylabel(f"Delta coordination distance (1 - |cos|, {args.feature_space})")

        handles = [Patch(color=cond_colors[c], label=cond_labels[c]) for c in conds_plot]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    else:
        conditions = ["clean", "safe", "mixed"]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
        x = torch.arange(len(pairs), dtype=torch.float32)
        width = 0.82 / max(1, len(labels))
        all_plot_vals = [float(r["plot_value"]) for r in rows]
        title_map = {"clean": "(a) Clean", "safe": "(b) Safe", "mixed": "(c) Mixed"}

        if args.figure_role == "appendix":
            if args.appendix_ymin is not None and args.appendix_ymax is not None:
                y_lo = float(args.appendix_ymin)
                y_hi = float(args.appendix_ymax)
            else:
                y_min = min(all_plot_vals) if all_plot_vals else 0.0
                y_max = max(all_plot_vals) if all_plot_vals else 1.0
                span = y_max - y_min
                if span < 1e-6:
                    span = 0.04
                pad = max(0.02, 0.2 * span)
                y_lo = y_min - pad
                y_hi = y_max + pad
            if y_hi <= y_lo:
                y_hi = y_lo + 1e-3
        else:
            y_lo, y_hi = None, None

        for cidx, cond in enumerate(conditions):
            ax = axes[cidx]
            for midx, label in enumerate(labels):
                ys = []
                for pair in pairs:
                    row = next(r for r in rows if r["checkpoint"] == label and r["condition"] == cond and r["pair"] == pair)
                    ys.append(float(row["plot_value"]))
                bars = ax.bar(x + midx * width, ys, width=width)
                if args.figure_role == "appendix":
                    y_span = max((y_hi - y_lo) if (y_hi is not None and y_lo is not None) else 1e-3, 1e-3)
                    for b, val in zip(bars, ys):
                        ax.text(
                            b.get_x() + b.get_width() / 2.0,
                            val + 0.015 * y_span,
                            f"{val:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
            ax.set_title(title_map[cond] if args.figure_role == "appendix" else cond.title())
            ax.set_xticks(x + (len(labels) - 1) * width / 2.0)
            ax.set_xticklabels(pair_labels)
            ax.grid(True, axis="y", alpha=0.3)
            if args.figure_role == "appendix":
                ax.set_ylim(float(y_lo), float(y_hi))
            else:
                if metric_mode == "cosine":
                    ax.set_ylim(-0.05, 1.05)
                elif metric_mode in {"distance", "distance_abs"}:
                    max_v = max(all_plot_vals) if all_plot_vals else 0.0
                    ax.set_ylim(0.0, max(1e-3, max_v * 1.15))
            if cidx == 0:
                if metric_mode == "cosine":
                    ax.set_ylabel(f"cosine similarity ({args.feature_space})")
                elif metric_mode == "distance":
                    ax.set_ylabel(f"Cosine distance (1 - cos, {args.feature_space})")
                elif metric_mode == "distance_abs":
                    ax.set_ylabel(f"Cosine distance (1 - |cos|, {args.feature_space})")
                elif metric_mode == "delta_abs":
                    ax.set_ylabel(f"Delta distance from clean (1 - |cos|, {args.feature_space})")
                else:
                    ax.set_ylabel(f"delta (clean - condition, {args.feature_space})")
            if args.figure_role != "appendix":
                ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

    out_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(out_root)
    sev_hash = stable_hash(
        {
            "safe_shift_bins": args.safe_shift_bins,
            "safe_scale_ratio": args.safe_scale_ratio,
            "safe_color_max_db": args.safe_color_max_db,
            "mixed_shift_bins": mixed_bs,
            "mixed_scale_ratio": args.mixed_scale_ratio,
            "mixed_color_max_db": mixed_gs,
            "mixed_trials": mixed_trials,
            "feature_space": args.feature_space,
            "metric": metric_mode,
            "figure_role": args.figure_role,
            "appendix_plot": args.appendix_plot,
        }
    )
    stem = build_tag(
        "multiview_coordination",
        args.figure_role,
        dataset_name,
        args.split,
        f"seed{args.seed}",
        f"feat{args.feature_space}",
        f"metric{metric_mode}",
        f"sev{sev_hash}",
    )
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    agg_csv = csv_dir / f"{stem}_agg_plot.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, rows)
    if agg_plot_rows:
        write_csv(agg_csv, agg_plot_rows)
    meta = write_run_meta(
        output_root=out_root,
        script_name="scripts/eval_multiview_coordination.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "agg_plot_csv": str(agg_csv) if agg_plot_rows else "",
            "labels": labels,
            "checkpoints": [str(p) for p in ckpt_paths],
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    if agg_plot_rows:
        print(f"saved_agg_csv={agg_csv}")
    print(f"saved_meta={meta}")


if __name__ == "__main__":
    main()
