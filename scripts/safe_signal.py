import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, stable_hash, write_csv, write_run_meta  # noqa: E402
from train_uea import collate_fn  # noqa: E402
from transforms import (  # noqa: E402
    band_shift_time,
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_bandwidth,
    spectral_centroid,
    spectral_coloring,
    stft_magnitude,
)


def _parse_float_list(raw: str) -> List[float]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_band_list(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _mean_std_ci95(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item()) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return mean, std, ci95


def _signal_ratio_values(x_clean: torch.Tensor, x_aug: torch.Tensor, view_config: ViewConfig) -> Tuple[List[float], List[float]]:
    # Inputs follow (B, C, T). Ratios are computed per sample-channel pair.
    x_clean = x_clean.detach()
    x_aug = x_aug.detach()
    if x_clean.dim() != 3 or x_aug.dim() != 3:
        raise ValueError(f"Expected (B, C, T), got clean={tuple(x_clean.shape)} aug={tuple(x_aug.shape)}")
    if x_clean.shape != x_aug.shape:
        raise ValueError(f"Shape mismatch between clean and augmented batches: {x_clean.shape} vs {x_aug.shape}")

    centroid_ratios: List[float] = []
    bandwidth_ratios: List[float] = []
    eps = 1e-8
    for b in range(x_clean.shape[0]):
        for c in range(x_clean.shape[1]):
            clean_mag = stft_magnitude(
                x_clean[b, c],
                n_fft=view_config.n_fft,
                hop_length=view_config.hop_length,
                win_length=view_config.win_length,
                window_name=view_config.window_name,
                center=view_config.center,
                magnitude_power=view_config.magnitude_power,
            )
            aug_mag = stft_magnitude(
                x_aug[b, c],
                n_fft=view_config.n_fft,
                hop_length=view_config.hop_length,
                win_length=view_config.win_length,
                window_name=view_config.window_name,
                center=view_config.center,
                magnitude_power=view_config.magnitude_power,
            )
            centroid_clean = spectral_centroid(clean_mag)
            centroid_aug = spectral_centroid(aug_mag)
            bw_clean = spectral_bandwidth(clean_mag)
            bw_aug = spectral_bandwidth(aug_mag)
            centroid_ratio = (centroid_aug - centroid_clean).abs() / (centroid_clean.abs() + eps)
            bandwidth_ratio = (bw_aug - bw_clean).abs() / (bw_clean.abs() + eps)
            centroid_ratios.append(float(centroid_ratio.item()))
            bandwidth_ratios.append(float(bandwidth_ratio.item()))
    return centroid_ratios, bandwidth_ratios


def _signal_eval(
    loader: DataLoader,
    view_config: ViewConfig,
    transform_fn,
    max_param_rows: int,
    transform_name: str,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    centroid_values: List[float] = []
    bandwidth_values: List[float] = []
    param_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x_clean = batch["x_time"]
            x_aug, meta = transform_fn(x_clean, batch_idx)
            c_vals, bw_vals = _signal_ratio_values(x_clean, x_aug, view_config=view_config)
            centroid_values.extend(c_vals)
            bandwidth_values.extend(bw_vals)

            if len(param_rows) < max_param_rows:
                param_rows.append(
                    {
                        "transform": transform_name,
                        "batch_idx": batch_idx,
                        "batch_size": int(x_clean.shape[0]),
                        "severity_id": int(meta.get("severity_id", 0)),
                        "rho": float(meta.get("rho", 1.0)),
                        "g_db": float(meta.get("g_db", 0.0)),
                        "b_bins": float(meta.get("b_bins", 0.0)),
                        "seed": int(meta.get("seed", 0)),
                        "shift_fill": str(meta.get("shift_fill", "na")),
                    }
                )

    c_mean, c_std, c_ci95 = _mean_std_ci95(centroid_values)
    bw_mean, bw_std, bw_ci95 = _mean_std_ci95(bandwidth_values)
    return {
        "delta_centroid_over_centroid_mean": c_mean,
        "delta_centroid_over_centroid_std": c_std,
        "delta_centroid_over_centroid_ci95": c_ci95,
        "delta_bandwidth_over_bandwidth_mean": bw_mean,
        "delta_bandwidth_over_bandwidth_std": bw_std,
        "delta_bandwidth_over_bandwidth_ci95": bw_ci95,
        # Backward-compatible aliases.
        "centroid_ratio": c_mean,
        "bandwidth_ratio": bw_mean,
        "count": len(centroid_values),
    }, param_rows


def _resolve_eps(default_value: float, override_value: Optional[float]) -> float:
    if override_value is None:
        return float(default_value)
    return float(override_value)


def _clip_lower(mean_vals: List[float], spread_vals: List[float]) -> List[float]:
    # Relative-change metrics are non-negative by definition; clip visual lower band at 0.
    return [max(m - s, 0.0) for m, s in zip(mean_vals, spread_vals)]


def _format_safe_values(values: List[float], precision: int = 2) -> str:
    if not values:
        return "none"
    uniq = sorted(set(round(v, precision) for v in values))
    return ", ".join(f"{v:.{precision}f}" for v in uniq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--severity-source",
        type=str,
        default="train",
        choices=["fixed", "train", "val"],
        help="Where severity bins were selected from. Use train/val for no-leakage protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--stft-win-length", type=int, default=None)
    parser.add_argument("--stft-window", type=str, default="hann", choices=["hann", "hamming"])
    parser.add_argument("--stft-center", action="store_true", default=True)
    parser.add_argument("--no-stft-center", dest="stft_center", action="store_false")
    parser.add_argument("--stft-magnitude-power", type=float, default=1.0)
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
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--color-trials", type=int, default=1)

    parser.add_argument("--centroid-epsilon", type=float, default=0.15)
    parser.add_argument("--bandwidth-epsilon", type=float, default=0.15)
    parser.add_argument("--shift-centroid-epsilon", type=float, default=None)
    parser.add_argument("--shift-bandwidth-epsilon", type=float, default=None)
    parser.add_argument("--scale-centroid-epsilon", type=float, default=None)
    parser.add_argument("--scale-bandwidth-epsilon", type=float, default=None)
    parser.add_argument("--color-centroid-epsilon", type=float, default=0.60)
    parser.add_argument("--color-bandwidth-epsilon", type=float, default=0.60)

    parser.add_argument("--max-param-rows", type=int, default=1000)
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    view_config = ViewConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.stft_win_length,
        window_name=args.stft_window,
        center=args.stft_center,
        magnitude_power=args.stft_magnitude_power,
        tf_log1p=args.tf_log1p,
        tf_flatten=args.tf_flatten,
        normalize_mode=args.normalize_mode,
        shift_mode=args.shift_fill,
    )
    preprocess_config = view_config.to_preprocess_config()
    dataset = UEATimeSeriesDataset(
        args.dataset,
        split=args.split,
        pad_to_max=args.pad_to_max,
        view_config=view_config,
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    shift_bins = _parse_float_list(args.shift_bins)
    scale_ratios = _parse_float_list(args.scale_ratios)
    color_levels = _parse_float_list(args.color_max_db)
    color_active_bands = _parse_band_list(args.color_active_bands)
    color_trials = max(1, int(args.color_trials))
    num_bins = dataset.data[0].shape[-1] // 2 + 1

    eps_by_transform = {
        "shift": {
            "centroid": _resolve_eps(args.centroid_epsilon, args.shift_centroid_epsilon),
            "bandwidth": _resolve_eps(args.bandwidth_epsilon, args.shift_bandwidth_epsilon),
        },
        "scale": {
            "centroid": _resolve_eps(args.centroid_epsilon, args.scale_centroid_epsilon),
            "bandwidth": _resolve_eps(args.bandwidth_epsilon, args.scale_bandwidth_epsilon),
        },
        "color": {
            "centroid": _resolve_eps(args.centroid_epsilon, args.color_centroid_epsilon),
            "bandwidth": _resolve_eps(args.bandwidth_epsilon, args.color_bandwidth_epsilon),
        },
    }

    summary_rows = []
    param_rows = []

    def _is_safe(transform_name: str, metrics: Dict[str, float]) -> bool:
        return (
            metrics["delta_centroid_over_centroid_mean"] <= eps_by_transform[transform_name]["centroid"]
            and metrics["delta_bandwidth_over_bandwidth_mean"] <= eps_by_transform[transform_name]["bandwidth"]
        )

    for severity_id, bins in enumerate(shift_bins):
        def _shift_transform(x: torch.Tensor, _batch_idx: int, b: float = bins, sid: int = severity_id):
            if args.shift_mode == "stft":
                out = apply_per_sample_channel(
                    x,
                    lambda s: band_shift_time_stft(
                        s,
                        shift_bins=b,
                        n_fft=view_config.n_fft,
                        hop_length=view_config.hop_length,
                        win_length=view_config.win_length,
                        window_name=view_config.window_name,
                        center=view_config.center,
                        shift_mode=args.shift_fill,
                    ),
                )
            else:
                out = apply_per_sample_channel(x, lambda s: band_shift_time(s, b, shift_mode=args.shift_fill))
            return out, {
                "severity_id": sid,
                "rho": 1.0,
                "g_db": 0.0,
                "b_bins": b,
                "seed": args.seed,
                "shift_fill": args.shift_fill,
            }

        metrics, rows = _signal_eval(loader, view_config, _shift_transform, args.max_param_rows, "shift")
        param_rows.extend(rows[: max(0, args.max_param_rows - len(param_rows))])
        summary_rows.append(
            {
                "transform": "shift",
                "severity_id": severity_id,
                "rho": 1.0,
                "g_db": 0.0,
                "b_bins": bins,
                "trial": 0,
                **metrics,
                "safe": int(_is_safe("shift", metrics)),
                "safe_criterion": "delta_ratio",
                "count": metrics["count"],
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": args.shift_fill,
                "centroid_epsilon": eps_by_transform["shift"]["centroid"],
                "bandwidth_epsilon": eps_by_transform["shift"]["bandwidth"],
            }
        )

    for severity_id, ratio in enumerate(scale_ratios):
        def _scale_transform(x: torch.Tensor, _batch_idx: int, r: float = ratio, sid: int = severity_id):
            out = apply_per_sample_channel(x, lambda s: frequency_scale_time(s, r))
            return out, {
                "severity_id": sid,
                "rho": r,
                "g_db": 0.0,
                "b_bins": 0.0,
                "seed": args.seed,
                "shift_fill": "na",
            }

        metrics, rows = _signal_eval(loader, view_config, _scale_transform, args.max_param_rows, "scale")
        param_rows.extend(rows[: max(0, args.max_param_rows - len(param_rows))])
        summary_rows.append(
            {
                "transform": "scale",
                "severity_id": severity_id,
                "rho": ratio,
                "g_db": 0.0,
                "b_bins": 0.0,
                "trial": 0,
                **metrics,
                "safe": int(_is_safe("scale", metrics)),
                "safe_criterion": "delta_ratio",
                "count": metrics["count"],
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": "na",
                "centroid_epsilon": eps_by_transform["scale"]["centroid"],
                "bandwidth_epsilon": eps_by_transform["scale"]["bandwidth"],
            }
        )

    for severity_id, max_db in enumerate(color_levels):
        trial_rows = []
        for trial in range(color_trials):
            trial_seed = args.seed + severity_id * 1000 + trial
            gains = make_coloring_gains(
                num_bins=num_bins,
                bands=args.color_bands,
                max_gain_db=max_db,
                active_bands=color_active_bands if color_active_bands else None,
                generator=torch.Generator().manual_seed(trial_seed),
            )

            def _color_transform(
                x: torch.Tensor,
                _batch_idx: int,
                g: torch.Tensor = gains,
                sid: int = severity_id,
                db: float = max_db,
                current_seed: int = trial_seed,
            ):
                out = apply_per_sample_channel(x, lambda s: spectral_coloring(s, g))
                return out, {
                    "severity_id": sid,
                    "rho": 1.0,
                    "g_db": db,
                    "b_bins": 0.0,
                    "seed": current_seed,
                    "shift_fill": "na",
                }

            metrics, rows = _signal_eval(loader, view_config, _color_transform, args.max_param_rows, "color")
            param_rows.extend(rows[: max(0, args.max_param_rows - len(param_rows))])
            row = {
                "transform": "color",
                "severity_id": severity_id,
                "rho": 1.0,
                "g_db": max_db,
                "b_bins": 0.0,
                "trial": trial,
                **metrics,
                "safe": int(_is_safe("color", metrics)),
                "safe_criterion": "delta_ratio",
                "count": metrics["count"],
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": "na",
                "centroid_epsilon": eps_by_transform["color"]["centroid"],
                "bandwidth_epsilon": eps_by_transform["color"]["bandwidth"],
            }
            trial_rows.append(row)
            summary_rows.append(row)
        if color_trials > 1 and trial_rows:
            mean_metrics: Dict[str, float] = {}
            metric_keys = [
                "delta_centroid_over_centroid_mean",
                "delta_centroid_over_centroid_std",
                "delta_centroid_over_centroid_ci95",
                "delta_bandwidth_over_bandwidth_mean",
                "delta_bandwidth_over_bandwidth_std",
                "delta_bandwidth_over_bandwidth_ci95",
                "centroid_ratio",
                "bandwidth_ratio",
                "count",
            ]
            for key in metric_keys:
                mean_metrics[key] = sum(float(row[key]) for row in trial_rows) / len(trial_rows)
            summary_rows.append(
                {
                    "transform": "color",
                    "severity_id": severity_id,
                    "rho": 1.0,
                    "g_db": max_db,
                    "b_bins": 0.0,
                    "trial": "mean",
                    **mean_metrics,
                    "safe": int(_is_safe("color", mean_metrics)),
                    "safe_criterion": "delta_ratio",
                    "count": len(dataset),
                    "split": args.split,
                    "severity_source": args.severity_source,
                    "shift_fill": "na",
                    "centroid_epsilon": eps_by_transform["color"]["centroid"],
                    "bandwidth_epsilon": eps_by_transform["color"]["bandwidth"],
                }
            )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    shift_points = sorted([row for row in summary_rows if row["transform"] == "shift"], key=lambda r: float(r["b_bins"]))
    scale_points = sorted([row for row in summary_rows if row["transform"] == "scale"], key=lambda r: float(r["rho"]))
    color_all = [row for row in summary_rows if row["transform"] == "color"]
    color_mean = [row for row in color_all if str(row["trial"]) == "mean"]
    if color_mean:
        color_points = sorted(color_mean, key=lambda r: float(r["g_db"]))
    else:
        color_points = sorted([row for row in color_all if str(row["trial"]) == "0"], key=lambda r: float(r["g_db"]))

    def _plot_delta(
        ax,
        rows: List[Dict[str, object]],
        x_key: str,
        safe_symbol: str,
        title: str,
        xlabel: str,
        centroid_eps: float,
        bandwidth_eps: float,
        show_safe_legend: bool = False,
        force_left_xlim: Optional[float] = None,
    ) -> None:
        x = [float(row[x_key]) for row in rows]
        c_mean = [float(row["delta_centroid_over_centroid_mean"]) for row in rows]
        c_std = [float(row["delta_centroid_over_centroid_std"]) for row in rows]
        bw_mean = [float(row["delta_bandwidth_over_bandwidth_mean"]) for row in rows]
        bw_std = [float(row["delta_bandwidth_over_bandwidth_std"]) for row in rows]
        safe_mask = [int(row.get("safe", 0)) == 1 for row in rows]
        safe_x = [x[i] for i in range(len(x)) if safe_mask[i]]
        safe_c = [c_mean[i] for i in range(len(x)) if safe_mask[i]]
        safe_bw = [bw_mean[i] for i in range(len(x)) if safe_mask[i]]

        ax.plot(x, c_mean, marker="o", label="Delta centroid / centroid")
        ax.plot(x, bw_mean, marker="s", label="Delta bandwidth / bandwidth")
        ax.fill_between(x, _clip_lower(c_mean, c_std), [m + s for m, s in zip(c_mean, c_std)], alpha=0.18)
        ax.fill_between(x, _clip_lower(bw_mean, bw_std), [m + s for m, s in zip(bw_mean, bw_std)], alpha=0.18)
        ax.axhline(centroid_eps, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(bandwidth_eps, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.7)
        if safe_x:
            label = "Selected safe severities (used in Sec. 4.3)" if show_safe_legend else "_nolegend_"
            ax.scatter(safe_x, safe_c, facecolors="none", edgecolors="tab:green", s=56, linewidths=1.2, zorder=4, label=label)
            ax.scatter(safe_x, safe_bw, facecolors="none", edgecolors="tab:green", s=56, linewidths=1.2, zorder=4, label="_nolegend_")
        safe_text = _format_safe_values(safe_x, precision=2)
        ax.text(
            0.02,
            0.96,
            f"safe {safe_symbol}: {safe_text}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative change (Delta metric / metric)")
        ax.set_ylim(bottom=0.0)
        if force_left_xlim is not None and x:
            right = max(x)
            pad = max((right - force_left_xlim) * 0.05, 0.2)
            ax.set_xlim(left=float(force_left_xlim), right=right + pad)
        ax.grid(True, alpha=0.3)

    _plot_delta(
        axes[0],
        shift_points,
        "b_bins",
        "b",
        "Shift Signal Safety",
        "b (shift bins)",
        eps_by_transform["shift"]["centroid"],
        eps_by_transform["shift"]["bandwidth"],
        show_safe_legend=True,
    )
    _plot_delta(
        axes[1],
        scale_points,
        "rho",
        "rho",
        "Scale Signal Safety",
        "rho (scale)",
        eps_by_transform["scale"]["centroid"],
        eps_by_transform["scale"]["bandwidth"],
    )
    _plot_delta(
        axes[2],
        color_points,
        "g_db",
        "g",
        "Color Signal Safety",
        "g (max dB)",
        eps_by_transform["color"]["centroid"],
        eps_by_transform["color"]["bandwidth"],
        force_left_xlim=0.0,
    )
    axes[0].legend(loc="best", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Point=sample-wise mean over split; band=+/-1 std with non-negative lower clip; "
        "green circles are selected safe severities used in Sec. 4.3; "
        "transform-specific thresholds are used (eps_b, eps_rho, eps_g).",
        ha="center",
        fontsize=8.6,
    )
    plt.tight_layout()

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = preprocess_config.hash()
    sev_hash = stable_hash(
        {
            "shift_bins": shift_bins,
            "scale_ratios": scale_ratios,
            "color_max_db": color_levels,
            "shift_mode": args.shift_mode,
            "shift_fill": args.shift_fill,
            "color_bands": args.color_bands,
            "color_active_bands": color_active_bands,
            "color_trials": color_trials,
            "eps_by_transform": eps_by_transform,
        }
    )
    stem = build_tag(
        "safe_signal",
        args.dataset,
        args.split,
        f"seed{args.seed}",
        f"stft{stft_hash}",
        f"sh{len(shift_bins)}-sc{len(scale_ratios)}-co{len(color_levels)}",
        f"sev{sev_hash}",
    )
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    params_csv = csv_dir / f"{stem}_params.csv"
    safe_shift_csv = csv_dir / f"{stem}_safe_shift.csv"
    selected_safe_csv = csv_dir / f"{stem}_selected_safe_severities.csv"

    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, summary_rows)
    write_csv(params_csv, param_rows[: args.max_param_rows])

    safe_shift_rows = []
    for row in shift_points:
        safe_shift_rows.append(
            {
                "b_bins": float(row["b_bins"]),
                "delta_centroid_over_centroid_mean": float(row["delta_centroid_over_centroid_mean"]),
                "delta_centroid_over_centroid_std": float(row["delta_centroid_over_centroid_std"]),
                "delta_bandwidth_over_bandwidth_mean": float(row["delta_bandwidth_over_bandwidth_mean"]),
                "delta_bandwidth_over_bandwidth_std": float(row["delta_bandwidth_over_bandwidth_std"]),
                "safe": int(row["safe"]),
                "safe_criterion": "delta_ratio",
                "centroid_epsilon": float(row["centroid_epsilon"]),
                "bandwidth_epsilon": float(row["bandwidth_epsilon"]),
                "shift_fill": args.shift_fill,
                "shift_mode": args.shift_mode,
            }
        )
    write_csv(safe_shift_csv, safe_shift_rows)

    selected_safe_rows = []
    for row in shift_points:
        if int(row["safe"]) != 1:
            continue
        selected_safe_rows.append(
            {
                "transform": "shift",
                "severity_id": int(row["severity_id"]),
                "severity_value": float(row["b_bins"]),
                "severity_symbol": "b",
                "safe": 1,
                "split": args.split,
                "safe_criterion": str(row["safe_criterion"]),
                "delta_centroid_over_centroid_mean": float(row["delta_centroid_over_centroid_mean"]),
                "delta_bandwidth_over_bandwidth_mean": float(row["delta_bandwidth_over_bandwidth_mean"]),
                "centroid_epsilon": float(row["centroid_epsilon"]),
                "bandwidth_epsilon": float(row["bandwidth_epsilon"]),
            }
        )
    for row in scale_points:
        if int(row["safe"]) != 1:
            continue
        selected_safe_rows.append(
            {
                "transform": "scale",
                "severity_id": int(row["severity_id"]),
                "severity_value": float(row["rho"]),
                "severity_symbol": "rho",
                "safe": 1,
                "split": args.split,
                "safe_criterion": str(row["safe_criterion"]),
                "delta_centroid_over_centroid_mean": float(row["delta_centroid_over_centroid_mean"]),
                "delta_bandwidth_over_bandwidth_mean": float(row["delta_bandwidth_over_bandwidth_mean"]),
                "centroid_epsilon": float(row["centroid_epsilon"]),
                "bandwidth_epsilon": float(row["bandwidth_epsilon"]),
            }
        )
    for row in color_points:
        if int(row["safe"]) != 1:
            continue
        selected_safe_rows.append(
            {
                "transform": "color",
                "severity_id": int(row["severity_id"]),
                "severity_value": float(row["g_db"]),
                "severity_symbol": "g",
                "safe": 1,
                "split": args.split,
                "safe_criterion": str(row["safe_criterion"]),
                "delta_centroid_over_centroid_mean": float(row["delta_centroid_over_centroid_mean"]),
                "delta_bandwidth_over_bandwidth_mean": float(row["delta_bandwidth_over_bandwidth_mean"]),
                "centroid_epsilon": float(row["centroid_epsilon"]),
                "bandwidth_epsilon": float(row["bandwidth_epsilon"]),
            }
        )
    write_csv(selected_safe_csv, selected_safe_rows)

    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/safe_signal.py",
        device="cpu",
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "params_csv": str(params_csv),
            "safe_shift_csv": str(safe_shift_csv),
            "selected_safe_csv": str(selected_safe_csv),
            "stft_hash": stft_hash,
            "severity_hash": sev_hash,
            "safe_b_note": (
                "Safe-B uses sample-wise relative changes: Delta centroid / centroid and "
                "Delta bandwidth / bandwidth; transform-specific thresholds supported."
            ),
            "thresholds": eps_by_transform,
            "leakage_guard": "No safety threshold tuning is performed on split=test.",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_params_csv={params_csv}")
    print(f"saved_safe_shift_csv={safe_shift_csv}")
    print(f"saved_selected_safe_csv={selected_safe_csv}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
