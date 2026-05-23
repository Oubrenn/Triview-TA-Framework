import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, write_csv, write_run_meta  # noqa: E402
from preprocessing import PreprocessConfig, compute_stft_magnitude  # noqa: E402


def _parse_int_list(raw: str) -> List[int]:
    text = (raw or "").strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _resolve_mag(stft_mag: torch.Tensor, channel_mode: str) -> torch.Tensor:
    if stft_mag.dim() == 2:
        return stft_mag
    if stft_mag.dim() != 3:
        raise ValueError(f"Expected STFT magnitude with shape (F, T) or (C, F, T), got {tuple(stft_mag.shape)}.")

    if channel_mode == "first":
        return stft_mag[0]
    if channel_mode == "mean_stft":
        return stft_mag.mean(dim=0)
    raise ValueError(f"Unsupported channel_mode: {channel_mode}")


def _pool_freq(mag_2d: torch.Tensor, pool_mode: str) -> torch.Tensor:
    if mag_2d.dim() != 2:
        raise ValueError(f"Expected 2D magnitude map (F, T), got {tuple(mag_2d.shape)}")
    if pool_mode == "mean":
        return mag_2d.mean(dim=-1)
    if pool_mode == "max":
        return mag_2d.max(dim=-1).values
    raise ValueError(f"Unsupported pool_mode: {pool_mode}")


def _smooth_curve(curve: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return curve
    if kernel_size % 2 == 0:
        raise ValueError("smooth_kernel must be odd to keep the curve aligned.")
    weight = torch.ones(1, 1, kernel_size, dtype=curve.dtype, device=curve.device) / float(kernel_size)
    padded = F.pad(curve.view(1, 1, -1), (kernel_size // 2, kernel_size // 2), mode="replicate")
    smoothed = F.conv1d(padded, weight)
    return smoothed.view(-1)


def _percentile_limits(tf_map: torch.Tensor, low_p: float, high_p: float) -> Tuple[float, float]:
    if low_p < 0 or low_p > 100 or high_p < 0 or high_p > 100:
        raise ValueError("Percentiles must be in [0, 100].")
    if low_p >= high_p:
        raise ValueError("tf-vmin percentile must be smaller than tf-vmax percentile.")
    flat = tf_map.detach().to(dtype=torch.float32).reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return 0.0, 1.0
    q_low = torch.quantile(flat, q=float(low_p) / 100.0)
    q_high = torch.quantile(flat, q=float(high_p) / 100.0)
    lo = float(q_low.item())
    hi = float(q_high.item())
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _enhance_tf_structure(
    tf_map: torch.Tensor,
    alpha: float,
    kernel_freq: int,
    kernel_time: int,
) -> torch.Tensor:
    if alpha <= 0:
        return tf_map
    if kernel_freq < 1 or kernel_time < 1:
        raise ValueError("Contrast kernels must be >= 1.")
    if kernel_freq % 2 == 0 or kernel_time % 2 == 0:
        raise ValueError("Contrast kernels must be odd.")

    x = tf_map.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    pad_f = kernel_freq // 2
    pad_t = kernel_time // 2
    x_pad = F.pad(x, (pad_t, pad_t, pad_f, pad_f), mode="replicate")
    blur = F.avg_pool2d(x_pad, kernel_size=(kernel_freq, kernel_time), stride=1)
    enhanced = x + float(alpha) * (x - blur)
    return enhanced.squeeze(0).squeeze(0)


def _extract_views(
    x_time: torch.Tensor,
    preprocess_config: PreprocessConfig,
    channel_mode: str,
    freq_pool: str,
    tf_log1p: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    stft_mag = compute_stft_magnitude(x_time, preprocess_config)
    mag_2d = _resolve_mag(stft_mag, channel_mode=channel_mode)
    freq_curve = _pool_freq(mag_2d, pool_mode=freq_pool)
    tf_map = mag_2d.log1p() if tf_log1p else mag_2d
    return freq_curve, tf_map


def _select_experiment_indices(
    dataset_len: int,
    num_samples: int,
    seed: int,
    anchor_index: int,
    manual_indices: Sequence[int],
) -> List[int]:
    if dataset_len <= 0:
        return []
    if manual_indices:
        cleaned = []
        for idx in manual_indices:
            if idx < 0 or idx >= dataset_len:
                raise IndexError(f"Experiment index out of range: {idx} not in [0, {dataset_len - 1}]")
            if idx not in cleaned:
                cleaned.append(idx)
        if anchor_index not in cleaned:
            cleaned.append(anchor_index)
        return cleaned

    count = max(1, min(int(num_samples), dataset_len))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(dataset_len, generator=generator).tolist()
    chosen = perm[:count]
    if anchor_index not in chosen:
        chosen[-1] = anchor_index
    return chosen


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    score = F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=1, eps=1e-12)
    return float(score.item())


def _run_consistency_experiment(
    dataset: UEATimeSeriesDataset,
    indices: Sequence[int],
    preprocess_config: PreprocessConfig,
    channel_mode: str,
    freq_pool: str,
    tf_log1p: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    mae_values: List[float] = []
    max_values: List[float] = []
    cosine_values: List[float] = []

    for idx in indices:
        sample = dataset[int(idx)]
        x_time = sample["x_time"]
        stft_mag = compute_stft_magnitude(x_time, preprocess_config)
        mag_2d = _resolve_mag(stft_mag, channel_mode=channel_mode)
        freq_from_mag = _pool_freq(mag_2d, pool_mode=freq_pool)

        tf_map = mag_2d.log1p() if tf_log1p else mag_2d
        recovered_mag = torch.expm1(tf_map) if tf_log1p else tf_map
        freq_from_tf = _pool_freq(recovered_mag, pool_mode=freq_pool)

        diff = (freq_from_mag - freq_from_tf).abs()
        mae = float(diff.mean().item())
        max_err = float(diff.max().item())
        cosine = _cosine(freq_from_mag, freq_from_tf)

        mae_values.append(mae)
        max_values.append(max_err)
        cosine_values.append(cosine)

        label_value = sample.get("y")
        label_int = int(label_value.item()) if torch.is_tensor(label_value) else -1

        rows.append(
            {
                "index": int(idx),
                "label": label_int,
                "freq_bins": int(freq_from_mag.shape[0]),
                "tf_frames": int(tf_map.shape[-1]),
                "consistency_mae": mae,
                "consistency_max_err": max_err,
                "consistency_cosine": cosine,
            }
        )

    if not rows:
        summary = {
            "num_samples": 0.0,
            "mae_mean": 0.0,
            "mae_std": 0.0,
            "max_err_mean": 0.0,
            "max_err_std": 0.0,
            "cosine_mean": 0.0,
            "cosine_std": 0.0,
        }
        return rows, summary

    mae_t = torch.tensor(mae_values, dtype=torch.float32)
    max_t = torch.tensor(max_values, dtype=torch.float32)
    cos_t = torch.tensor(cosine_values, dtype=torch.float32)
    summary = {
        "num_samples": float(len(rows)),
        "mae_mean": float(mae_t.mean().item()),
        "mae_std": float(mae_t.std(unbiased=False).item()) if len(rows) > 1 else 0.0,
        "max_err_mean": float(max_t.mean().item()),
        "max_err_std": float(max_t.std(unbiased=False).item()) if len(rows) > 1 else 0.0,
        "cosine_mean": float(cos_t.mean().item()),
        "cosine_std": float(cos_t.std(unbiased=False).item()) if len(rows) > 1 else 0.0,
    }
    return rows, summary


def _plot_frequency(
    curve: torch.Tensor,
    out_path: Path,
    title: str,
    dpi: int,
    *,
    thumbnail_mode: bool,
    max_bins: Optional[int],
    normalize_curve: bool,
) -> None:
    plot_curve = curve.clone()
    if max_bins is not None and max_bins > 0:
        plot_curve = plot_curve[: int(max_bins)]
    if normalize_curve:
        denom = float(plot_curve.max().item()) if plot_curve.numel() > 0 else 1.0
        plot_curve = plot_curve / (denom + 1e-8)

    y = plot_curve.detach().cpu().numpy()
    x = torch.arange(plot_curve.numel()).numpy()
    if thumbnail_mode:
        fig, ax = plt.subplots(figsize=(2.2, 1.5))
    else:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x, y, linewidth=1.6 if thumbnail_mode else 1.8, color="#2F6FB0")

    if thumbnail_mode:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#222222")
        fig.tight_layout(pad=0.1)
    else:
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.5)
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_tf(
    tf_map: torch.Tensor,
    out_path: Path,
    title: str,
    dpi: int,
    cmap: str,
    colorbar: bool,
    *,
    thumbnail_mode: bool,
    interpolation: str,
    vmin_pctl: float,
    vmax_pctl: float,
    structure_alpha: float,
    structure_kernel_freq: int,
    structure_kernel_time: int,
) -> None:
    tf_display = _enhance_tf_structure(
        tf_map=tf_map,
        alpha=structure_alpha,
        kernel_freq=structure_kernel_freq,
        kernel_time=structure_kernel_time,
    )
    arr = tf_display.detach().cpu().numpy()
    vmin, vmax = _percentile_limits(tf_display, low_p=vmin_pctl, high_p=vmax_pctl)
    if thumbnail_mode:
        fig, ax = plt.subplots(figsize=(2.2, 1.5))
    else:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
    )
    if thumbnail_mode:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#222222")
    else:
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Frequency bin")
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout(pad=0.1 if thumbnail_mode else 0.3)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export two triview-consistent images from one real sample: "
            "frequency view (time-pooled |STFT|) and time-frequency view (log1p(|STFT|))."
        )
    )
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    parser.add_argument("--normalize-mode", type=str, default="per_sample_channel", choices=["per_sample_channel", "none"])

    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--stft-win-length", type=int, default=None)
    parser.add_argument("--stft-window", type=str, default="hann", choices=["hann", "hamming"])
    parser.add_argument("--stft-center", action="store_true", default=True)
    parser.add_argument("--no-stft-center", dest="stft_center", action="store_false")
    parser.add_argument("--stft-magnitude-power", type=float, default=1.0)
    parser.add_argument("--tf-log1p", action="store_true", default=True)
    parser.add_argument("--no-tf-log1p", dest="tf_log1p", action="store_false")
    parser.add_argument("--channel-mode", type=str, default="mean_stft", choices=["first", "mean_stft"])
    parser.add_argument("--freq-pool", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--smooth-kernel", type=int, default=5)

    parser.add_argument("--experiment-samples", type=int, default=32)
    parser.add_argument("--experiment-indices", type=str, default="")

    parser.add_argument("--cmap", type=str, default="Blues")
    parser.add_argument("--tf-vmin-pctl", type=float, default=5.0)
    parser.add_argument("--tf-vmax-pctl", type=float, default=99.0)
    parser.add_argument("--tf-interpolation", type=str, default="bicubic")
    parser.add_argument("--tf-structure-alpha", type=float, default=0.30)
    parser.add_argument("--tf-structure-kernel-freq", type=int, default=7)
    parser.add_argument("--tf-structure-kernel-time", type=int, default=3)
    parser.add_argument("--thumbnail-mode", action="store_true", default=True)
    parser.add_argument("--no-thumbnail-mode", dest="thumbnail_mode", action="store_false")
    parser.add_argument("--freq-max-bins", type=int, default=50)
    parser.add_argument("--freq-normalize", action="store_true", default=True)
    parser.add_argument("--no-freq-normalize", dest="freq_normalize", action="store_false")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--colorbar", dest="colorbar", action="store_true")
    parser.add_argument("--no-colorbar", dest="colorbar", action="store_false")
    parser.add_argument("--output-root", type=str, default="outputs_two_views")
    parser.set_defaults(colorbar=False)
    args = parser.parse_args()

    if args.smooth_kernel < 1:
        raise ValueError("--smooth-kernel must be >= 1")
    if args.smooth_kernel % 2 == 0:
        raise ValueError("--smooth-kernel must be odd")

    torch.manual_seed(args.seed)
    view_config = ViewConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.stft_win_length,
        window_name=args.stft_window,
        center=args.stft_center,
        magnitude_power=args.stft_magnitude_power,
        tf_log1p=args.tf_log1p,
        tf_flatten=False,
        normalize_mode=args.normalize_mode,
    )
    preprocess_config = PreprocessConfig(
        n_fft=view_config.n_fft,
        hop_length=view_config.hop_length,
        win_length=view_config.win_length,
        window_name=view_config.window_name,
        center=view_config.center,
        magnitude_power=view_config.magnitude_power,
        tf_log1p=view_config.tf_log1p,
        tf_flatten=False,
        normalize_mode=view_config.normalize_mode,
    )

    dataset = UEATimeSeriesDataset(
        name=args.dataset,
        split=args.split,
        normalize=True,
        pad_to_max=args.pad_to_max,
        return_freq=False,
        view_config=view_config,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range: {args.index} not in [0, {len(dataset) - 1}]")

    sample = dataset[args.index]
    x_time = sample["x_time"]
    freq_curve, tf_map = _extract_views(
        x_time=x_time,
        preprocess_config=preprocess_config,
        channel_mode=args.channel_mode,
        freq_pool=args.freq_pool,
        tf_log1p=args.tf_log1p,
    )
    freq_curve_smoothed = _smooth_curve(freq_curve, kernel_size=args.smooth_kernel)

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = preprocess_config.hash()
    stem = build_tag(
        "two_views",
        args.dataset,
        args.split,
        f"idx{args.index}",
        f"seed{args.seed}",
        f"stft{stft_hash}",
        args.channel_mode,
        args.freq_pool,
    )
    freq_fig_path = figs_dir / f"{stem}_frequency_view.png"
    tf_fig_path = figs_dir / f"{stem}_time_frequency_view.png"

    _plot_frequency(
        curve=freq_curve_smoothed,
        out_path=freq_fig_path,
        title="Frequency view: pooled |STFT|",
        dpi=args.dpi,
        thumbnail_mode=bool(args.thumbnail_mode),
        max_bins=args.freq_max_bins if args.freq_max_bins > 0 else None,
        normalize_curve=bool(args.freq_normalize),
    )
    _plot_tf(
        tf_map=tf_map,
        out_path=tf_fig_path,
        title="Time-frequency view: log1p(|STFT|)" if args.tf_log1p else "Time-frequency view: |STFT|",
        dpi=args.dpi,
        cmap=args.cmap,
        colorbar=bool(args.colorbar),
        thumbnail_mode=bool(args.thumbnail_mode),
        interpolation=args.tf_interpolation,
        vmin_pctl=float(args.tf_vmin_pctl),
        vmax_pctl=float(args.tf_vmax_pctl),
        structure_alpha=float(args.tf_structure_alpha),
        structure_kernel_freq=int(args.tf_structure_kernel_freq),
        structure_kernel_time=int(args.tf_structure_kernel_time),
    )

    manual_indices = _parse_int_list(args.experiment_indices)
    eval_indices = _select_experiment_indices(
        dataset_len=len(dataset),
        num_samples=args.experiment_samples,
        seed=args.seed,
        anchor_index=args.index,
        manual_indices=manual_indices,
    )
    exp_rows, exp_summary = _run_consistency_experiment(
        dataset=dataset,
        indices=eval_indices,
        preprocess_config=preprocess_config,
        channel_mode=args.channel_mode,
        freq_pool=args.freq_pool,
        tf_log1p=args.tf_log1p,
    )
    exp_csv_path = csv_dir / f"{stem}_consistency_rows.csv"
    exp_summary_csv_path = csv_dir / f"{stem}_consistency_summary.csv"
    write_csv(exp_csv_path, exp_rows)
    write_csv(exp_summary_csv_path, [exp_summary])

    summary_txt_path = output_root / f"{stem}_experiment_summary.txt"
    summary_lines = [
        f"dataset={args.dataset}",
        f"split={args.split}",
        f"index={args.index}",
        f"channel_mode={args.channel_mode}",
        f"freq_pool={args.freq_pool}",
        f"tf_log1p={args.tf_log1p}",
        f"stft_hash={stft_hash}",
        f"thumbnail_mode={args.thumbnail_mode}",
        f"cmap={args.cmap}",
        f"tf_percentiles={args.tf_vmin_pctl},{args.tf_vmax_pctl}",
        f"tf_interpolation={args.tf_interpolation}",
        f"tf_structure_alpha={args.tf_structure_alpha}",
        f"tf_structure_kernel={args.tf_structure_kernel_freq}x{args.tf_structure_kernel_time}",
        f"num_samples={int(exp_summary['num_samples'])}",
        f"mae_mean={exp_summary['mae_mean']:.8e}",
        f"mae_std={exp_summary['mae_std']:.8e}",
        f"max_err_mean={exp_summary['max_err_mean']:.8e}",
        f"max_err_std={exp_summary['max_err_std']:.8e}",
        f"cosine_mean={exp_summary['cosine_mean']:.8f}",
        f"cosine_std={exp_summary['cosine_std']:.8f}",
        f"frequency_figure={freq_fig_path}",
        f"time_frequency_figure={tf_fig_path}",
        f"rows_csv={exp_csv_path}",
        f"summary_csv={exp_summary_csv_path}",
    ]
    summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/export_two_views_experiment.py",
        device="cpu",
        config=vars(args),
        extra={
            "stft_hash": stft_hash,
            "frequency_figure": str(freq_fig_path),
            "time_frequency_figure": str(tf_fig_path),
            "consistency_rows_csv": str(exp_csv_path),
            "consistency_summary_csv": str(exp_summary_csv_path),
            "experiment_summary_txt": str(summary_txt_path),
            "x_freq_definition": "|STFT| pooled over time",
            "x_tf_definition": "log1p(|STFT|)",
            "display_only_adjustments": {
                "thumbnail_mode": bool(args.thumbnail_mode),
                "cmap": args.cmap,
                "tf_vmin_pctl": float(args.tf_vmin_pctl),
                "tf_vmax_pctl": float(args.tf_vmax_pctl),
                "tf_interpolation": args.tf_interpolation,
                "tf_structure_alpha": float(args.tf_structure_alpha),
                "tf_structure_kernel_freq": int(args.tf_structure_kernel_freq),
                "tf_structure_kernel_time": int(args.tf_structure_kernel_time),
            },
        },
    )

    print(f"saved_frequency_figure={freq_fig_path}")
    print(f"saved_time_frequency_figure={tf_fig_path}")
    print(f"saved_experiment_rows={exp_csv_path}")
    print(f"saved_experiment_summary={exp_summary_csv_path}")
    print(f"saved_summary_txt={summary_txt_path}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
