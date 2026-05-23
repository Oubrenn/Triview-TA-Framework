import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, write_csv, write_run_meta  # noqa: E402
from preprocessing import build_augmented_triviews, build_triview_from_time  # noqa: E402
from transforms import make_coloring_gains  # noqa: E402


def _parse_int_list(raw: str):
    if raw is None:
        return None
    cleaned = [item.strip() for item in raw.split(",") if item.strip()]
    if not cleaned:
        return None
    return [int(item) for item in cleaned]


def _to_2d_map(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x[0]
    raise ValueError(f"Expected 1D/2D/3D tensor for visualization, got shape={tuple(x.shape)}.")


def _to_1d_spectrum(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        if x.shape[0] == 1:
            return x[0]
        return x.mean(dim=0)
    if x.dim() == 3:
        if x.shape[0] == 1:
            return _to_1d_spectrum(x[0])
        return _to_1d_spectrum(x.mean(dim=0))
    raise ValueError(f"Expected 1D/2D/3D tensor for spectrum, got shape={tuple(x.shape)}.")


def _plot_row(ax_time, ax_freq, ax_tf, x_time: torch.Tensor, x_freq: torch.Tensor, x_tf: torch.Tensor, row_name: str):
    time_wave = x_time[0].detach().cpu().numpy() if x_time.dim() == 2 else x_time.detach().cpu().numpy()
    ax_time.plot(time_wave, linewidth=1.2)
    ax_time.set_title(f"{row_name} time")
    ax_time.set_xlabel("t")

    freq_curve = _to_1d_spectrum(x_freq).detach().cpu().numpy()
    ax_freq.plot(freq_curve, linewidth=1.2)
    ax_freq.set_title(f"{row_name} freq")
    ax_freq.set_xlabel("f-bin")
    ax_freq.set_ylabel("magnitude")

    tf_map = _to_2d_map(x_tf).detach().cpu().numpy()
    ax_tf.imshow(tf_map, aspect="auto", origin="lower")
    ax_tf.set_title(f"{row_name} tf")
    ax_tf.set_xlabel("t-frame")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
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
    parser.add_argument("--shift-bins", type=float, default=0.1)
    parser.add_argument("--shift-fill", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--scale-ratio", type=float, default=1.1)
    parser.add_argument("--color-max-gain-db", type=float, default=6.0)
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    color_active_bands = _parse_int_list(args.color_active_bands)
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
        shift_bins=[args.shift_bins],
        scale_ratios=[args.scale_ratio],
        color_bands=args.color_bands,
        color_max_gain_db=args.color_max_gain_db,
        color_active_bands=color_active_bands,
    )
    preprocess_config = view_config.to_preprocess_config()

    dataset = UEATimeSeriesDataset(
        args.dataset,
        split=args.split,
        pad_to_max=args.pad_to_max,
        view_config=view_config,
        normalize=True,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range: {args.index} not in [0, {len(dataset) - 1}]")

    sample = dataset[args.index]
    x_time = sample["x_time"]
    clean_views = build_triview_from_time(x_time, preprocess_config)

    gains, band_gains = make_coloring_gains(
        num_bins=x_time.shape[-1] // 2 + 1,
        bands=view_config.color_bands,
        max_gain_db=view_config.color_max_gain_db,
        return_band_gains=True,
        active_bands=view_config.color_active_bands,
        generator=torch.Generator().manual_seed(args.seed),
    )
    augmented = build_augmented_triviews(
        x=x_time,
        config=preprocess_config,
        shift_bins=args.shift_bins,
        scale_ratio=args.scale_ratio,
        color_gains=gains,
        shift_mode=args.shift_fill,
    )

    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    _plot_row(
        axes[0, 0],
        axes[0, 1],
        axes[0, 2],
        x_time,
        clean_views["x_freq"],
        clean_views["x_tf"],
        "clean",
    )
    _plot_row(
        axes[1, 0],
        axes[1, 1],
        axes[1, 2],
        x_time,
        augmented["x_shift_freq"],
        augmented["x_shift_tf"],
        f"shift(b={args.shift_bins})",
    )
    _plot_row(
        axes[2, 0],
        axes[2, 1],
        axes[2, 2],
        augmented["x_scale"],
        augmented["x_scale_freq"],
        augmented["x_scale_tf"],
        f"scale(rho={args.scale_ratio})",
    )
    _plot_row(
        axes[3, 0],
        axes[3, 1],
        axes[3, 2],
        augmented["x_color"],
        augmented["x_color_freq"],
        augmented["x_color_tf"],
        f"color(g={args.color_max_gain_db}dB)",
    )
    plt.tight_layout()

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = preprocess_config.hash()
    severity_tag = f"rho{args.scale_ratio}_g{args.color_max_gain_db}_b{args.shift_bins}"
    stem = build_tag(
        "triview",
        args.dataset,
        args.split,
        f"idx{args.index}",
        f"seed{args.seed}",
        f"stft{stft_hash}",
        severity_tag,
    )
    fig_path = figs_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    csv_path = csv_dir / f"{stem}.csv"
    write_csv(
        csv_path,
        [
            {
                "dataset": args.dataset,
                "split": args.split,
                "index": args.index,
                "seed": args.seed,
                "stft_hash": stft_hash,
                "rho": args.scale_ratio,
                "g_db": args.color_max_gain_db,
                "b_bins": args.shift_bins,
                "severity_id": 0,
                "shift_fill": args.shift_fill,
                "band_gains": "|".join(f"{v:.6f}" for v in band_gains.tolist()),
            }
        ],
    )
    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/vis_triview.py",
        device="cpu",
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "csv": str(csv_path),
            "stft_hash": stft_hash,
            "shift_fill": args.shift_fill,
            "x_freq_definition": "|STFT| then mean over time",
            "x_tf_definition": "flatten (C*F, T) then optional log1p",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_csv={csv_path}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
