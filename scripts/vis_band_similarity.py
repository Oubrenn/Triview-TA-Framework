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
from models import MultiViewModel  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, write_csv, write_run_meta  # noqa: E402
from preprocessing import build_triview_from_time, ensure_ct_sample  # noqa: E402
from transforms import (  # noqa: E402
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
)


def _parse_int_list(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _apply_per_channel(x: torch.Tensor, fn) -> torch.Tensor:
    x = ensure_ct_sample(x, "x")
    if x.shape[0] == 1:
        return fn(x.squeeze(0)).unsqueeze(0)
    return torch.stack([fn(x[idx]) for idx in range(x.shape[0])], dim=0)


def _load_checkpoint(path: Path, device: str) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict with config/model_state: {path}")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {path}")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing model_state dict: {path}")
    return config, state


def _resolve_view_config(args, config: Dict[str, object]) -> ViewConfig:
    return ViewConfig(
        n_fft=args.n_fft if args.n_fft is not None else int(config.get("n_fft", 256)),
        hop_length=args.hop_length if args.hop_length is not None else int(config.get("hop_length", 64)),
        win_length=args.stft_win_length if args.stft_win_length is not None else config.get("stft_win_length"),
        window_name=args.stft_window or str(config.get("stft_window", "hann")),
        center=bool(config.get("stft_center", args.stft_center)),
        magnitude_power=(
            args.stft_magnitude_power
            if args.stft_magnitude_power is not None
            else float(config.get("stft_magnitude_power", 1.0))
        ),
        tf_log1p=bool(config.get("tf_log1p", args.tf_log1p)),
        tf_flatten=bool(config.get("tf_flatten", args.tf_flatten)),
        normalize_mode=args.normalize_mode,
        shift_mode=args.shift_fill,
    )


def _build_band_slices(num_bins: int, bands: int) -> List[Tuple[int, int]]:
    if bands <= 0:
        raise ValueError("bands must be positive.")
    slices: List[Tuple[int, int]] = []
    for idx in range(bands):
        start = int(idx * num_bins / bands)
        end = int((idx + 1) * num_bins / bands)
        if end <= start:
            end = min(num_bins, start + 1)
        slices.append((start, end))
    if slices:
        slices[-1] = (slices[-1][0], num_bins)
    return slices


def _mask_freq_band(x_freq: torch.Tensor, band_slice: Tuple[int, int]) -> torch.Tensor:
    x_freq = ensure_ct_sample(x_freq, "x_freq")
    if x_freq.dim() != 2:
        raise ValueError(f"x_freq must follow (C, F), got {tuple(x_freq.shape)}")
    start, end = band_slice
    masked = torch.zeros_like(x_freq)
    masked[:, start:end] = x_freq[:, start:end]
    return masked


def _mask_tf_band(
    x_tf: torch.Tensor,
    band_slice: Tuple[int, int],
    channels: int,
    freq_bins: int,
) -> torch.Tensor:
    if x_tf.dim() == 2:
        frames = x_tf.shape[-1]
        if x_tf.shape[0] != channels * freq_bins:
            raise ValueError(
                "Flattened TF shape mismatch: "
                f"expected C*F={channels * freq_bins}, got {x_tf.shape[0]}"
            )
        tf_map = x_tf.view(channels, freq_bins, frames)
        masked = torch.zeros_like(tf_map)
        start, end = band_slice
        masked[:, start:end, :] = tf_map[:, start:end, :]
        return masked.view(channels * freq_bins, frames)
    if x_tf.dim() == 3:
        if x_tf.shape[0] != channels or x_tf.shape[1] != freq_bins:
            raise ValueError(
                "Non-flattened TF shape mismatch: "
                f"expected ({channels}, {freq_bins}, T), got {tuple(x_tf.shape)}"
            )
        masked = torch.zeros_like(x_tf)
        start, end = band_slice
        masked[:, start:end, :] = x_tf[:, start:end, :]
        return masked
    raise ValueError(f"x_tf must follow (C*F, T) or (C, F, T), got {tuple(x_tf.shape)}")


def _build_model(
    config: Dict[str, object],
    input_dim_time: int,
    input_dim_freq: int,
    input_dim_tf: int,
    device: str,
) -> MultiViewModel:
    return MultiViewModel(
        input_dim_time=input_dim_time,
        input_dim_freq=input_dim_freq,
        input_dim_tf=input_dim_tf,
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


def _encode_band_matrix(
    model: MultiViewModel,
    x_freq_anchor: torch.Tensor,
    x_tf_target: torch.Tensor,
    band_slices: Sequence[Tuple[int, int]],
    channels: int,
    freq_bins: int,
) -> torch.Tensor:
    freq_embeds = []
    tf_embeds = []
    for band_slice in band_slices:
        x_freq_masked = _mask_freq_band(x_freq_anchor, band_slice).unsqueeze(0)
        x_tf_masked = _mask_tf_band(x_tf_target, band_slice, channels, freq_bins).unsqueeze(0)
        h_freq = model.freq_encoder(x_freq_masked)
        h_tf = model.tf_encoder(x_tf_masked)
        z_freq = model.freq_projector(h_freq).squeeze(0)
        z_tf = model.tf_projector(h_tf).squeeze(0)
        freq_embeds.append(z_freq)
        tf_embeds.append(z_tf)
    freq_m = F.normalize(torch.stack(freq_embeds, dim=0), p=2, dim=-1, eps=1e-6)
    tf_m = F.normalize(torch.stack(tf_embeds, dim=0), p=2, dim=-1, eps=1e-6)
    return freq_m @ tf_m.transpose(0, 1)


def _apply_transform(x: torch.Tensor, args, view_config: ViewConfig, sample_seed: int) -> torch.Tensor:
    if args.transform == "none":
        return x
    if args.transform == "shift":
        return _apply_per_channel(
            x,
            lambda c: band_shift_time_stft(
                c,
                shift_bins=args.shift_bins,
                n_fft=view_config.n_fft,
                hop_length=view_config.hop_length,
                win_length=view_config.win_length,
                window_name=view_config.window_name,
                center=view_config.center,
                shift_mode=args.shift_fill,
            ),
        )
    if args.transform == "scale":
        return _apply_per_channel(x, lambda c: frequency_scale_time(c, args.scale_ratio))

    num_bins = x.shape[-1] // 2 + 1
    active_bands = _parse_int_list(args.color_active_bands)
    gains = make_coloring_gains(
        num_bins=num_bins,
        bands=args.color_bands,
        max_gain_db=args.color_max_gain_db,
        active_bands=active_bands if active_bands else None,
        generator=torch.Generator().manual_seed(sample_seed),
    )
    return _apply_per_channel(x, lambda c: spectral_coloring(c, gains))


def _find_band_index(position: float, band_slices: Sequence[Tuple[int, int]]) -> int:
    for idx, (start, end) in enumerate(band_slices):
        if position < end:
            return idx
    return len(band_slices) - 1


def _expected_mapping(
    band_slices: Sequence[Tuple[int, int]],
    transform: str,
    shift_bins: float,
) -> List[int]:
    if transform != "shift":
        return list(range(len(band_slices)))
    mapping = []
    for start, end in band_slices:
        center = 0.5 * (start + end - 1)
        shifted_center = max(0.0, center + shift_bins)
        mapping.append(_find_band_index(shifted_center, band_slices))
    return mapping


def _matching_metrics(matrix: torch.Tensor, expected: Sequence[int]) -> Dict[str, float]:
    predicted = matrix.argmax(dim=1)
    expected_t = torch.tensor(expected, dtype=torch.long, device=matrix.device)
    top1 = float((predicted == expected_t).to(dtype=torch.float32).mean().item())
    tol1 = float((predicted - expected_t).abs().le(1).to(dtype=torch.float32).mean().item())
    diag_vals = matrix.diag()
    offdiag_mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    offdiag_vals = matrix[offdiag_mask]
    return {
        "top1": top1,
        "top1_tol1": tol1,
        "diag_mean": float(diag_vals.mean().item()),
        "offdiag_mean": float(offdiag_vals.mean().item()) if offdiag_vals.numel() > 0 else 0.0,
    }


def _align_shift_matrix(matrix: torch.Tensor, shift_bins: float, band_slices: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not band_slices:
        return matrix
    widths = [end - start for start, end in band_slices]
    mean_width = float(sum(widths) / len(widths))
    if mean_width <= 0:
        return matrix
    delta = int(round(shift_bins / mean_width))
    if delta == 0:
        return matrix
    return torch.roll(matrix, shifts=-delta, dims=1)


def _write_matrix_csv(path: Path, matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = matrix.detach().cpu()
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in matrix:
            handle.write(",".join(f"{float(v):.8g}" for v in row.tolist()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
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
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--transform", type=str, default="none", choices=["none", "shift", "scale", "color"])
    parser.add_argument("--shift-bins", type=float, default=0.0)
    parser.add_argument("--shift-fill", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--scale-ratio", type=float, default=1.0)
    parser.add_argument("--color-max-gain-db", type=float, default=0.0)
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--align-shift", action="store_true", default=False)
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    config, state = _load_checkpoint(checkpoint_path, device=device)
    dataset_name = args.dataset or str(config.get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset name must be provided via --dataset or checkpoint config.")

    view_config = _resolve_view_config(args, config)
    preprocess_config = view_config.to_preprocess_config()
    dataset = UEATimeSeriesDataset(
        dataset_name,
        split=args.split,
        pad_to_max=args.pad_to_max,
        view_config=view_config,
        normalize=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"Dataset split is empty: {dataset_name}/{args.split}")

    sample0 = dataset[0]
    x0 = ensure_ct_sample(sample0["x_time"], "x_time")
    v0 = build_triview_from_time(x0, preprocess_config)
    input_dim_time = x0.shape[0]
    input_dim_freq = v0["x_freq"].shape[0] if v0["x_freq"].dim() > 1 else 1
    input_dim_tf = v0["x_tf"].shape[0] if v0["x_tf"].dim() > 1 else 1
    freq_bins = v0["x_freq"].shape[-1] if v0["x_freq"].dim() > 1 else v0["x_freq"].shape[0]
    band_slices = _build_band_slices(freq_bins, args.bands)

    model = _build_model(
        config=config,
        input_dim_time=input_dim_time,
        input_dim_freq=input_dim_freq,
        input_dim_tf=input_dim_tf,
        device=device,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint into MultiViewModel. "
            "Use a pretrain multiview checkpoint (e.g., *_pretrain_last.pt)."
        ) from exc
    model.eval()

    max_samples = min(len(dataset), max(1, args.max_samples))
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    indices = perm[:max_samples]

    matrix_sum = torch.zeros((args.bands, args.bands), dtype=torch.float32, device=device)
    with torch.no_grad():
        for sample_rank, sample_idx in enumerate(indices):
            item = dataset[sample_idx]
            x_time = ensure_ct_sample(item["x_time"], "x_time")
            x_target = _apply_transform(
                x_time,
                args,
                view_config=view_config,
                sample_seed=args.seed + sample_rank,
            )
            views_anchor = build_triview_from_time(x_time, preprocess_config)
            views_target = build_triview_from_time(x_target, preprocess_config)
            x_freq_anchor = views_anchor["x_freq"].to(device)
            x_tf_target = views_target["x_tf"].to(device)
            matrix = _encode_band_matrix(
                model=model,
                x_freq_anchor=x_freq_anchor,
                x_tf_target=x_tf_target,
                band_slices=band_slices,
                channels=x_time.shape[0],
                freq_bins=freq_bins,
            )
            matrix_sum += matrix
    matrix_mean = matrix_sum / float(max_samples)

    expected = _expected_mapping(band_slices, transform=args.transform, shift_bins=args.shift_bins)
    raw_metrics = _matching_metrics(matrix_mean, expected)
    raw_metrics["variant"] = "raw"

    aligned_metrics = None
    matrix_aligned = None
    if args.transform == "shift" and args.align_shift:
        matrix_aligned = _align_shift_matrix(matrix_mean, args.shift_bins, band_slices)
        aligned_metrics = _matching_metrics(matrix_aligned, list(range(args.bands)))
        aligned_metrics["variant"] = "aligned"

    ncols = 2 if matrix_aligned is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]
    raw_np = matrix_mean.detach().cpu().numpy()
    im = axes[0].imshow(raw_np, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title(
        f"Band Similarity ({args.transform})\n"
        f"Top1={raw_metrics['top1']:.3f}, Tol1={raw_metrics['top1_tol1']:.3f}"
    )
    axes[0].set_xlabel("Target TF band")
    axes[0].set_ylabel("Anchor Freq band")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    if matrix_aligned is not None:
        aligned_np = matrix_aligned.detach().cpu().numpy()
        im2 = axes[1].imshow(aligned_np, cmap="viridis", origin="lower", aspect="auto")
        axes[1].set_title(
            "Aligned Shift Matrix\n"
            f"Top1={aligned_metrics['top1']:.3f}, Tol1={aligned_metrics['top1_tol1']:.3f}"
        )
        axes[1].set_xlabel("Aligned TF band")
        axes[1].set_ylabel("Anchor Freq band")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = preprocess_config.hash()
    transform_tag = (
        f"shift-b-{args.shift_bins}"
        if args.transform == "shift"
        else f"scale-rho-{args.scale_ratio}"
        if args.transform == "scale"
        else f"color-g-{args.color_max_gain_db}"
        if args.transform == "color"
        else "none"
    )
    stem = build_tag(
        "band_similarity",
        dataset_name,
        args.split,
        f"seed{args.seed}",
        f"stft{stft_hash}",
        f"K{args.bands}",
        transform_tag,
    )
    fig_path = figs_dir / f"{stem}.png"
    matrix_csv = csv_dir / f"{stem}_matrix.csv"
    metrics_csv = csv_dir / f"{stem}_metrics.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    _write_matrix_csv(matrix_csv, matrix_mean)

    metric_rows = [
        {
            "variant": raw_metrics["variant"],
            "top1": raw_metrics["top1"],
            "top1_tol1": raw_metrics["top1_tol1"],
            "diag_mean": raw_metrics["diag_mean"],
            "offdiag_mean": raw_metrics["offdiag_mean"],
            "transform": args.transform,
            "shift_bins": args.shift_bins,
            "scale_ratio": args.scale_ratio,
            "color_max_gain_db": args.color_max_gain_db,
            "bands": args.bands,
            "n_samples": max_samples,
            "checkpoint": str(checkpoint_path),
            "dataset": dataset_name,
            "split": args.split,
            "stft_hash": stft_hash,
        }
    ]
    if aligned_metrics is not None:
        metric_rows.append(
            {
                "variant": aligned_metrics["variant"],
                "top1": aligned_metrics["top1"],
                "top1_tol1": aligned_metrics["top1_tol1"],
                "diag_mean": aligned_metrics["diag_mean"],
                "offdiag_mean": aligned_metrics["offdiag_mean"],
                "transform": args.transform,
                "shift_bins": args.shift_bins,
                "scale_ratio": args.scale_ratio,
                "color_max_gain_db": args.color_max_gain_db,
                "bands": args.bands,
                "n_samples": max_samples,
                "checkpoint": str(checkpoint_path),
                "dataset": dataset_name,
                "split": args.split,
                "stft_hash": stft_hash,
            }
        )
    write_csv(metrics_csv, metric_rows)

    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/vis_band_similarity.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "matrix_csv": str(matrix_csv),
            "metrics_csv": str(metrics_csv),
            "stft_hash": stft_hash,
            "note": "Band-level freq-vs-tf similarity for mechanism verification.",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_matrix_csv={matrix_csv}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
