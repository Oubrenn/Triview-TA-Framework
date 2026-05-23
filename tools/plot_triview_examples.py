import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from preprocessing import normalize_time_series  # noqa: E402


STFT_CONFIGS: Dict[str, Dict[str, int]] = {
    "HHAR": {
        "n_fft": 64,
        "win_length": 64,
        "hop_length": 16,
    },
    "Heartbeat": {
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 64,
    },
    "JapaneseVowels": {
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 64,
    },
}

PREFERRED_LABELS = {
    "HHAR": "walk",
    "Heartbeat": "abnormal",
}

HEATMAP_CMAP_NAME = "iceblue"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    HEATMAP_CMAP_NAME,
    [
        "#E6F4FF",
        "#B7DDF2",
        "#6BAED6",
        "#2171B5",
        "#08306B",
    ],
)
LINE_COLOR = "#174A7C"
HEATMAP_CLIP_PERCENTILE = 97.0
COLORBAR_FRACTION = 0.030
COLORBAR_PAD = 0.018
SAVE_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot qualitative tri-view construction examples: time view, STFT magnitude, "
            "temporally pooled frequency view, and log-compressed time-frequency view."
        )
    )
    parser.add_argument("--hhar", type=Path, default=None, help="Optional HHAR .npz sample path.")
    parser.add_argument("--heartbeat", type=Path, default=None, help="Optional Heartbeat .npz sample path.")
    parser.add_argument("--out", type=Path, default=Path("figures/fig_triview_examples.pdf"))
    parser.add_argument("--sample-dir", type=Path, default=Path("vis_samples"))
    parser.add_argument("--root-dir", type=Path, default=None, help="Dataset root; defaults to dataset/all_datasets.")
    parser.add_argument("--hhar-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--heartbeat-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--hhar-index", type=int, default=None, help="HHAR sample index. Default: auto.")
    parser.add_argument("--heartbeat-index", type=int, default=None, help="Heartbeat sample index. Default: auto.")
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Use unnormalized dataset values when auto-exporting samples.",
    )
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for STFT computation. Default auto uses CUDA when available.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def ensure_ct(x: np.ndarray) -> np.ndarray:
    """Convert [T], [C, T], or [T, C] into [C, T]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D sample, got shape={x.shape}.")

    first, second = x.shape
    if first > second and second <= 128:
        return x.T
    return x


def load_npz_sample(path: Path) -> Tuple[np.ndarray, int, Dict[str, object]]:
    payload = np.load(path, allow_pickle=True)
    if "x" not in payload:
        raise KeyError(f"Sample file missing key 'x': {path}")
    x_ct = ensure_ct(payload["x"])
    y = int(payload["y"]) if "y" in payload else -1
    meta: Dict[str, object] = {"path": str(path)}
    if "meta" in payload:
        raw_meta = payload["meta"]
        if raw_meta.shape == ():
            meta_payload = raw_meta.item()
            if isinstance(meta_payload, dict):
                meta.update(meta_payload)
            else:
                meta["meta"] = meta_payload
        else:
            meta["meta"] = raw_meta.tolist()
    return x_ct, y, meta


def sample_score(x_ct: torch.Tensor, length: int) -> float:
    valid = x_ct[..., :length]
    if valid.numel() == 0:
        return float("-inf")
    return float(torch.nan_to_num(valid).var(dim=-1, unbiased=False).max().item())


def select_dataset_index(
    dataset: UEATimeSeriesDataset,
    dataset_name: str,
    requested_index: Optional[int],
) -> int:
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(dataset):
            raise IndexError(f"{dataset_name} index {requested_index} is outside [0, {len(dataset) - 1}].")
        return requested_index

    preferred_label = PREFERRED_LABELS.get(dataset_name)
    preferred_id = dataset.label_to_index.get(preferred_label) if preferred_label else None

    candidates = range(len(dataset))
    if preferred_id is not None:
        preferred = torch.nonzero(dataset.labels == int(preferred_id), as_tuple=False).flatten()
        if preferred.numel() > 0:
            candidates = [int(i) for i in preferred.tolist()]

    best_idx = 0
    best_score = float("-inf")
    for idx in candidates:
        length = int(dataset.lengths[idx].item())
        score = sample_score(dataset.data[idx], length)
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def export_dataset_sample(
    dataset_name: str,
    split: str,
    index: Optional[int],
    sample_dir: Path,
    root_dir: Optional[Path],
    normalize: bool,
) -> Tuple[np.ndarray, int, Path, Dict[str, object]]:
    view_config = ViewConfig(normalize_mode="per_sample_channel" if normalize else "none")
    raw_dataset = UEATimeSeriesDataset(
        dataset_name,
        split=split,
        root_dir=root_dir,
        normalize=False,
        pad_to_max=True,
        view_config=view_config,
    )
    selected_idx = select_dataset_index(raw_dataset, dataset_name, index)

    item = raw_dataset[selected_idx]
    x_tensor = item["x_time"].detach().cpu()
    length = int(item["length"].item())
    if normalize:
        x_tensor = normalize_time_series(x_tensor, length=length, mode=view_config.normalize_mode)
    y = int(item["y"].item())

    label = raw_dataset.class_labels[y] if 0 <= y < len(raw_dataset.class_labels) else str(y)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{dataset_name.lower()}_sample.npz"
    meta = {
        "dataset": dataset_name,
        "split": split,
        "index": selected_idx,
        "label": label,
        "length": length,
        "normalized": bool(normalize),
    }
    if "meta" in item and isinstance(item["meta"], dict):
        meta["item_meta"] = {
            key: int(value.item()) if torch.is_tensor(value) and value.numel() == 1 else str(value)
            for key, value in item["meta"].items()
        }

    np.savez(sample_path, x=x_tensor.numpy(), y=y, meta=np.array(meta, dtype=object))
    return x_tensor.numpy(), y, sample_path, meta


def choose_representative_channel(x_ct: np.ndarray) -> int:
    variances = np.nanvar(x_ct, axis=1)
    return int(np.nanargmax(variances))


def compute_stft_views(
    x_1d: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(x_1d, dtype=torch.float32, device=device)
    window = torch.hann_window(win_length, device=device)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = torch.abs(spec)
    pooled_freq = mag.mean(dim=1)
    log_tf = torch.log1p(mag)
    return (
        mag.detach().cpu().numpy(),
        pooled_freq.detach().cpu().numpy(),
        log_tf.detach().cpu().numpy(),
    )


def robust_imshow(ax: plt.Axes, values: np.ndarray, title: str) -> None:
    finite = values[np.isfinite(values)]
    vmax = float(np.percentile(finite, HEATMAP_CLIP_PERCENTILE)) if finite.size else None
    if vmax is not None and vmax <= 0.0:
        vmax = None
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=HEATMAP_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Freq. bin", fontsize=10)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    cbar.ax.tick_params(labelsize=8)


def plot_one_dataset(
    axes: np.ndarray,
    dataset_name: str,
    x_ct: np.ndarray,
    y: int,
    meta: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    cfg = STFT_CONFIGS[dataset_name]
    channel = choose_representative_channel(x_ct)
    x_1d = np.nan_to_num(x_ct[channel], nan=0.0, posinf=0.0, neginf=0.0)
    mag, pooled_freq, log_tf = compute_stft_views(x_1d, device=device, **cfg)

    label = str(meta.get("label", y))
    row_prefix = f"{dataset_name} ({label})"

    ax = axes[0]
    ax.plot(x_1d, linewidth=1.1, color=LINE_COLOR)
    ax.set_title(f"{row_prefix}\nRaw time view", fontsize=11)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.text(
        0.98,
        0.94,
        f"Ch. {channel}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )

    robust_imshow(axes[1], mag, r"$|\mathrm{STFT}(x)|$")

    ax = axes[2]
    ax.plot(pooled_freq, linewidth=1.1, color=LINE_COLOR)
    ax.set_title(r"$\mathrm{Pool}_t(|\mathrm{STFT}(x)|)$", fontsize=11)
    ax.set_xlabel("Freq. bin", fontsize=10)
    ax.set_ylabel("Magnitude", fontsize=10)
    ax.grid(True, alpha=0.18, linewidth=0.5)

    robust_imshow(axes[3], log_tf, r"$\log(1+|\mathrm{STFT}(x)|)$")

    return {
        "dataset": dataset_name,
        "label": label,
        "channel": channel,
        "sample_meta": meta,
        "stft": cfg,
        "mag_shape": list(mag.shape),
        "heatmap_clip_percentile": HEATMAP_CLIP_PERCENTILE,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    sample_records = []
    if args.hhar is not None:
        hhar_x, hhar_y, hhar_meta = load_npz_sample(args.hhar)
        hhar_path = args.hhar
    else:
        hhar_x, hhar_y, hhar_path, hhar_meta = export_dataset_sample(
            "HHAR",
            args.hhar_split,
            args.hhar_index,
            args.sample_dir,
            args.root_dir,
            args.normalize,
        )
    hhar_meta.setdefault("sample_path", str(hhar_path))

    if args.heartbeat is not None:
        heartbeat_x, heartbeat_y, heartbeat_meta = load_npz_sample(args.heartbeat)
        heartbeat_path = args.heartbeat
    else:
        heartbeat_x, heartbeat_y, heartbeat_path, heartbeat_meta = export_dataset_sample(
            "Heartbeat",
            args.heartbeat_split,
            args.heartbeat_index,
            args.sample_dir,
            args.root_dir,
            args.normalize,
        )
    heartbeat_meta.setdefault("sample_path", str(heartbeat_path))

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.0), constrained_layout=True)
    sample_records.append(plot_one_dataset(axes[0], "HHAR", hhar_x, hhar_y, hhar_meta, device))
    sample_records.append(plot_one_dataset(axes[1], "Heartbeat", heartbeat_x, heartbeat_y, heartbeat_meta, device))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=SAVE_DPI, bbox_inches="tight")
    png_path = args.out.with_suffix(".png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    meta_path = args.out.with_suffix(".json")
    run_meta = {
        "figure_pdf": str(args.out),
        "figure_png": str(png_path),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dpi": SAVE_DPI,
        "line_color": LINE_COLOR,
        "heatmap_cmap": HEATMAP_CMAP_NAME,
        "heatmap_clip_percentile": HEATMAP_CLIP_PERCENTILE,
        "colorbar_fraction": COLORBAR_FRACTION,
        "colorbar_pad": COLORBAR_PAD,
        "samples": sample_records,
        "definitions": {
            "time_view": "selected raw time-domain channel from x_time",
            "stft_magnitude": "|STFT(x)|",
            "pooled_frequency": "mean over STFT frames",
            "log_time_frequency": "log(1 + |STFT(x)|)",
        },
    }
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_name={torch.cuda.get_device_name(0)}")
    print(f"hhar_sample={hhar_path}")
    print(f"heartbeat_sample={heartbeat_path}")
    print(f"saved_pdf={args.out}")
    print(f"saved_png={png_path}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
