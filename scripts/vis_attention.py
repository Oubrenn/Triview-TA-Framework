import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

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
from preprocessing import ensure_ct_sample  # noqa: E402
from train_uea import UEAClassifier  # noqa: E402
from transforms import (  # noqa: E402
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
)


def _parse_int_list(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _apply_per_channel(x: torch.Tensor, fn) -> torch.Tensor:
    x = ensure_ct_sample(x, "x")
    if x.shape[0] == 1:
        return fn(x.squeeze(0)).unsqueeze(0)
    return torch.stack([fn(x[idx]) for idx in range(x.shape[0])], dim=0)


def _select_attn_tensor(attn_obj, attn_key: str = "") -> torch.Tensor:
    if isinstance(attn_obj, torch.Tensor):
        return attn_obj
    if isinstance(attn_obj, dict):
        if attn_key:
            if attn_key not in attn_obj or not isinstance(attn_obj[attn_key], torch.Tensor):
                available = ", ".join(sorted(attn_obj.keys()))
                raise ValueError(f"--attn-key '{attn_key}' is unavailable. available=[{available}]")
            return attn_obj[attn_key]
        for key in sorted(attn_obj.keys()):
            value = attn_obj[key]
            if isinstance(value, torch.Tensor):
                return value
        available = ", ".join(sorted(attn_obj.keys()))
        raise ValueError(
            "Attention features are all None. "
            "Use a checkpoint with attention enabled (use_temporal_attn/use_shared_qk_attn). "
            f"available_keys=[{available}]"
        )
    raise ValueError("Unexpected attention object type. Expected Tensor or dict of Tensor.")


def _matrix_from_attn_feat(attn_feat: torch.Tensor) -> torch.Tensor:
    if attn_feat.dim() == 2:
        attn_feat = attn_feat.unsqueeze(-1)
    if attn_feat.dim() != 3:
        raise ValueError(f"Expected attention feature shape (B,C,T) or (B,C), got {tuple(attn_feat.shape)}")
    tokens = attn_feat[0].transpose(0, 1).float()  # (T, C)
    tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
    matrix = torch.matmul(tokens, tokens.transpose(0, 1))  # (T, T)
    return matrix


def _write_matrix_csv(path: Path, matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat = matrix.detach().cpu()
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in mat:
            handle.write(",".join(f"{float(v):.8g}" for v in row.tolist()) + "\n")


def _build_model_from_checkpoint(checkpoint_path: Path, device: str) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict: {checkpoint_path}")
    config = checkpoint.get("config")
    model_state = checkpoint.get("model_state")
    if not isinstance(config, dict) or not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing config/model_state: {checkpoint_path}")
    return {"checkpoint": checkpoint, "config": config, "model_state": model_state}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
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
    parser.add_argument("--transform", type=str, default="shift", choices=["shift", "scale", "color"])
    parser.add_argument("--shift-bins", type=float, default=0.2)
    parser.add_argument("--shift-fill", type=str, default="border", choices=["zero", "circular", "border", "reflect"])
    parser.add_argument("--scale-ratio", type=float, default=1.1)
    parser.add_argument("--color-max-gain-db", type=float, default=6.0)
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--attn-key", type=str, default="")
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    loaded = _build_model_from_checkpoint(Path(args.checkpoint), device=device)
    config = loaded["config"]
    state = loaded["model_state"]

    dataset_name = args.dataset or str(config.get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset name must be provided via --dataset or checkpoint config.")

    view_config = ViewConfig(
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

    dataset = UEATimeSeriesDataset(
        dataset_name,
        split=args.split,
        pad_to_max=args.pad_to_max,
        view_config=view_config,
        normalize=True,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range: {args.index} not in [0, {len(dataset) - 1}]")

    sample = dataset[args.index]
    x_time = ensure_ct_sample(sample["x_time"], "x_time")

    color_active_bands = _parse_int_list(args.color_active_bands)
    transform_detail = ""
    if args.transform == "shift":
        x_transformed = _apply_per_channel(
            x_time,
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
        transform_detail = f"b={args.shift_bins}"
    elif args.transform == "scale":
        x_transformed = _apply_per_channel(x_time, lambda c: frequency_scale_time(c, args.scale_ratio))
        transform_detail = f"rho={args.scale_ratio}"
    else:
        num_bins = x_time.shape[-1] // 2 + 1
        gains = make_coloring_gains(
            num_bins=num_bins,
            bands=args.color_bands,
            max_gain_db=args.color_max_gain_db,
            active_bands=color_active_bands if color_active_bands else None,
            generator=torch.Generator().manual_seed(args.seed),
        )
        x_transformed = _apply_per_channel(x_time, lambda c: spectral_coloring(c, gains))
        transform_detail = f"g={args.color_max_gain_db}dB"

    input_dim = dataset.data[0].shape[0]
    num_classes = len(dataset.class_labels)
    model = UEAClassifier(
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
    model.load_state_dict(state, strict=True)
    model.eval()

    x_clean_bct = x_time.unsqueeze(0).to(device)
    x_trans_bct = x_transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        _, attn_clean_raw = model.encoder.forward_with_attn(x_clean_bct)
        _, attn_trans_raw = model.encoder.forward_with_attn(x_trans_bct)

    attn_clean = _select_attn_tensor(attn_clean_raw, attn_key=args.attn_key)
    attn_trans = _select_attn_tensor(attn_trans_raw, attn_key=args.attn_key)

    A = _matrix_from_attn_feat(attn_clean)
    A_tilde = _matrix_from_attn_feat(attn_trans)

    mse = float(torch.mean((A - A_tilde) ** 2).item())
    p = torch.softmax(A, dim=-1)
    q = torch.softmax(A_tilde, dim=-1)
    kl = float(F.kl_div((q + 1e-8).log(), p, reduction="batchmean").item())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vmin = float(min(A.min().item(), A_tilde.min().item()))
    vmax = float(max(A.max().item(), A_tilde.max().item()))
    axes[0].imshow(A.detach().cpu().numpy(), cmap="Blues", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    axes[0].set_title("A (base attention)")
    axes[0].set_xlabel("token")
    axes[0].set_ylabel("token")
    axes[1].imshow(A_tilde.detach().cpu().numpy(), cmap="Blues", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    axes[1].set_title("A_tilde (transformed attention)")
    axes[1].set_xlabel("token")
    axes[1].set_ylabel("token")
    fig.suptitle(f"transform={args.transform} ({transform_detail}) | MSE={mse:.6f}, KL={kl:.6f}", fontsize=11)
    plt.tight_layout()

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = view_config.to_preprocess_config().hash()
    stem = build_tag(
        "attn_pair",
        dataset_name,
        args.split,
        f"idx{args.index}",
        f"seed{args.seed}",
        f"stft{stft_hash}",
        f"{args.transform}-{transform_detail}",
    )
    fig_path = figs_dir / f"{stem}.png"
    a_csv = csv_dir / f"{stem}_A.csv"
    at_csv = csv_dir / f"{stem}_A_tilde.csv"
    metrics_csv = csv_dir / f"{stem}_metrics.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    _write_matrix_csv(a_csv, A)
    _write_matrix_csv(at_csv, A_tilde)
    write_csv(
        metrics_csv,
        [
            {
                "dataset": dataset_name,
                "split": args.split,
                "index": args.index,
                "transform": args.transform,
                "transform_detail": transform_detail,
                "mse": mse,
                "kl": kl,
                "attn_key": args.attn_key,
                "stft_hash": stft_hash,
                "checkpoint": str(Path(args.checkpoint)),
                "device": device,
            }
        ],
    )
    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/vis_attention.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "A_csv": str(a_csv),
            "A_tilde_csv": str(at_csv),
            "metrics_csv": str(metrics_csv),
            "mse": mse,
            "kl": kl,
            "stft_hash": stft_hash,
            "attention_note": "A/A_tilde are cosine-similarity token matrices from encoder attention features.",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_A_csv={a_csv}")
    print(f"saved_A_tilde_csv={at_csv}")
    print(f"saved_metrics_csv={metrics_csv}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
