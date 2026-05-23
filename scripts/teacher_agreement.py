import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, stable_hash, write_csv, write_run_meta  # noqa: E402
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


def _parse_band_list(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _load_checkpoint(path: Path, device: str) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dict with config/model_state: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint missing config dict: {path}")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing model_state dict: {path}")
    return checkpoint, config, state


def _build_model(config: Dict[str, object], input_dim: int, num_classes: int, device: str) -> UEAClassifier:
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
    return model


def _agreement_eval(model, loader, device: str, transform_fn, max_param_rows: int, transform_name: str):
    model.eval()
    total = 0
    total_agree = 0
    total_kl = 0.0
    total_clean_acc = 0
    total_aug_acc = 0
    param_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x_time"].to(device)
            y = batch["y"].to(device)
            clean_logits = model(x)
            x_aug, meta = transform_fn(x, batch_idx)
            aug_logits = model(x_aug)

            clean_pred = clean_logits.argmax(dim=1)
            aug_pred = aug_logits.argmax(dim=1)
            agree = (clean_pred == aug_pred)
            total_agree += int(agree.sum().item())
            total += int(y.size(0))
            total_clean_acc += int((clean_pred == y).sum().item())
            total_aug_acc += int((aug_pred == y).sum().item())

            clean_log_prob = F.log_softmax(clean_logits, dim=1)
            aug_prob = F.softmax(aug_logits, dim=1)
            kl = F.kl_div(clean_log_prob, aug_prob, reduction="batchmean")
            total_kl += float(kl.item()) * int(y.size(0))

            if len(param_rows) < max_param_rows:
                param_rows.append(
                    {
                        "transform": transform_name,
                        "batch_idx": batch_idx,
                        "batch_size": int(y.size(0)),
                        "severity_id": meta.get("severity_id", 0),
                        "rho": meta.get("rho", 1.0),
                        "g_db": meta.get("g_db", 0.0),
                        "b_bins": meta.get("b_bins", 0),
                        "seed": meta.get("seed", 0),
                        "shift_fill": meta.get("shift_fill", "na"),
                    }
                )

    return {
        "agreement": total_agree / max(1, total),
        "kl": total_kl / max(1, total),
        "clean_acc": total_clean_acc / max(1, total),
        "aug_acc": total_aug_acc / max(1, total),
        "count": total,
    }, param_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--shift-fill",
        type=str,
        default="border",
        choices=["zero", "circular", "border", "reflect"],
        help="Shift implementation: zero-padding (aggressive) or circular (edge-preserving).",
    )
    parser.add_argument(
        "--safe-agreement-threshold",
        type=float,
        default=0.95,
        help="Agreement threshold used to define safe shift range.",
    )
    parser.add_argument("--scale-ratios", type=str, default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--color-max-db", type=str, default="0,3,6,9")
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--color-trials", type=int, default=1)
    parser.add_argument("--max-param-rows", type=int, default=1000)
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    _, config, state = _load_checkpoint(checkpoint_path, device=device)
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
    preprocess_config = view_config.to_preprocess_config()
    dataset = UEATimeSeriesDataset(
        dataset_name,
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
    input_dim = dataset.data[0].shape[0]
    num_classes = len(dataset.class_labels)
    model = _build_model(config, input_dim, num_classes, device=device)
    model.load_state_dict(state, strict=True)

    shift_bins = _parse_float_list(args.shift_bins)
    scale_ratios = _parse_float_list(args.scale_ratios)
    color_levels = _parse_float_list(args.color_max_db)
    color_active_bands = _parse_band_list(args.color_active_bands)
    color_trials = max(1, int(args.color_trials))
    num_bins = dataset.data[0].shape[-1] // 2 + 1

    summary_rows = []
    param_rows = []

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

        metrics, rows = _agreement_eval(model, loader, device, _shift_transform, args.max_param_rows, "shift")
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
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": args.shift_fill,
            }
        )

    for severity_id, ratio in enumerate(scale_ratios):
        def _scale_transform(x: torch.Tensor, _batch_idx: int, r: float = ratio, sid: int = severity_id):
            out = apply_per_sample_channel(x, lambda s: frequency_scale_time(s, r))
            return out, {"severity_id": sid, "rho": r, "g_db": 0.0, "b_bins": 0, "seed": args.seed, "shift_fill": "na"}

        metrics, rows = _agreement_eval(model, loader, device, _scale_transform, args.max_param_rows, "scale")
        param_rows.extend(rows[: max(0, args.max_param_rows - len(param_rows))])
        summary_rows.append(
            {
                "transform": "scale",
                "severity_id": severity_id,
                "rho": ratio,
                "g_db": 0.0,
                "b_bins": 0,
                "trial": 0,
                **metrics,
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": "na",
            }
        )

    for severity_id, max_db in enumerate(color_levels):
        trial_agreements = []
        trial_kls = []
        trial_clean_acc = []
        trial_aug_acc = []
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
                    "b_bins": 0,
                    "seed": current_seed,
                    "shift_fill": "na",
                }

            metrics, rows = _agreement_eval(model, loader, device, _color_transform, args.max_param_rows, "color")
            param_rows.extend(rows[: max(0, args.max_param_rows - len(param_rows))])
            summary_rows.append(
                {
                    "transform": "color",
                    "severity_id": severity_id,
                    "rho": 1.0,
                    "g_db": max_db,
                    "b_bins": 0,
                    "trial": trial,
                    **metrics,
                    "split": args.split,
                    "severity_source": args.severity_source,
                    "shift_fill": "na",
                }
            )
            trial_agreements.append(metrics["agreement"])
            trial_kls.append(metrics["kl"])
            trial_clean_acc.append(metrics["clean_acc"])
            trial_aug_acc.append(metrics["aug_acc"])
        if color_trials > 1:
            summary_rows.append(
                {
                    "transform": "color",
                    "severity_id": severity_id,
                    "rho": 1.0,
                    "g_db": max_db,
                    "b_bins": 0,
                    "trial": "mean",
                    "agreement": sum(trial_agreements) / len(trial_agreements),
                    "kl": sum(trial_kls) / len(trial_kls),
                    "clean_acc": sum(trial_clean_acc) / len(trial_clean_acc),
                    "aug_acc": sum(trial_aug_acc) / len(trial_aug_acc),
                    "count": len(dataset),
                    "split": args.split,
                    "severity_source": args.severity_source,
                    "shift_fill": "na",
                }
            )

    def _pick_color_curve_points(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        mean_rows = [row for row in rows if str(row.get("trial")) == "mean"]
        chosen = mean_rows if mean_rows else [row for row in rows if str(row.get("trial")) in {"0", "0.0"}]
        if not chosen:
            chosen = rows
        dedup = {}
        for row in chosen:
            dedup[float(row["g_db"])] = row
        return [dedup[key] for key in sorted(dedup.keys())]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    shift_points = sorted([row for row in summary_rows if row["transform"] == "shift"], key=lambda row: float(row["b_bins"]))
    scale_points = sorted([row for row in summary_rows if row["transform"] == "scale"], key=lambda row: float(row["rho"]))
    color_points = _pick_color_curve_points([row for row in summary_rows if row["transform"] == "color"])

    shift_x = [row["b_bins"] for row in shift_points]
    shift_y = [row["agreement"] for row in shift_points]
    if len(shift_points) <= 1:
        axes[0].scatter(shift_x, shift_y, marker="o")
    else:
        axes[0].plot(shift_x, shift_y, marker="o")
    axes[0].axhline(
        y=args.safe_agreement_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"safe>= {args.safe_agreement_threshold:.2f}",
    )
    axes[1].plot([row["rho"] for row in scale_points], [row["agreement"] for row in scale_points], marker="o")
    axes[2].plot([row["g_db"] for row in color_points], [row["agreement"] for row in color_points], marker="o")
    axes[0].set_title("Shift Agreement")
    axes[0].set_xlabel("b (shift bins)")
    axes[0].set_ylabel("Agreement")
    axes[1].set_title("Scale Agreement")
    axes[1].set_xlabel("rho (scale)")
    axes[1].set_ylabel("Agreement")
    axes[2].set_title("Color Agreement")
    axes[2].set_xlabel("g (max dB)")
    axes[2].set_ylabel("Agreement")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    safe_nonzero_exists = any(
        (float(row["agreement"]) >= float(args.safe_agreement_threshold)) and (abs(float(row["b_bins"])) > 1e-8)
        for row in shift_points
    )
    if not safe_nonzero_exists:
        axes[0].text(
            0.02,
            0.04,
            "No safe shift found under agreement>=0.95 (dataset-specific).",
            transform=axes[0].transAxes,
            fontsize=8,
            color="tab:red",
            ha="left",
            va="bottom",
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
            "color_bands": args.color_bands,
            "color_active_bands": color_active_bands,
            "color_trials": color_trials,
        }
    )
    stem = build_tag(
        "teacher_agreement",
        dataset_name,
        args.split,
        f"seed{args.seed}",
        f"stft{stft_hash}",
        f"sh{len(shift_bins)}-sc{len(scale_ratios)}-co{len(color_levels)}",
        f"sev{sev_hash}",
    )
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    params_csv = csv_dir / f"{stem}_params.csv"
    safe_csv = csv_dir / f"{stem}_safe_shift.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, summary_rows)
    write_csv(params_csv, param_rows[: args.max_param_rows])
    safe_shift_rows = []
    safe_bins = []
    for row in shift_points:
        b_val = float(row["b_bins"])
        agree = float(row["agreement"])
        safe = agree >= float(args.safe_agreement_threshold)
        if safe:
            safe_bins.append(b_val)
        safe_shift_rows.append(
            {
                "b_bins": b_val,
                "agreement": agree,
                "safe": int(safe),
                "threshold": float(args.safe_agreement_threshold),
                "shift_fill": args.shift_fill,
                "shift_mode": args.shift_mode,
            }
        )
    write_csv(safe_csv, safe_shift_rows)

    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/teacher_agreement.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "params_csv": str(params_csv),
            "safe_shift_csv": str(safe_csv),
            "stft_hash": stft_hash,
            "severity_hash": sev_hash,
            "shift_fill": args.shift_fill,
            "safe_agreement_threshold": args.safe_agreement_threshold,
            "safe_shift_bins": safe_bins,
            "leakage_guard": "No agreement threshold tuning is performed on split=test.",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_params_csv={params_csv}")
    print(f"saved_safe_shift_csv={safe_csv}")
    print(f"safe_shift_bins={safe_bins}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
