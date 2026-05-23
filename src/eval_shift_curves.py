import argparse
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import UEATimeSeriesDataset
from models import build_encoder
from transforms import (
    band_shift_time,
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
)


class UEAClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        res_blocks: int,
        backbone: str,
        use_temporal_attn: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone=backbone,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_temporal_attn=use_temporal_attn,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)


def _parse_list(raw: str, cast) -> List:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_band_list(raw: str, num_bands: int) -> List[int]:
    values = _parse_list(raw, int)
    if not values:
        return []
    unique = sorted(set(values))
    for idx in unique:
        if idx < 0 or idx >= num_bands:
            raise ValueError(f"Band index {idx} out of range for num_bands={num_bands}.")
    return unique


def _apply_per_sample(x: torch.Tensor, fn) -> torch.Tensor:
    if x.dim() == 2:
        return torch.stack([fn(x[i]) for i in range(x.size(0))], dim=0)
    if x.dim() == 3:
        return torch.stack(
            [torch.stack([fn(x[i, j]) for j in range(x.size(1))], dim=0) for i in range(x.size(0))],
            dim=0,
        )
    raise ValueError("Expected batch tensor with shape (B, L) or (B, C, L).")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_count = 0
    confusion = None
    with torch.no_grad():
        for batch in loader:
            x = batch["x_time"].to(device)
            y = batch["y"].to(device)
            if transform_fn is not None:
                x = transform_fn(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)
            if confusion is None:
                confusion = torch.zeros((logits.size(1), logits.size(1)), dtype=torch.long)
            y_cpu = y.view(-1).to(torch.long).cpu()
            preds_cpu = preds.view(-1).to(torch.long).cpu()
            idx = y_cpu * logits.size(1) + preds_cpu
            bins = torch.bincount(idx, minlength=logits.size(1) * logits.size(1))
            confusion += bins.view(logits.size(1), logits.size(1))
    acc = total_correct / max(1, total_count)
    if confusion is None:
        return acc, 0.0
    conf = confusion.to(dtype=torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return acc, f1.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument(
        "--backbone",
        type=str,
        default="inception",
        choices=["inception", "resnet", "inception_resattn", "tfc_resnet", "timesnet", "tslanet", "all"],
    )
    parser.add_argument("--use-temporal-attn", action="store_true", default=False)
    parser.add_argument("--shift-seen", type=str, default="3,-3")
    parser.add_argument("--shift-unseen", type=str, default="6,-6")
    parser.add_argument(
        "--shift-mode",
        type=str,
        default="rfft",
        choices=["rfft", "stft"],
        help="Use rFFT bin shift (rfft) or STFT bin shift with iSTFT (stft).",
    )
    parser.add_argument(
        "--shift-fill",
        type=str,
        default="border",
        choices=["zero", "circular", "border", "reflect"],
        help="Shift implementation: zero-padding (aggressive) or circular (edge-preserving).",
    )
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--scale-ratios", type=str, default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument(
        "--scale-seen",
        type=str,
        default="",
        help="Optional seen scale ratios for explicit OOD protocol.",
    )
    parser.add_argument(
        "--scale-unseen",
        type=str,
        default="",
        help="Optional unseen scale ratios for explicit OOD protocol.",
    )
    parser.add_argument("--color-max-db", type=str, default="0,3,6,9")
    parser.add_argument(
        "--color-max-db-seen",
        type=str,
        default="",
        help="Optional seen color strengths (dB) for explicit OOD protocol.",
    )
    parser.add_argument(
        "--color-max-db-unseen",
        type=str,
        default="",
        help="Optional unseen color strengths (dB) for explicit OOD protocol.",
    )
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument(
        "--color-seen-bands",
        type=str,
        default="",
        help="Optional seen color band indices (e.g. '0,1,2,3').",
    )
    parser.add_argument(
        "--color-unseen-bands",
        type=str,
        default="",
        help="Optional unseen color band indices used for band holdout OOD.",
    )
    parser.add_argument(
        "--color-band-holdout-max-db",
        type=float,
        default=6.0,
        help="Color strength used for band holdout OOD evaluation.",
    )
    parser.add_argument(
        "--color-band-holdout-trials",
        type=int,
        default=1,
        help="Number of random color draws per band subset; report mean metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    test_ds = UEATimeSeriesDataset(args.dataset, split="test", pad_to_max=args.pad_to_max)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = test_ds.data[0].shape[0]
    num_classes = len(test_ds.class_labels)
    model = UEAClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_classes=num_classes,
        num_heads=args.num_heads,
        res_blocks=args.res_blocks,
        backbone=args.backbone,
        use_temporal_attn=args.use_temporal_attn,
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)

    baseline_acc, baseline_mf1 = _evaluate(model, test_loader, args.device)
    print(f"baseline_acc={baseline_acc:.4f} baseline_mf1={baseline_mf1:.4f}")

    shift_seen = _parse_list(args.shift_seen, int)
    for bins in shift_seen:
        if args.shift_mode == "stft":
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, b=bins: _apply_per_sample(
                    x,
                    lambda s: band_shift_time_stft(
                        s,
                        b,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        shift_mode=args.shift_fill,
                    ),
                ),
            )
        else:
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, b=bins: band_shift_time(x, b, shift_mode=args.shift_fill),
            )
        print(f"shift_seen_bins={bins} acc={acc:.4f} mf1={mf1:.4f}")

    shift_unseen = _parse_list(args.shift_unseen, int)
    for bins in shift_unseen:
        if args.shift_mode == "stft":
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, b=bins: _apply_per_sample(
                    x,
                    lambda s: band_shift_time_stft(
                        s,
                        b,
                        n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        shift_mode=args.shift_fill,
                    ),
                ),
            )
        else:
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, b=bins: band_shift_time(x, b, shift_mode=args.shift_fill),
            )
        print(f"shift_unseen_bins={bins} acc={acc:.4f} mf1={mf1:.4f}")

    scale_seen = _parse_list(args.scale_seen, float)
    scale_unseen = _parse_list(args.scale_unseen, float)
    if scale_seen or scale_unseen:
        for ratio in scale_seen:
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, r=ratio: _apply_per_sample(x, lambda s: frequency_scale_time(s, r)),
            )
            print(f"scale_seen_ratio={ratio} acc={acc:.4f} mf1={mf1:.4f}")
        for ratio in scale_unseen:
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, r=ratio: _apply_per_sample(x, lambda s: frequency_scale_time(s, r)),
            )
            print(f"scale_unseen_ratio={ratio} acc={acc:.4f} mf1={mf1:.4f}")
    else:
        scale_ratios = _parse_list(args.scale_ratios, float)
        for ratio in scale_ratios:
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, r=ratio: _apply_per_sample(x, lambda s: frequency_scale_time(s, r)),
            )
            print(f"scale_ratio={ratio} acc={acc:.4f} mf1={mf1:.4f}")

    color_seen_levels = _parse_list(args.color_max_db_seen, float)
    color_unseen_levels = _parse_list(args.color_max_db_unseen, float)
    series_length = int(test_ds.data[0].shape[-1])
    num_bins = series_length // 2 + 1
    if color_seen_levels or color_unseen_levels:
        for max_db in color_seen_levels:
            torch.manual_seed(args.seed)
            gains = make_coloring_gains(num_bins=num_bins, bands=args.color_bands, max_gain_db=max_db)
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, g=gains: _apply_per_sample(x, lambda s: spectral_coloring(s, g)),
            )
            print(f"color_seen_max_db={max_db} acc={acc:.4f} mf1={mf1:.4f}")
        for max_db in color_unseen_levels:
            torch.manual_seed(args.seed)
            gains = make_coloring_gains(num_bins=num_bins, bands=args.color_bands, max_gain_db=max_db)
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, g=gains: _apply_per_sample(x, lambda s: spectral_coloring(s, g)),
            )
            print(f"color_unseen_max_db={max_db} acc={acc:.4f} mf1={mf1:.4f}")
    else:
        color_levels = _parse_list(args.color_max_db, float)
        for max_db in color_levels:
            torch.manual_seed(args.seed)
            gains = make_coloring_gains(num_bins=num_bins, bands=args.color_bands, max_gain_db=max_db)
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, g=gains: _apply_per_sample(x, lambda s: spectral_coloring(s, g)),
            )
            print(f"color_max_db={max_db} acc={acc:.4f} mf1={mf1:.4f}")

    color_seen_bands = _parse_band_list(args.color_seen_bands, args.color_bands)
    color_unseen_bands = _parse_band_list(args.color_unseen_bands, args.color_bands)
    trials = max(1, args.color_band_holdout_trials)
    if color_seen_bands:
        seen_acc = 0.0
        seen_mf1 = 0.0
        for trial in range(trials):
            torch.manual_seed(args.seed + trial)
            gains = make_coloring_gains(
                num_bins=num_bins,
                bands=args.color_bands,
                max_gain_db=args.color_band_holdout_max_db,
                active_bands=color_seen_bands,
            )
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, g=gains: _apply_per_sample(x, lambda s: spectral_coloring(s, g)),
            )
            seen_acc += acc
            seen_mf1 += mf1
        print(
            "color_band_seen="
            f"{','.join(str(i) for i in color_seen_bands)} "
            f"max_db={args.color_band_holdout_max_db:.4f} "
            f"acc={seen_acc / trials:.4f} mf1={seen_mf1 / trials:.4f}"
        )
    if color_unseen_bands:
        unseen_acc = 0.0
        unseen_mf1 = 0.0
        for trial in range(trials):
            torch.manual_seed(args.seed + 1000 + trial)
            gains = make_coloring_gains(
                num_bins=num_bins,
                bands=args.color_bands,
                max_gain_db=args.color_band_holdout_max_db,
                active_bands=color_unseen_bands,
            )
            acc, mf1 = _evaluate(
                model,
                test_loader,
                args.device,
                transform_fn=lambda x, g=gains: _apply_per_sample(x, lambda s: spectral_coloring(s, g)),
            )
            unseen_acc += acc
            unseen_mf1 += mf1
        print(
            "color_band_unseen="
            f"{','.join(str(i) for i in color_unseen_bands)} "
            f"max_db={args.color_band_holdout_max_db:.4f} "
            f"acc={unseen_acc / trials:.4f} mf1={unseen_mf1 / trials:.4f}"
        )


if __name__ == "__main__":
    main()
