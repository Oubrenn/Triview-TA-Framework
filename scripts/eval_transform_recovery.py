import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEAPretrainDataset, ViewConfig  # noqa: E402
from models import MultiViewModel, TransformPredictor  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, save_eval_records, stable_hash, write_csv, write_run_meta  # noqa: E402
from train_uea import UEAClassifier, collate_pretrain  # noqa: E402
from transforms import band_shift_time, band_shift_time_stft  # noqa: E402


def _parse_float_list(raw) -> List[float]:
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    raw = str(raw or "").strip()
    if not raw:
        return []
    raw = raw.strip("[]()")
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _normalization_range(levels: List[float], use_zero_floor_for_single_level: bool = False) -> float:
    if not levels:
        return 1.0
    uniq = sorted({float(v) for v in levels})
    if len(uniq) >= 2:
        span = uniq[-1] - uniq[0]
        if span > 1e-8:
            return float(span)
    if use_zero_floor_for_single_level:
        single = max(abs(v) for v in uniq)
        if single > 1e-8:
            return float(single)
    fallback = max(abs(v) for v in uniq)
    return float(fallback if fallback > 1e-8 else 1.0)


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


def _mean_std(rows: List[Dict[str, object]], method: str, key: str) -> Tuple[float, float, int]:
    vals = [float(r[key]) for r in rows if str(r.get("checkpoint", "")) == method]
    if not vals:
        return 0.0, 0.0, 0
    vals_t = torch.tensor(vals, dtype=torch.float32)
    mean = float(vals_t.mean().item())
    std = float(vals_t.std(unbiased=False).item()) if vals_t.numel() > 1 else 0.0
    return mean, std, int(vals_t.numel())


def _ensure_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected (B,T) or (B,C,T), got {tuple(x.shape)}")


def _macro_f1(confusion: torch.Tensor) -> float:
    conf = confusion.to(torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


def _to_max_abs_db(gains: torch.Tensor) -> torch.Tensor:
    safe = gains.abs().clamp_min(1e-6)
    db = 20.0 * torch.log10(safe)
    return db.abs().max(dim=-1).values


def _load_checkpoint(path: Path, device: str) -> Dict[str, object]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Invalid checkpoint: {path}")
    return ckpt


def _checkpoint_type(model_state: Dict[str, torch.Tensor]) -> str:
    keys = list(model_state.keys())
    if any(k.startswith("time_encoder.") for k in keys):
        return "pretrain_multiview"
    if any(k.startswith("encoder.") for k in keys):
        return "classifier"
    raise ValueError("Unknown checkpoint model_state layout.")


class _FeatureAdapter:
    def __init__(self, ckpt: Dict[str, object], sample: Dict[str, torch.Tensor], device: str, feature_space: str) -> None:
        self.ckpt = ckpt
        self.config = ckpt["config"]
        self.state = ckpt["model_state"]
        self.device = device
        self.feature_space = feature_space
        self.kind = _checkpoint_type(self.state)
        self.model = self._build_model(sample).to(device)
        self.model.load_state_dict(self.state, strict=True)
        self.model.eval()
        self.aux_heads = self._build_aux_heads() if ("aux_heads_state" in ckpt and self.kind == "pretrain_multiview") else None

    def _build_model(self, sample: Dict[str, torch.Tensor]):
        if self.kind == "pretrain_multiview":
            return MultiViewModel(
                input_dim_time=sample["x_time"].shape[0],
                input_dim_freq=sample["x_freq"].shape[0],
                input_dim_tf=sample["x_tf"].shape[0],
                hidden_dim=int(self.config.get("hidden_dim", 64)),
                output_dim=int(self.config.get("embed_dim", 128)),
                num_heads=int(self.config.get("num_heads", 4)),
                res_blocks=int(self.config.get("res_blocks", 2)),
                backbone=str(self.config.get("backbone", "all")),
                use_se=bool(self.config.get("use_se", False)),
                se_reduction=int(self.config.get("se_reduction", 16)),
                use_temporal_attn=bool(self.config.get("use_temporal_attn", False)),
                use_shared_qk_attn=bool(self.config.get("use_shared_qk_attn", False)),
                shared_qk_heads=int(self.config.get("shared_qk_heads", 4)),
                shared_qk_dropout=float(self.config.get("shared_qk_dropout", 0.0)),
                fuse_dropout=float(self.config.get("fuse_dropout", 0.0)),
            )
        return UEAClassifier(
            input_dim=sample["x_time"].shape[0],
            hidden_dim=int(self.config.get("hidden_dim", 64)),
            embed_dim=int(self.config.get("embed_dim", 128)),
            num_classes=int(self.config.get("num_classes", 8)),
            num_heads=int(self.config.get("num_heads", 4)),
            res_blocks=int(self.config.get("res_blocks", 2)),
            backbone=str(self.config.get("backbone", "all")),
            use_temporal_attn=bool(self.config.get("use_temporal_attn", False)),
            use_se=bool(self.config.get("use_se", False)),
            se_reduction=int(self.config.get("se_reduction", 16)),
            use_shared_qk_attn=bool(self.config.get("use_shared_qk_attn", False)),
            shared_qk_heads=int(self.config.get("shared_qk_heads", 4)),
            shared_qk_dropout=float(self.config.get("shared_qk_dropout", 0.0)),
            fuse_dropout=float(self.config.get("fuse_dropout", 0.0)),
            head_dropout=float(self.config.get("head_dropout", 0.0)),
        )

    def _build_aux_heads(self):
        embed_dim = int(self.config.get("embed_dim", 128))
        color_bands = int(self.config.get("pretrain_color_bands", 8))
        heads = {
            "shift": TransformPredictor(embed_dim, 1).to(self.device),
            "scale": TransformPredictor(embed_dim, 1).to(self.device),
            "color": TransformPredictor(embed_dim, color_bands).to(self.device),
        }
        for name, head in heads.items():
            head.load_state_dict(self.ckpt["aux_heads_state"][name], strict=True)
            head.eval()
        return heads

    def supports_aux(self) -> bool:
        return self.aux_heads is not None

    def _single_encoder_feat(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(_ensure_bct(x), return_intermediates=True)
        return out[self.feature_space]

    def _multi_out(self, x_time=None, x_freq=None, x_tf=None) -> Dict[str, torch.Tensor]:
        return self.model(
            _ensure_bct(x_time) if x_time is not None else None,
            _ensure_bct(x_freq) if x_freq is not None else None,
            _ensure_bct(x_tf) if x_tf is not None else None,
            return_intermediates=True,
        )

    def _shift_time_from_batch(self, x_time: torch.Tensor, shift_bins: torch.Tensor) -> torch.Tensor:
        shift_mode = str(self.config.get("pretrain_shift_mode", "border"))
        use_stft = True
        n_fft = int(self.config.get("n_fft", 256))
        hop_length = int(self.config.get("hop_length", 64))
        win_length = self.config.get("stft_win_length")
        window_name = str(self.config.get("stft_window", "hann"))
        center = bool(self.config.get("stft_center", True))
        outs = []
        for i in range(int(x_time.shape[0])):
            b = float(shift_bins[i].item())
            chans = []
            for c in range(int(x_time.shape[1])):
                s = x_time[i, c]
                if use_stft:
                    shifted = band_shift_time_stft(
                        s,
                        shift_bins=b,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window_name=window_name,
                        center=center,
                        shift_mode=shift_mode,
                    )
                else:
                    shifted = band_shift_time(s, shift_bins=b, shift_mode=shift_mode)
                chans.append(shifted)
            outs.append(torch.stack(chans, dim=0))
        return torch.stack(outs, dim=0)

    def extract_batch_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.kind == "pretrain_multiview":
            key = "h_" if self.feature_space == "h" else "z_"
            shift_out = self._multi_out(x_freq=batch["x_shift_freq"].to(self.device), x_tf=batch["x_shift_tf"].to(self.device))
            scale_out = self._multi_out(
                x_time=batch["x_scale"].to(self.device),
                x_freq=batch["x_scale_freq"].to(self.device),
                x_tf=batch["x_scale_tf"].to(self.device),
            )
            color_out = self._multi_out(
                x_time=batch["x_color"].to(self.device),
                x_freq=batch["x_color_freq"].to(self.device),
                x_tf=batch["x_color_tf"].to(self.device),
            )
            feat_b = torch.cat([shift_out[f"{key}freq"], shift_out[f"{key}tf"]], dim=-1)
            feat_rho = torch.cat([scale_out[f"{key}time"], scale_out[f"{key}freq"], scale_out[f"{key}tf"]], dim=-1)
            feat_g = torch.cat([color_out[f"{key}time"], color_out[f"{key}freq"], color_out[f"{key}tf"]], dim=-1)
            return {
                "feat_b": feat_b,
                "feat_rho": feat_rho,
                "feat_g": feat_g,
                "z_shift_freq": shift_out["z_freq"],
                "z_shift_tf": shift_out["z_tf"],
                "z_scale_time": scale_out["z_time"],
                "z_scale_freq": scale_out["z_freq"],
                "z_scale_tf": scale_out["z_tf"],
                "z_color_time": color_out["z_time"],
                "z_color_freq": color_out["z_freq"],
                "z_color_tf": color_out["z_tf"],
            }

        shift_bins = batch["meta"]["transform_params"]["b"].to(device=self.device, dtype=torch.float32)
        x_shift_time = self._shift_time_from_batch(_ensure_bct(batch["x_time"].to(self.device)), shift_bins)
        feat_b = self._single_encoder_feat(x_shift_time)
        feat_rho = self._single_encoder_feat(batch["x_scale"].to(self.device))
        feat_g = self._single_encoder_feat(batch["x_color"].to(self.device))
        return {"feat_b": feat_b, "feat_rho": feat_rho, "feat_g": feat_g}

    def aux_predict(self, batch_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.aux_heads is None:
            raise ValueError("aux heads not available")
        pred_b = 0.5 * (
            self.aux_heads["shift"](batch_feats["z_shift_freq"])
            + self.aux_heads["shift"](batch_feats["z_shift_tf"])
        )
        pred_b = pred_b.squeeze(-1)
        pred_rho = (
            self.aux_heads["scale"](batch_feats["z_scale_time"])
            + self.aux_heads["scale"](batch_feats["z_scale_freq"])
            + self.aux_heads["scale"](batch_feats["z_scale_tf"])
        ) / 3.0
        pred_rho = pred_rho.squeeze(-1)
        pred_gains = (
            self.aux_heads["color"](batch_feats["z_color_time"])
            + self.aux_heads["color"](batch_feats["z_color_freq"])
            + self.aux_heads["color"](batch_feats["z_color_tf"])
        ) / 3.0
        pred_g = _to_max_abs_db(pred_gains)
        return {"pred_b": pred_b, "pred_rho": pred_rho, "pred_g": pred_g}


def _collect_features_and_targets(
    adapter: _FeatureAdapter,
    loader: DataLoader,
    device: str,
    shift_bins: List[float],
) -> Dict[str, torch.Tensor]:
    feat_b, feat_rho, feat_g = [], [], []
    target_b_idx, target_b, target_rho, target_g = [], [], [], []
    bins = torch.tensor(shift_bins, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in loader:
            out = adapter.extract_batch_features(batch)
            meta = batch["meta"]["transform_params"]
            b = meta["b"].to(device=device, dtype=torch.float32)
            rho = meta["rho"].to(device=device, dtype=torch.float32)
            g = meta["g_db"].to(device=device, dtype=torch.float32)
            b_idx = torch.argmin((b.unsqueeze(-1) - bins.unsqueeze(0)).abs(), dim=-1)

            feat_b.append(out["feat_b"])
            feat_rho.append(out["feat_rho"])
            feat_g.append(out["feat_g"])
            target_b_idx.append(b_idx)
            target_b.append(b)
            target_rho.append(rho)
            target_g.append(g)

    return {
        "feat_b": torch.cat(feat_b, dim=0),
        "feat_rho": torch.cat(feat_rho, dim=0),
        "feat_g": torch.cat(feat_g, dim=0),
        "target_b_idx": torch.cat(target_b_idx, dim=0),
        "target_b": torch.cat(target_b, dim=0),
        "target_rho": torch.cat(target_rho, dim=0),
        "target_g": torch.cat(target_g, dim=0),
    }


def _train_probe_classifier(x: torch.Tensor, y: torch.Tensor, num_classes: int, epochs: int, lr: float, wd: float) -> nn.Module:
    model = nn.Linear(x.shape[1], num_classes, bias=True).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_probe_regressor(x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, wd: float) -> nn.Module:
    model = nn.Linear(x.shape[1], 1, bias=True).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ds = TensorDataset(x, y.unsqueeze(-1))
    loader = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _metrics_from_preds(
    pred_b_idx: torch.Tensor,
    pred_b: torch.Tensor,
    pred_rho: torch.Tensor,
    pred_g: torch.Tensor,
    target_b_idx: torch.Tensor,
    target_b: torch.Tensor,
    target_rho: torch.Tensor,
    target_g: torch.Tensor,
    rho_norm_range: float,
    g_norm_range: float,
) -> Dict[str, float]:
    n = int(target_b.shape[0])
    num_classes = int(target_b_idx.max().item()) + 1
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(target_b_idx.detach().cpu(), pred_b_idx.detach().cpu()):
        confusion[int(t.item()), int(p.item())] += 1

    rho_range = float(rho_norm_range if rho_norm_range > 1e-8 else 1.0)
    g_range = float(g_norm_range if g_norm_range > 1e-8 else 1.0)
    rho_mae = float(torch.abs(pred_rho - target_rho).mean().item())
    g_mae = float(torch.abs(pred_g - target_g).mean().item())

    return {
        "b_acc": float((pred_b_idx == target_b_idx).float().mean().item()),
        "b_macro_f1": _macro_f1(confusion),
        "b_mae": float(torch.abs(pred_b - target_b).mean().item()),
        "rho_mae": rho_mae,
        "rho_mse": float(torch.square(pred_rho - target_rho).mean().item()),
        "rho_nmae": rho_mae / rho_range,
        "rho_norm_range": rho_range,
        "g_mae": g_mae,
        "g_mse": float(torch.square(pred_g - target_g).mean().item()),
        "g_nmae": g_mae / g_range,
        "g_norm_range": g_range,
        "count": float(n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoints.")
    parser.add_argument("--labels", type=str, default="", help="Comma-separated labels; defaults to checkpoint stems.")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--probe-train-split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-seed", type=int, default=2026, help="Deterministic transform draw seed.")
    parser.add_argument("--feature-space", type=str, default="h", choices=["h", "z"])
    parser.add_argument("--recovery-mode", type=str, default="auto", choices=["auto", "aux", "probe"])
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-lr", type=float, default=5e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--nmae-view",
        type=str,
        default="score",
        choices=["score", "nmae"],
        help="score: plot 1-NMAE (higher is better); nmae: plot NMAE directly.",
    )
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
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

    rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for ckpt, ckpt_path, label in zip(loaded, ckpt_paths, labels):
        config = ckpt["config"]
        shift_bins = _parse_float_list(str(config.get("resolved_pretrain_shift_bins", config.get("pretrain_shift_bins", "3,-3"))))
        if not shift_bins:
            shift_bins = [-3.0, 3.0]
        scale_ratios = _parse_float_list(
            str(config.get("resolved_pretrain_scale_ratios", config.get("pretrain_scale_ratios", "0.9,1.1")))
        )
        if not scale_ratios:
            scale_ratios = [0.9, 1.1]
        color_levels = _parse_float_list(
            str(config.get("resolved_pretrain_color_max_gain_db_levels", config.get("pretrain_color_max_gain_db", "6")))
        )
        if not color_levels:
            color_levels = [6.0]
        rho_norm_range = _normalization_range(scale_ratios, use_zero_floor_for_single_level=False)
        g_norm_range = _normalization_range(color_levels, use_zero_floor_for_single_level=True)
        view_config = ViewConfig(
            n_fft=int(config.get("n_fft", 256)),
            hop_length=int(config.get("hop_length", 64)),
            win_length=config.get("stft_win_length"),
            window_name=str(config.get("stft_window", "hann")),
            center=bool(config.get("stft_center", True)),
            magnitude_power=float(config.get("stft_magnitude_power", 1.0)),
            tf_log1p=bool(config.get("tf_log1p", True)),
            tf_flatten=bool(config.get("tf_flatten", True)),
            normalize_mode=str(config.get("normalize_mode", "per_sample_channel")),
            shift_mode=str(config.get("pretrain_shift_mode", "border")),
            shift_bins=shift_bins,
            scale_ratios=scale_ratios if scale_ratios else [0.9, 1.1],
            color_bands=int(config.get("pretrain_color_bands", 8)),
            color_max_gain_db=float(config.get("pretrain_color_max_gain_db", max(color_levels) if color_levels else 6.0)),
            color_max_gain_db_levels=color_levels if color_levels else None,
        )
        ds_train = UEAPretrainDataset(
            dataset_name,
            split=args.probe_train_split,
            pad_to_max=args.pad_to_max,
            view_config=view_config,
            base_seed=args.base_seed,
        )
        ds_eval = UEAPretrainDataset(
            dataset_name,
            split=args.split,
            pad_to_max=args.pad_to_max,
            view_config=view_config,
            base_seed=args.base_seed,
        )
        loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pretrain)
        loader_eval = DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pretrain)
        adapter = _FeatureAdapter(ckpt=ckpt, sample=ds_eval[0], device=device, feature_space=args.feature_space)

        mode = args.recovery_mode
        if mode == "auto":
            mode = "aux" if adapter.supports_aux() else "probe"
        if mode == "aux" and not adapter.supports_aux():
            raise ValueError(f"Checkpoint {ckpt_path.name} has no aux heads; use --recovery-mode probe or auto.")

        eval_data = _collect_features_and_targets(adapter, loader_eval, device, shift_bins=shift_bins)
        if mode == "aux":
            pred_b_idx, pred_b, pred_rho, pred_g = [], [], [], []
            with torch.no_grad():
                bins = torch.tensor(shift_bins, dtype=torch.float32, device=device)
                for batch in loader_eval:
                    feats = adapter.extract_batch_features(batch)
                    pred = adapter.aux_predict(feats)
                    pred_b.append(pred["pred_b"])
                    pred_rho.append(pred["pred_rho"])
                    pred_g.append(pred["pred_g"])
                    pred_b_idx.append(torch.argmin((pred["pred_b"].unsqueeze(-1) - bins.unsqueeze(0)).abs(), dim=-1))
            pred_b_idx = torch.cat(pred_b_idx, dim=0)
            pred_b = torch.cat(pred_b, dim=0)
            pred_rho = torch.cat(pred_rho, dim=0)
            pred_g = torch.cat(pred_g, dim=0)
        else:
            train_data = _collect_features_and_targets(adapter, loader_train, device, shift_bins=shift_bins)
            num_classes = int(train_data["target_b_idx"].max().item()) + 1
            probe_b = _train_probe_classifier(
                train_data["feat_b"], train_data["target_b_idx"], num_classes, args.probe_epochs, args.probe_lr, args.probe_weight_decay
            )
            probe_rho = _train_probe_regressor(
                train_data["feat_rho"], train_data["target_rho"], args.probe_epochs, args.probe_lr, args.probe_weight_decay
            )
            probe_g = _train_probe_regressor(
                train_data["feat_g"], train_data["target_g"], args.probe_epochs, args.probe_lr, args.probe_weight_decay
            )
            with torch.no_grad():
                logits_b = probe_b(eval_data["feat_b"])
                pred_b_idx = logits_b.argmax(dim=1)
                bins = torch.tensor(shift_bins, dtype=torch.float32, device=device)
                pred_b = bins[pred_b_idx]
                pred_rho = probe_rho(eval_data["feat_rho"]).squeeze(-1)
                pred_g = probe_g(eval_data["feat_g"]).squeeze(-1)

        metrics = _metrics_from_preds(
            pred_b_idx=pred_b_idx,
            pred_b=pred_b,
            pred_rho=pred_rho,
            pred_g=pred_g,
            target_b_idx=eval_data["target_b_idx"],
            target_b=eval_data["target_b"],
            target_rho=eval_data["target_rho"],
            target_g=eval_data["target_g"],
            rho_norm_range=rho_norm_range,
            g_norm_range=g_norm_range,
        )
        rows.append(
            {
                "checkpoint": label,
                "checkpoint_path": str(ckpt_path),
                "mode": mode,
                "feature_space": args.feature_space,
                "run_id": ckpt_path.stem,
                **metrics,
            }
        )

        for i in range(int(eval_data["target_b"].shape[0])):
            sample_rows.append(
                {
                    "sample_id": i,
                    "method": label,
                    "mode": mode,
                    "feature_space": args.feature_space,
                    "b_true": float(eval_data["target_b"][i].item()),
                    "b_pred": float(pred_b[i].item()),
                    "b_true_idx": int(eval_data["target_b_idx"][i].item()),
                    "b_pred_idx": int(pred_b_idx[i].item()),
                    "rho_true": float(eval_data["target_rho"][i].item()),
                    "rho_pred": float(pred_rho[i].item()),
                    "g_true": float(eval_data["target_g"][i].item()),
                    "g_pred": float(pred_g[i].item()),
                    "run_id": ckpt_path.stem,
                }
            )

    methods = _ordered_unique(labels)
    xs = list(range(len(methods)))
    colors = [_method_color(m, i) for i, m in enumerate(methods)]

    b_means, b_stds = [], []
    rho_means, rho_stds = [], []
    g_means, g_stds = [], []
    agg_rows: List[Dict[str, object]] = []
    for method in methods:
        b_mean, b_std, n_runs = _mean_std(rows, method, "b_acc")
        rho_mean, rho_std, _ = _mean_std(rows, method, "rho_nmae")
        g_mean, g_std, _ = _mean_std(rows, method, "g_nmae")
        if args.nmae_view == "score":
            rho_plot_mean = 1.0 - rho_mean
            g_plot_mean = 1.0 - g_mean
            rho_plot_std = rho_std
            g_plot_std = g_std
        else:
            rho_plot_mean = rho_mean
            g_plot_mean = g_mean
            rho_plot_std = rho_std
            g_plot_std = g_std
        b_means.append(b_mean)
        b_stds.append(b_std)
        rho_means.append(rho_plot_mean)
        rho_stds.append(rho_plot_std)
        g_means.append(g_plot_mean)
        g_stds.append(g_plot_std)
        agg_rows.append(
            {
                "checkpoint": method,
                "n_runs": n_runs,
                "b_acc_mean": b_mean,
                "b_acc_std": b_std,
                "rho_plot_mean": rho_plot_mean,
                "rho_plot_std": rho_plot_std,
                "g_plot_mean": g_plot_mean,
                "g_plot_std": g_plot_std,
                "nmae_view": args.nmae_view,
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))
    bar_kw = {"width": 0.68, "edgecolor": "black", "linewidth": 0.4}
    bars0 = axes[0].bar(xs, b_means, yerr=b_stds, capsize=3.2, color=colors, **bar_kw)
    bars1 = axes[1].bar(xs, rho_means, yerr=rho_stds, capsize=3.2, color=colors, **bar_kw)
    bars2 = axes[2].bar(xs, g_means, yerr=g_stds, capsize=3.2, color=colors, **bar_kw)
    axes[0].set_title("(a) b Accuracy")
    if args.nmae_view == "score":
        axes[1].set_title("(b) 1 - ρ NMAE")
        axes[2].set_title("(c) 1 - g NMAE")
    else:
        axes[1].set_title("(b) ρ NMAE ↓")
        axes[2].set_title("(c) g NMAE ↓")
    axes[0].set_ylim(0.0, 1.02)

    all_vals = b_means + rho_means + g_means
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
        pad = 0.12 * (hi - lo) if hi > lo else 0.08
        lo_plot = max(0.0, lo - pad) if args.nmae_view == "nmae" else lo - pad
        hi_plot = hi + pad
        if hi_plot <= lo_plot:
            hi_plot = lo_plot + 1e-3
        axes[1].set_ylim(lo_plot, hi_plot)
        axes[2].set_ylim(lo_plot, hi_plot)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=0, ha="center")
        ax.grid(True, axis="y", alpha=0.3)

    for bars, vals, errs, ax in [
        (bars0, b_means, b_stds, axes[0]),
        (bars1, rho_means, rho_stds, axes[1]),
        (bars2, g_means, g_stds, axes[2]),
    ]:
        y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-6)
        for bar, val, err in zip(bars, vals, errs):
            y_text = bar.get_height() + err + 0.015 * y_span
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_text,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    legend_handles = [
        Line2D([0], [0], color=c, lw=6, label=m)
        for c, m in zip(colors, methods)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(legend_handles)),
        frameon=False,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    out_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(out_root)
    sev_hash = stable_hash(
        {
            "labels": labels,
            "split": args.split,
            "probe_train_split": args.probe_train_split,
            "base_seed": args.base_seed,
            "mode": args.recovery_mode,
            "feature_space": args.feature_space,
            "nmae_view": args.nmae_view,
        }
    )
    stem = build_tag("transform_recovery", dataset_name, args.split, f"seed{args.seed}", f"sev{sev_hash}")
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    agg_csv = csv_dir / f"{stem}_agg_plot.csv"
    sample_csv = csv_dir / f"{stem}_sample_records.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, rows)
    write_csv(agg_csv, agg_rows)
    save_eval_records(sample_rows, sample_csv)
    meta = write_run_meta(
        output_root=out_root,
        script_name="scripts/eval_transform_recovery.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "agg_plot_csv": str(agg_csv),
            "sample_csv": str(sample_csv),
            "labels": labels,
            "checkpoints": [str(p) for p in ckpt_paths],
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_agg_csv={agg_csv}")
    print(f"saved_sample_csv={sample_csv}")
    print(f"saved_meta={meta}")


if __name__ == "__main__":
    main()
