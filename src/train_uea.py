import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR

from datasets import UEAPretrainDataset, UEATimeSeriesDataset, ViewConfig
from losses import multi_domain_consistency_loss, ta_cfc_loss, color_regression_loss
from models import MultiViewModel, build_encoder, TransformPredictor, build_shared_qk_attn


DATASET_TRAIN_PROFILES = {
    "facedetection": {
        "n_fft": 16,
        "hop_length": 4,
        "val_split": 0.3,
        "ta_pair_mode": "plain_cfc",
    },
    "heartbeat": {
        "val_split": 0.3,
        "class_weight_mode": "balanced",
        "label_smoothing": 0.0,
    },
    "handwriting": {
        # Handwriting has only 150 train samples over 26 classes; hold-out split can
        # collapse per-class support and hurt optimization stability.
        "val_split": 0.0,
        # Handwriting sequence length is short (~152), so default STFT settings from
        # larger datasets (n_fft=256/hop=64) are not a good fit for tri-view/pretrain.
        "n_fft": 32,
        "hop_length": 8,
    },
    "hhar": {
        # HHAR windows are typically short (e.g., 128), so reduce STFT defaults.
        "n_fft": 64,
        "hop_length": 16,
        # Keep a small validation split for model selection on train domains.
        "val_split": 0.1,
        "val_split_mode": "domain_stratified",
    },
}


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
        use_se: bool = False,
        se_reduction: int = 16,
        use_shared_qk_attn: bool = False,
        shared_qk_heads: int = 4,
        shared_qk_dropout: float = 0.0,
        fuse_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        shared_qk_attn = (
            build_shared_qk_attn(backbone, hidden_dim, shared_qk_heads, shared_qk_dropout)
            if use_shared_qk_attn
            else None
        )
        self.encoder = build_encoder(
            backbone=backbone,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_heads=num_heads,
            res_blocks=res_blocks,
            use_se=use_se,
            se_reduction=se_reduction,
            use_temporal_attn=use_temporal_attn,
            shared_qk_attn=shared_qk_attn,
            shared_domain="time",
            fuse_dropout=fuse_dropout,
        )
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        if return_intermediates:
            if hasattr(self.encoder, "forward_with_attn"):
                try:
                    h, attn = self.encoder.forward_with_attn(x)
                except TypeError:
                    h, attn = self.encoder.forward_with_attn(x, is_tf=False)
            else:
                h, attn = self.encoder(x), None
            z = self.head_dropout(h)
            logits = self.classifier(z)
            return {
                "h": h,
                "z": z,
                "attn": attn,
                "logits": logits,
                "pred_b": None,
                "pred_rho": None,
                "pred_g": None,
            }
        z = self.encoder(x)
        z = self.head_dropout(z)
        return self.classifier(z)


class UEATriViewClassifier(nn.Module):
    def __init__(
        self,
        input_dim_time: int,
        input_dim_freq: int,
        input_dim_tf: int,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        res_blocks: int,
        backbone: str,
        use_temporal_attn: bool = False,
        use_se: bool = False,
        se_reduction: int = 16,
        use_shared_qk_attn: bool = False,
        shared_qk_heads: int = 4,
        shared_qk_dropout: float = 0.0,
        triview_fusion: str = "gated",
        gate_hidden_dim: int = 64,
        gate_dropout: float = 0.0,
        gate_temperature: float = 1.0,
        fuse_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if triview_fusion not in {"concat", "gated"}:
            raise ValueError("triview_fusion must be one of {'concat', 'gated'}.")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")
        self.encoder = MultiViewModel(
            input_dim_time=input_dim_time,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            use_projectors=False,
            num_heads=num_heads,
            res_blocks=res_blocks,
            backbone=backbone,
            use_se=use_se,
            se_reduction=se_reduction,
            use_temporal_attn=use_temporal_attn,
            use_shared_qk_attn=use_shared_qk_attn,
            shared_qk_heads=shared_qk_heads,
            shared_qk_dropout=shared_qk_dropout,
            fuse_dropout=fuse_dropout,
        )
        self.triview_fusion = triview_fusion
        self.gate_temperature = gate_temperature
        fused_dim = embed_dim * 3
        if triview_fusion == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(fused_dim, gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(gate_dropout) if gate_dropout > 0.0 else nn.Identity(),
                nn.Linear(gate_hidden_dim, 3),
            )
        else:
            self.gate_mlp = None
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor,
        x_tf: torch.Tensor,
    ) -> torch.Tensor:
        h_time, h_freq, h_tf = self.encoder(x_time=x_time, x_freq=x_freq, x_tf=x_tf)
        if h_time is None or h_freq is None or h_tf is None:
            raise ValueError("Tri-view classifier requires x_time, x_freq, and x_tf.")
        stacked = torch.stack([h_time, h_freq, h_tf], dim=1)  # (B, 3, D)
        if self.gate_mlp is not None:
            gate_input = torch.cat([h_time, h_freq, h_tf], dim=-1)
            gate_logits = self.gate_mlp(gate_input) / self.gate_temperature
            gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, 3, 1)
            stacked = stacked * gate_weights
        fused = stacked.flatten(1)
        fused = self.head_dropout(fused)
        return self.classifier(fused)


class UEAFreqViewClassifier(nn.Module):
    def __init__(
        self,
        input_dim_time: int,
        input_dim_freq: int,
        hidden_dim: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        res_blocks: int,
        backbone: str,
        use_temporal_attn: bool = False,
        use_se: bool = False,
        se_reduction: int = 16,
        use_shared_qk_attn: bool = False,
        shared_qk_heads: int = 4,
        shared_qk_dropout: float = 0.0,
        triview_fusion: str = "gated",
        gate_hidden_dim: int = 64,
        gate_dropout: float = 0.0,
        gate_temperature: float = 1.0,
        fuse_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if triview_fusion not in {"concat", "gated"}:
            raise ValueError("triview_fusion must be one of {'concat', 'gated'}.")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")
        self.encoder = MultiViewModel(
            input_dim_time=input_dim_time,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_freq,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            use_projectors=False,
            num_heads=num_heads,
            res_blocks=res_blocks,
            backbone=backbone,
            use_se=use_se,
            se_reduction=se_reduction,
            use_temporal_attn=use_temporal_attn,
            use_shared_qk_attn=use_shared_qk_attn,
            shared_qk_heads=shared_qk_heads,
            shared_qk_dropout=shared_qk_dropout,
            fuse_dropout=fuse_dropout,
        )
        self.triview_fusion = triview_fusion
        self.gate_temperature = gate_temperature
        fused_dim = embed_dim * 2
        if triview_fusion == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(fused_dim, gate_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(gate_dropout) if gate_dropout > 0.0 else nn.Identity(),
                nn.Linear(gate_hidden_dim, 2),
            )
        else:
            self.gate_mlp = None
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        x_time: torch.Tensor,
        x_freq: torch.Tensor,
    ) -> torch.Tensor:
        h_time, h_freq, _ = self.encoder(x_time=x_time, x_freq=x_freq, x_tf=None)
        if h_time is None or h_freq is None:
            raise ValueError("Time+freq classifier requires x_time and x_freq.")
        stacked = torch.stack([h_time, h_freq], dim=1)  # (B, 2, D)
        if self.gate_mlp is not None:
            gate_input = torch.cat([h_time, h_freq], dim=-1)
            gate_logits = self.gate_mlp(gate_input) / self.gate_temperature
            gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)  # (B, 2, 1)
            stacked = stacked * gate_weights
        fused = stacked.flatten(1)
        fused = self.head_dropout(fused)
        return self.classifier(fused)


def collate_pretrain(batch) -> Dict[str, torch.Tensor]:
    x_time = torch.stack([b["x_time"] for b in batch])
    x_freq = torch.stack([b["x_freq"] for b in batch])
    x_tf = torch.stack([b["x_tf"] for b in batch])
    x_shift_freq = torch.stack([b["x_shift_freq"] for b in batch])
    x_shift_tf = torch.stack([b["x_shift_tf"] for b in batch])
    x_scale = torch.stack([b["x_scale"] for b in batch])
    x_scale_freq = torch.stack([b["x_scale_freq"] for b in batch])
    x_scale_tf = torch.stack([b["x_scale_tf"] for b in batch])
    x_color = torch.stack([b["x_color"] for b in batch])
    x_color_freq = torch.stack([b["x_color_freq"] for b in batch])
    x_color_tf = torch.stack([b["x_color_tf"] for b in batch])
    meta_shift = torch.stack([b["meta"]["shift"]["shift_bins"] for b in batch])
    meta_scale = torch.stack([b["meta"]["scale"]["scale_ratio"] for b in batch])
    meta_color = torch.stack([b["meta"]["color"]["color_gains"] for b in batch])
    meta_color_max_db = (
        torch.stack([b["meta"]["color"]["max_gain_db"] for b in batch])
        if "max_gain_db" in batch[0]["meta"]["color"]
        else None
    )
    meta_seed = torch.stack([b["meta"]["seed"] for b in batch]) if "seed" in batch[0]["meta"] else None
    meta_domain_id = (
        torch.stack([b["meta"]["domain"]["id"] for b in batch])
        if "domain" in batch[0]["meta"] and "id" in batch[0]["meta"]["domain"]
        else None
    )
    meta_transform_b = (
        torch.stack([b["meta"]["transform_params"]["b"] for b in batch])
        if "transform_params" in batch[0]["meta"] and "b" in batch[0]["meta"]["transform_params"]
        else None
    )
    meta_transform_rho = (
        torch.stack([b["meta"]["transform_params"]["rho"] for b in batch])
        if "transform_params" in batch[0]["meta"] and "rho" in batch[0]["meta"]["transform_params"]
        else None
    )
    meta_transform_g_db = (
        torch.stack([b["meta"]["transform_params"]["g_db"] for b in batch])
        if "transform_params" in batch[0]["meta"] and "g_db" in batch[0]["meta"]["transform_params"]
        else None
    )
    meta_transform_color_id = (
        torch.stack([b["meta"]["transform_params"]["color_id"] for b in batch])
        if "transform_params" in batch[0]["meta"] and "color_id" in batch[0]["meta"]["transform_params"]
        else None
    )
    shift_severity = (
        torch.stack([b["meta"]["shift"]["severity_id"] for b in batch])
        if "severity_id" in batch[0]["meta"]["shift"]
        else None
    )
    scale_severity = (
        torch.stack([b["meta"]["scale"]["severity_id"] for b in batch])
        if "severity_id" in batch[0]["meta"]["scale"]
        else None
    )
    color_severity = (
        torch.stack([b["meta"]["color"]["severity_id"] for b in batch])
        if "severity_id" in batch[0]["meta"]["color"]
        else None
    )
    return {
        "x_time": x_time,
        "x_freq": x_freq,
        "x_tf": x_tf,
        "x_shift_freq": x_shift_freq,
        "x_shift_tf": x_shift_tf,
        "x_scale": x_scale,
        "x_scale_freq": x_scale_freq,
        "x_scale_tf": x_scale_tf,
        "x_color": x_color,
        "x_color_freq": x_color_freq,
        "x_color_tf": x_color_tf,
        "meta": {
            "shift_bins": meta_shift,
            "scale_ratio": meta_scale,
            "color_gains": meta_color,
            "color_max_gain_db": (
                meta_color_max_db if meta_color_max_db is not None else torch.zeros(meta_shift.shape[0], dtype=torch.float32)
            ),
            "seed": meta_seed if meta_seed is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long),
            "domain_id": meta_domain_id if meta_domain_id is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long),
            "shift_severity_id": (
                shift_severity if shift_severity is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long)
            ),
            "scale_severity_id": (
                scale_severity if scale_severity is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long)
            ),
            "color_severity_id": (
                color_severity if color_severity is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long)
            ),
            "transform_params": {
                "b": meta_transform_b if meta_transform_b is not None else meta_shift.to(dtype=torch.float32),
                "rho": meta_transform_rho if meta_transform_rho is not None else meta_scale.to(dtype=torch.float32),
                "g_db": (
                    meta_transform_g_db
                    if meta_transform_g_db is not None
                    else (
                        meta_color_max_db
                        if meta_color_max_db is not None
                        else torch.zeros(meta_shift.shape[0], dtype=torch.float32)
                    )
                ),
                "color_id": (
                    meta_transform_color_id
                    if meta_transform_color_id is not None
                    else (
                        color_severity if color_severity is not None else torch.zeros(meta_shift.shape[0], dtype=torch.long)
                    )
                ),
            },
        },
    }


def collate_fn(batch) -> Dict[str, torch.Tensor]:
    x_time = torch.stack([b["x_time"] if b["x_time"].dim() == 2 else b["x_time"].unsqueeze(0) for b in batch])
    y = torch.stack([b["y"] for b in batch])
    lengths = torch.stack([b["length"] for b in batch])
    out = {"x_time": x_time, "y": y, "length": lengths}
    if "x_freq" in batch[0]:
        out["x_freq"] = torch.stack(
            [b["x_freq"] if b["x_freq"].dim() == 2 else b["x_freq"].unsqueeze(0) for b in batch]
        )
    if "x_tf" in batch[0]:
        out["x_tf"] = torch.stack(
            [b["x_tf"] if b["x_tf"].dim() == 2 else b["x_tf"].unsqueeze(0) for b in batch]
        )
    if "meta" in batch[0]:
        domain_ids = []
        has_domain = True
        for sample in batch:
            meta = sample.get("meta", {})
            domain_id = None
            if isinstance(meta, dict):
                if "domain_id" in meta:
                    domain_id = meta["domain_id"]
                elif isinstance(meta.get("domain"), dict) and "id" in meta["domain"]:
                    domain_id = meta["domain"]["id"]
            if domain_id is None:
                has_domain = False
                break
            if not torch.is_tensor(domain_id):
                domain_id = torch.tensor(int(domain_id), dtype=torch.long)
            domain_ids.append(domain_id.to(dtype=torch.long).view(()))
        if has_domain and domain_ids:
            out["domain_id"] = torch.stack(domain_ids)
    return out


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    for_pretrain: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    sampler=None,
) -> DataLoader:
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        collate_fn=collate_pretrain if for_pretrain else collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def _update_confusion(confusion: Optional[torch.Tensor], y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    if confusion is None:
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        y_true = y_true.view(-1).to(torch.long).cpu()
        y_pred = y_pred.view(-1).to(torch.long).cpu()
        idx = y_true * num_classes + y_pred
        bins = torch.bincount(idx, minlength=num_classes * num_classes)
        confusion += bins.view(num_classes, num_classes)
    return confusion


def _macro_f1_from_confusion(confusion: Optional[torch.Tensor]) -> float:
    if confusion is None:
        return 0.0
    conf = confusion.to(dtype=torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item()


def _parse_float_list(raw: str) -> list:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _check_pretrain_compatibility(args, checkpoint_config: Dict[str, object]) -> Optional[str]:
    # These fields define encoder architecture and must match to load pretrain weights safely.
    keys = [
        "backbone",
        "hidden_dim",
        "embed_dim",
        "num_heads",
        "res_blocks",
        "use_temporal_attn",
        "use_shared_qk_attn",
        "shared_qk_heads",
    ]
    mismatches = []
    for key in keys:
        expected = checkpoint_config.get(key)
        actual = getattr(args, key, None)
        if expected != actual:
            mismatches.append(f"{key}: ckpt={expected} current={actual}")
    if not mismatches:
        return None
    return "; ".join(mismatches)


def _explicit_arg_dests(parser: argparse.ArgumentParser) -> set:
    explicit_opts = set()
    for token in sys.argv[1:]:
        if token == "--":
            break
        if token.startswith("--"):
            explicit_opts.add(token.split("=", 1)[0])
    explicit_dests = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        if any(opt in explicit_opts for opt in action.option_strings):
            explicit_dests.add(action.dest)
    return explicit_dests


def _apply_dataset_profile(
    args,
    parser: argparse.ArgumentParser,
    explicit_dests: Optional[set] = None,
) -> Dict[str, object]:
    profile_key = (args.dataset_profile or "auto").strip().lower()
    if profile_key == "none":
        return {}
    if profile_key == "auto":
        profile_key = str(args.dataset).strip().lower()
    profile = DATASET_TRAIN_PROFILES.get(profile_key)
    if profile is None:
        return {}

    applied = {}
    explicit_dests = explicit_dests or set()
    for key, value in profile.items():
        if not hasattr(args, key):
            continue
        if key in explicit_dests:
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
            applied[key] = value
    return applied


def _linear_ramp(epoch: int, start: int, length: int) -> float:
    if length <= 0:
        return 1.0
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start + 1) / length)


def _pool_attn_feat(attn_feat: torch.Tensor) -> torch.Tensor:
    if attn_feat.dim() == 3:
        return attn_feat.mean(dim=-1)
    if attn_feat.dim() == 2:
        return attn_feat
    raise ValueError("Expected attention feature with shape (B, C, T) or (B, C).")


def _attn_consistency_loss(attn_time, attn_freq, attn_tf) -> torch.Tensor:
    if attn_time is None or attn_freq is None or attn_tf is None:
        return 0.0
    if isinstance(attn_time, dict):
        losses = []
        for key, feat_time in attn_time.items():
            feat_freq = attn_freq.get(key) if isinstance(attn_freq, dict) else None
            feat_tf = attn_tf.get(key) if isinstance(attn_tf, dict) else None
            if feat_time is None or feat_freq is None or feat_tf is None:
                continue
            vec_time = _pool_attn_feat(feat_time)
            vec_freq = _pool_attn_feat(feat_freq)
            vec_tf = _pool_attn_feat(feat_tf)
            losses.append(multi_domain_consistency_loss(vec_time, vec_freq, vec_tf))
        if not losses:
            return 0.0
        return sum(losses) / len(losses)
    vec_time = _pool_attn_feat(attn_time)
    vec_freq = _pool_attn_feat(attn_freq)
    vec_tf = _pool_attn_feat(attn_tf)
    return multi_domain_consistency_loss(vec_time, vec_freq, vec_tf)


def _update_bn_from_loader(loader: DataLoader, model: nn.Module, device: str, supervised_views: str = "time") -> None:
    bn_layers = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            bn_layers.append(module)
    if not bn_layers:
        return

    momenta = {bn: bn.momentum for bn in bn_layers}
    for bn in bn_layers:
        bn.momentum = None

    was_training = model.training
    model.train()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                if supervised_views == "triview" and all(k in batch for k in ("x_time", "x_freq", "x_tf")):
                    x_time = batch["x_time"].to(device)
                    x_freq = batch["x_freq"].to(device)
                    x_tf = batch["x_tf"].to(device)
                    model(x_time, x_freq, x_tf)
                    continue
                if supervised_views == "timefreq" and all(k in batch for k in ("x_time", "x_freq")):
                    x_time = batch["x_time"].to(device)
                    x_freq = batch["x_freq"].to(device)
                    model(x_time, x_freq)
                    continue
                if "x_time" in batch:
                    inputs = batch["x_time"]
                elif "x" in batch:
                    inputs = batch["x"]
                else:
                    continue
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            if inputs is None:
                continue
            inputs = inputs.to(device)
            model(inputs)

    for bn in bn_layers:
        bn.momentum = momenta[bn]
    model.train(was_training)


def _ensure_bct(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must follow (B, C, T), got shape={tuple(x.shape)}.")


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: str, strict: bool = True) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"model_state": checkpoint}


def _classification_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float,
    class_weights: Optional[torch.Tensor],
    loss_type: str,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_type == "ce":
        return F.cross_entropy(logits, y, weight=class_weights, label_smoothing=label_smoothing)
    if loss_type != "focal":
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # Focal loss with optional class-balancing alpha from class_weights.
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    pt = torch.softmax(logits, dim=1).gather(1, y.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
    focal_weight = (1.0 - pt).pow(focal_gamma)
    if class_weights is not None:
        focal_weight = focal_weight * class_weights[y]
    return (focal_weight * ce).mean()


def _rex_penalty(domain_losses: List[torch.Tensor]) -> torch.Tensor:
    if len(domain_losses) <= 1:
        device = domain_losses[0].device if domain_losses else "cpu"
        return torch.tensor(0.0, device=device)
    stacked = torch.stack(domain_losses)
    mean = stacked.mean()
    return ((stacked - mean) ** 2).mean()


def _irm_penalty(
    domain_logits: List[torch.Tensor],
    domain_targets: List[torch.Tensor],
    *,
    label_smoothing: float,
    class_weights: Optional[torch.Tensor],
    loss_type: str,
    focal_gamma: float,
) -> torch.Tensor:
    if len(domain_logits) <= 1:
        device = domain_logits[0].device if domain_logits else "cpu"
        return torch.tensor(0.0, device=device)
    scale = torch.tensor(1.0, device=domain_logits[0].device, requires_grad=True)
    penalties = []
    for logits_env, targets_env in zip(domain_logits, domain_targets):
        env_loss = _classification_loss(
            logits=logits_env * scale,
            y=targets_env,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        grad = torch.autograd.grad(env_loss, [scale], create_graph=True)[0]
        penalties.append(grad.pow(2))
    return torch.stack(penalties).mean()


def run_epoch(
    model,
    loader,
    optimizer=None,
    device="cpu",
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    supervised_views: str = "time",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    logit_adjustment: Optional[torch.Tensor] = None,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    dg_method: str = "erm",
    dg_lambda: float = 0.0,
    dg_min_group_size: int = 2,
):
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = None
    loss_weight = class_weights.to(device=device, dtype=torch.float32) if class_weights is not None else None
    logit_adjust = (
        logit_adjustment.to(device=device, dtype=torch.float32).view(1, -1)
        if logit_adjustment is not None
        else None
    )

    if optimizer is None:
        model.eval()
        context = torch.no_grad()
    else:
        model.train()
        context = torch.enable_grad()

    with context:
        for batch in loader:
            x_time = _ensure_bct(batch["x_time"].to(device), "x_time")
            y = batch["y"].to(device)
            domain_ids = batch.get("domain_id")
            if domain_ids is not None:
                if not torch.is_tensor(domain_ids):
                    domain_ids = torch.tensor(domain_ids, dtype=torch.long)
                domain_ids = domain_ids.to(device=device, dtype=torch.long)
            mixup_active = False
            mixup_lam = 1.0
            y_mix = None
            if (
                optimizer is not None
                and mixup_alpha > 0.0
                and mixup_prob > 0.0
                and y.size(0) > 1
                and torch.rand(1).item() < mixup_prob
            ):
                beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
                mixup_lam = float(beta.sample(()).item())
                perm = torch.randperm(y.size(0), device=device)
                x_time = mixup_lam * x_time + (1.0 - mixup_lam) * x_time[perm]
                y_mix = y[perm]
                mixup_active = True

            if supervised_views == "triview":
                x_freq = _ensure_bct(batch["x_freq"].to(device), "x_freq")
                x_tf = _ensure_bct(batch["x_tf"].to(device), "x_tf")
                if mixup_active:
                    x_freq = mixup_lam * x_freq + (1.0 - mixup_lam) * x_freq[perm]
                    x_tf = mixup_lam * x_tf + (1.0 - mixup_lam) * x_tf[perm]
                logits = model(x_time, x_freq, x_tf)
            elif supervised_views == "timefreq":
                x_freq = _ensure_bct(batch["x_freq"].to(device), "x_freq")
                if mixup_active:
                    x_freq = mixup_lam * x_freq + (1.0 - mixup_lam) * x_freq[perm]
                logits = model(x_time, x_freq)
            else:
                logits = model(x_time)
            logits_for_loss = logits + logit_adjust if logit_adjust is not None else logits
            if mixup_active and y_mix is not None:
                loss_main = _classification_loss(
                    logits=logits_for_loss,
                    y=y,
                    label_smoothing=label_smoothing,
                    class_weights=loss_weight,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                )
                loss_aux = _classification_loss(
                    logits=logits_for_loss,
                    y=y_mix,
                    label_smoothing=label_smoothing,
                    class_weights=loss_weight,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                )
                loss = mixup_lam * loss_main + (1.0 - mixup_lam) * loss_aux
            else:
                loss = _classification_loss(
                    logits=logits_for_loss,
                    y=y,
                    label_smoothing=label_smoothing,
                    class_weights=loss_weight,
                    loss_type=loss_type,
                    focal_gamma=focal_gamma,
                )
            if (
                optimizer is not None
                and dg_method != "erm"
                and dg_lambda > 0.0
                and domain_ids is not None
                and not mixup_active
            ):
                domain_losses: List[torch.Tensor] = []
                domain_logits: List[torch.Tensor] = []
                domain_targets: List[torch.Tensor] = []
                for domain_id in torch.unique(domain_ids):
                    mask = domain_ids == domain_id
                    if int(mask.sum().item()) < max(1, int(dg_min_group_size)):
                        continue
                    logits_env = logits_for_loss[mask]
                    targets_env = y[mask]
                    domain_logits.append(logits_env)
                    domain_targets.append(targets_env)
                    domain_losses.append(
                        _classification_loss(
                            logits=logits_env,
                            y=targets_env,
                            label_smoothing=label_smoothing,
                            class_weights=loss_weight,
                            loss_type=loss_type,
                            focal_gamma=focal_gamma,
                        )
                    )
                if len(domain_losses) >= 2:
                    if dg_method == "rex":
                        dg_penalty = _rex_penalty(domain_losses)
                    elif dg_method == "irm":
                        dg_penalty = _irm_penalty(
                            domain_logits,
                            domain_targets,
                            label_smoothing=label_smoothing,
                            class_weights=loss_weight,
                            loss_type=loss_type,
                            focal_gamma=focal_gamma,
                        )
                    else:
                        raise ValueError(f"Unsupported dg_method: {dg_method}")
                    loss = loss + dg_lambda * dg_penalty

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits_for_loss.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            confusion = _update_confusion(confusion, y, preds, logits.size(1))
            total_count += y.size(0)

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    avg_mf1 = _macro_f1_from_confusion(confusion)
    return avg_loss, avg_acc, avg_mf1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    supervised_views: str = "time",
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    logit_adjustment: Optional[torch.Tensor] = None,
) -> tuple:
    return run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        supervised_views=supervised_views,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
        logit_adjustment=logit_adjustment,
    )


def run_pretrain_epoch(
    model: MultiViewModel,
    loader: DataLoader,
    optimizer=None,
    device="cpu",
    ta_mode: str = "vicreg",
    ta_pair_mode: str = "full",
    ta_shuffle_pairs: bool = False,
    lambda_md: float = 1.0,
    lambda_ta: float = 1.0,
    lambda_shift: float = 1.0,
    lambda_scale: float = 1.0,
    lambda_color: float = 1.0,
    lambda_attn: float = 1.0,
    aux_heads=None,
):
    total_loss = 0.0
    total_md = 0.0
    total_ta = 0.0
    total_aux = 0.0
    total_shift_loss = 0.0
    total_scale_loss = 0.0
    total_color_loss = 0.0
    total_shift_mae = 0.0
    total_scale_mae = 0.0
    total_color_mae = 0.0
    total_count = 0

    if optimizer is None:
        model.eval()
        context = torch.no_grad()
    else:
        model.train()
        context = torch.enable_grad()
    if aux_heads is not None:
        if optimizer is None:
            for head in aux_heads.values():
                head.eval()
        else:
            for head in aux_heads.values():
                head.train()

    with context:
        for batch in loader:
            x_time = _ensure_bct(batch["x_time"].to(device), "x_time")
            x_freq = _ensure_bct(batch["x_freq"].to(device), "x_freq")
            x_tf = _ensure_bct(batch["x_tf"].to(device), "x_tf")
            x_shift_freq = _ensure_bct(batch["x_shift_freq"].to(device), "x_shift_freq")
            x_shift_tf = _ensure_bct(batch["x_shift_tf"].to(device), "x_shift_tf")
            x_scale = _ensure_bct(batch["x_scale"].to(device), "x_scale")
            x_scale_freq = _ensure_bct(batch["x_scale_freq"].to(device), "x_scale_freq")
            x_scale_tf = _ensure_bct(batch["x_scale_tf"].to(device), "x_scale_tf")
            x_color = _ensure_bct(batch["x_color"].to(device), "x_color")
            x_color_freq = _ensure_bct(batch["x_color_freq"].to(device), "x_color_freq")
            x_color_tf = _ensure_bct(batch["x_color_tf"].to(device), "x_color_tf")

            if hasattr(model, "forward_with_attn"):
                (z_time, z_freq, z_tf), attn_info = model.forward_with_attn(x_time, x_freq, x_tf)
            else:
                z_time, z_freq, z_tf = model(x_time, x_freq, x_tf)
                attn_info = None
            z_scale_time, z_scale_freq, z_scale_tf = model(x_scale, x_scale_freq, x_scale_tf)
            z_color_time, z_color_freq, z_color_tf = model(x_color, x_color_freq, x_color_tf)
            _, z_shift_freq, z_shift_tf = model(None, x_shift_freq, x_shift_tf)

            loss_md = multi_domain_consistency_loss(z_time, z_freq, z_tf)
            pair_perm = None
            if ta_shuffle_pairs and z_time.shape[0] > 1:
                pair_perm = torch.randperm(z_time.shape[0], device=z_time.device)

            def _paired(z: torch.Tensor) -> torch.Tensor:
                if pair_perm is None:
                    return z
                return z[pair_perm]

            if ta_pair_mode == "full":
                ta_pairs = [
                    (z_time, z_scale_time),
                    (z_freq, z_scale_freq),
                    (z_tf, z_scale_tf),
                    (z_time, z_color_time),
                    (z_freq, z_color_freq),
                    (z_tf, z_color_tf),
                    (z_freq, z_shift_freq),
                    (z_tf, z_shift_tf),
                    # Cross-domain transform consistency (shifted freq vs shifted TF).
                    (z_shift_freq, z_shift_tf),
                    # Cross-domain alignment to time view (shifted freq/TF vs time).
                    (z_time, z_shift_freq),
                    (z_time, z_shift_tf),
                ]
            elif ta_pair_mode == "same_domain":
                ta_pairs = [
                    (z_time, z_scale_time),
                    (z_freq, z_scale_freq),
                    (z_tf, z_scale_tf),
                    (z_time, z_color_time),
                    (z_freq, z_color_freq),
                    (z_tf, z_color_tf),
                    (z_freq, z_shift_freq),
                    (z_tf, z_shift_tf),
                ]
            elif ta_pair_mode == "plain_cfc":
                # Non-transform-aware cross-frequency consistency on clean views only.
                ta_pairs = [
                    (z_time, z_freq),
                    (z_time, z_tf),
                    (z_freq, z_tf),
                ]
            else:
                raise ValueError(f"Unknown ta_pair_mode: {ta_pair_mode}")

            ta_terms = [ta_cfc_loss(z_anchor, _paired(z_warped), mode=ta_mode) for z_anchor, z_warped in ta_pairs]
            loss_ta = sum(ta_terms) / len(ta_terms) if ta_terms else torch.tensor(0.0, device=x_time.device)
            loss_attn = 0.0
            if attn_info is not None:
                loss_attn = _attn_consistency_loss(
                    attn_info.get("time"),
                    attn_info.get("freq"),
                    attn_info.get("tf"),
                )
            loss_aux = 0.0
            loss_shift = torch.tensor(0.0, device=x_time.device)
            loss_scale = torch.tensor(0.0, device=x_time.device)
            loss_color = torch.tensor(0.0, device=x_time.device)
            shift_mae = torch.tensor(0.0, device=x_time.device)
            scale_mae = torch.tensor(0.0, device=x_time.device)
            color_mae = torch.tensor(0.0, device=x_time.device)
            if aux_heads is not None:
                meta = batch["meta"]
                shift_bins = meta["shift_bins"].to(device=device, dtype=torch.float32).unsqueeze(-1)
                scale_ratio = meta["scale_ratio"].to(device=device, dtype=torch.float32).unsqueeze(-1)
                color_gains = meta["color_gains"].to(device=device, dtype=torch.float32)

                pred_shift_freq = aux_heads["shift"](z_shift_freq)
                pred_shift_tf = aux_heads["shift"](z_shift_tf)
                loss_shift = 0.5 * (
                    F.mse_loss(pred_shift_freq, shift_bins) + F.mse_loss(pred_shift_tf, shift_bins)
                )
                shift_mae = 0.5 * (
                    F.l1_loss(pred_shift_freq, shift_bins) + F.l1_loss(pred_shift_tf, shift_bins)
                )

                pred_scale_time = aux_heads["scale"](z_scale_time)
                pred_scale_freq = aux_heads["scale"](z_scale_freq)
                pred_scale_tf = aux_heads["scale"](z_scale_tf)
                loss_scale = (
                    F.mse_loss(pred_scale_time, scale_ratio)
                    + F.mse_loss(pred_scale_freq, scale_ratio)
                    + F.mse_loss(pred_scale_tf, scale_ratio)
                ) / 3.0
                scale_mae = (
                    F.l1_loss(pred_scale_time, scale_ratio)
                    + F.l1_loss(pred_scale_freq, scale_ratio)
                    + F.l1_loss(pred_scale_tf, scale_ratio)
                ) / 3.0

                pred_color_time = aux_heads["color"](z_color_time)
                pred_color_freq = aux_heads["color"](z_color_freq)
                pred_color_tf = aux_heads["color"](z_color_tf)
                loss_color = (
                    color_regression_loss(pred_color_time, color_gains)
                    + color_regression_loss(pred_color_freq, color_gains)
                    + color_regression_loss(pred_color_tf, color_gains)
                ) / 3.0
                color_mae = (
                    F.l1_loss(pred_color_time, color_gains)
                    + F.l1_loss(pred_color_freq, color_gains)
                    + F.l1_loss(pred_color_tf, color_gains)
                ) / 3.0

                loss_aux = (
                    lambda_shift * loss_shift
                    + lambda_scale * loss_scale
                    + lambda_color * loss_color
                )

            loss = lambda_md * loss_md + lambda_ta * loss_ta + loss_aux + lambda_attn * loss_attn

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = x_time.size(0)
            total_loss += loss.item() * batch_size
            total_md += loss_md.item() * batch_size
            total_ta += loss_ta.item() * batch_size
            total_aux += (loss_aux.item() if torch.is_tensor(loss_aux) else float(loss_aux)) * batch_size
            total_shift_loss += (
                loss_shift.item() if torch.is_tensor(loss_shift) else float(loss_shift)
            ) * batch_size
            total_scale_loss += (
                loss_scale.item() if torch.is_tensor(loss_scale) else float(loss_scale)
            ) * batch_size
            total_color_loss += (
                loss_color.item() if torch.is_tensor(loss_color) else float(loss_color)
            ) * batch_size
            total_shift_mae += (
                shift_mae.item() if torch.is_tensor(shift_mae) else float(shift_mae)
            ) * batch_size
            total_scale_mae += (
                scale_mae.item() if torch.is_tensor(scale_mae) else float(scale_mae)
            ) * batch_size
            total_color_mae += (
                color_mae.item() if torch.is_tensor(color_mae) else float(color_mae)
            ) * batch_size
            total_count += batch_size

    avg_loss = total_loss / max(1, total_count)
    avg_md = total_md / max(1, total_count)
    avg_ta = total_ta / max(1, total_count)
    avg_aux = total_aux / max(1, total_count)
    avg_shift_loss = total_shift_loss / max(1, total_count)
    avg_scale_loss = total_scale_loss / max(1, total_count)
    avg_color_loss = total_color_loss / max(1, total_count)
    avg_shift_mae = total_shift_mae / max(1, total_count)
    avg_scale_mae = total_scale_mae / max(1, total_count)
    avg_color_mae = total_color_mae / max(1, total_count)
    return {
        "loss": avg_loss,
        "md_loss": avg_md,
        "ta_loss": avg_ta,
        "aux_loss": avg_aux,
        "shift_loss": avg_shift_loss,
        "scale_loss": avg_scale_loss,
        "color_loss": avg_color_loss,
        "shift_mae": avg_shift_mae,
        "scale_mae": avg_scale_mae,
        "color_mae": avg_color_mae,
    }


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def _unfreeze_last_blocks(encoder: nn.Module, num_blocks: int) -> None:
    if all(hasattr(encoder, name) for name in ("time_encoder", "freq_encoder", "tf_encoder")):
        _unfreeze_last_blocks(encoder.time_encoder, num_blocks)
        _unfreeze_last_blocks(encoder.freq_encoder, num_blocks)
        _unfreeze_last_blocks(encoder.tf_encoder, num_blocks)
        return
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        _unfreeze_module(encoder)
        return
    total_blocks = len(blocks)
    if num_blocks <= 0:
        return
    if num_blocks >= total_blocks:
        _unfreeze_module(encoder)
        return
    for block in list(blocks)[-num_blocks:]:
        _unfreeze_module(block)
    proj = getattr(encoder, "proj", None)
    if proj is not None:
        _unfreeze_module(proj)


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _stratified_split(labels: torch.Tensor, val_split: float, seed: int):
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1) for stratified split.")
    gen = torch.Generator().manual_seed(seed)
    labels = labels.cpu()
    train_indices = []
    val_indices = []
    for cls in torch.unique(labels).tolist():
        idx = (labels == cls).nonzero(as_tuple=False).view(-1).tolist()
        if len(idx) <= 1:
            train_indices.extend(idx)
            continue
        perm = torch.randperm(len(idx), generator=gen).tolist()
        idx = [idx[i] for i in perm]
        val_n = max(1, int(round(len(idx) * val_split)))
        val_n = min(val_n, len(idx) - 1)
        val_indices.extend(idx[:val_n])
        train_indices.extend(idx[val_n:])
    if train_indices:
        train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=gen)].tolist()
    if val_indices:
        val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices), generator=gen)].tolist()
    return train_indices, val_indices


def _domain_stratified_split(labels: torch.Tensor, domain_ids: torch.Tensor, val_split: float, seed: int):
    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be in (0, 1) for domain-stratified split.")
    labels = labels.cpu()
    domain_ids = domain_ids.cpu()
    if labels.shape[0] != domain_ids.shape[0]:
        raise ValueError("labels and domain_ids must have the same length for domain-stratified split.")

    gen = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    for domain in torch.unique(domain_ids).tolist():
        domain_mask = domain_ids == domain
        labels_in_domain = labels[domain_mask]
        classes = torch.unique(labels_in_domain).tolist()
        for cls in classes:
            idx = ((domain_ids == domain) & (labels == cls)).nonzero(as_tuple=False).view(-1).tolist()
            if len(idx) <= 1:
                train_indices.extend(idx)
                continue
            perm = torch.randperm(len(idx), generator=gen).tolist()
            idx = [idx[i] for i in perm]
            val_n = max(1, int(round(len(idx) * val_split)))
            val_n = min(val_n, len(idx) - 1)
            val_indices.extend(idx[:val_n])
            train_indices.extend(idx[val_n:])

    if not val_indices:
        return _stratified_split(labels, val_split, seed)
    if train_indices:
        train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=gen)].tolist()
    if val_indices:
        val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices), generator=gen)].tolist()
    return train_indices, val_indices


def _labels_from_dataset(dataset, base_labels: torch.Tensor) -> torch.Tensor:
    if isinstance(dataset, torch.utils.data.Subset):
        idx = torch.tensor(dataset.indices, dtype=torch.long)
        return base_labels[idx]
    return base_labels


def _balanced_class_weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(dtype=torch.float32).clamp_min(1.0)
    num_classes = float(counts.numel())
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights / weights.mean().clamp_min(1e-12)


def _build_logit_adjustment_from_counts(counts: torch.Tensor, tau: float) -> torch.Tensor:
    counts = counts.to(dtype=torch.float32).clamp_min(1.0)
    priors = counts / counts.sum().clamp_min(1.0)
    return -tau * priors.log()


def _save_config(save_dir: Optional[Path], run_name: str, config: Dict[str, object]) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / f"{run_name}_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=True)


def _train_once(args) -> None:
    _set_seed(args.seed, args.deterministic)
    has_pretrain_checkpoint = bool(args.pretrain_checkpoint.strip())

    if args.finetune_epochs is None:
        args.finetune_epochs = args.epochs

    if args.lambda_md is None:
        args.lambda_md = args.lambda_tf

    if args.freeze_encoder is None:
        if args.no_freeze_encoder:
            args.freeze_encoder = False
        else:
            args.freeze_encoder = (args.pretrain_epochs > 0) or has_pretrain_checkpoint
    elif args.freeze_encoder:
        args.freeze_encoder = True
    elif args.no_freeze_encoder:
        args.freeze_encoder = False

    if args.pin_memory is None:
        args.pin_memory = str(args.device).lower().startswith("cuda")
    if args.persistent_workers is None:
        args.persistent_workers = args.num_workers > 0
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1.")
    if args.eval_num_workers < 0:
        raise ValueError("--eval-num-workers must be >= 0.")
    if args.loss_type == "focal" and args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be >= 0.")
    if args.logit_adjust_tau < 0.0:
        raise ValueError("--logit-adjust-tau must be >= 0.")
    if args.gate_hidden_dim < 1:
        raise ValueError("--gate-hidden-dim must be >= 1.")
    if not (0.0 <= args.gate_dropout < 1.0):
        raise ValueError("--gate-dropout must be in [0, 1).")
    if args.gate_temperature <= 0.0:
        raise ValueError("--gate-temperature must be > 0.")
    if args.mixup_alpha < 0.0:
        raise ValueError("--mixup-alpha must be >= 0.")
    if not (0.0 <= args.mixup_prob <= 1.0):
        raise ValueError("--mixup-prob must be in [0, 1].")
    if args.dg_lambda < 0.0:
        raise ValueError("--dg-lambda must be >= 0.")
    if args.dg_min_group_size < 1:
        raise ValueError("--dg-min-group-size must be >= 1.")
    if args.dg_method != "erm" and args.mixup_prob > 0.0:
        raise ValueError("Mixup is not supported for --dg-method irm/rex. Set --mixup-prob 0.")
    if args.num_workers > 0 and os.name == "nt":
        print(
            "windows_dataloader_note=1 "
            "if you hit MemoryError/EOFError with num_workers>0, rerun with --num-workers 0."
        )
    if args.eval_num_workers > 0 and os.name == "nt":
        print(
            "windows_eval_loader_note=1 "
            "FaceDetection can be memory-heavy with eval workers; prefer --eval-num-workers 0."
        )

    save_dir = Path(args.save_dir) if args.save_dir else None
    run_name = args.run_name.strip() or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pretrain_checkpoint_path = Path(args.pretrain_checkpoint).expanduser() if args.pretrain_checkpoint.strip() else None
    if pretrain_checkpoint_path is not None and not pretrain_checkpoint_path.exists():
        raise FileNotFoundError(f"--pretrain-checkpoint not found: {pretrain_checkpoint_path}")
    need_pretrain_encoder = (args.pretrain_epochs > 0) or (pretrain_checkpoint_path is not None)

    pretrain_shift_bins = _parse_float_list(args.pretrain_shift_bins)
    pretrain_scale_ratios = _parse_float_list(args.pretrain_scale_ratios)
    pretrain_color_max_gain_db_levels = _parse_float_list(args.pretrain_color_max_gain_db_levels)
    pretrain_color_active_bands = _parse_int_list(args.pretrain_color_active_bands)
    pretrain_source_domain_ids = _parse_int_list(args.pretrain_source_domain_ids)
    if not pretrain_shift_bins:
        raise ValueError("--pretrain-shift-bins must contain at least one value.")
    if not pretrain_scale_ratios:
        raise ValueError("--pretrain-scale-ratios must contain at least one value.")
    if not pretrain_color_max_gain_db_levels:
        pretrain_color_max_gain_db_levels = None
    if not pretrain_color_active_bands:
        pretrain_color_active_bands = None
    if not pretrain_source_domain_ids:
        pretrain_source_domain_ids = None

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
        shift_mode=args.pretrain_shift_mode,
        shift_bins=pretrain_shift_bins,
        scale_ratios=pretrain_scale_ratios,
        color_bands=args.pretrain_color_bands,
        color_max_gain_db=args.pretrain_color_max_gain_db,
        color_max_gain_db_levels=pretrain_color_max_gain_db_levels,
        color_active_bands=pretrain_color_active_bands,
    )
    use_triview_supervised = args.supervised_views == "triview"
    use_timefreq_supervised = args.supervised_views == "timefreq"
    use_multiview_supervised = use_triview_supervised or use_timefreq_supervised

    train_full_clean = UEATimeSeriesDataset(
        args.dataset,
        split="train",
        pad_to_max=args.pad_to_max,
        return_freq=use_multiview_supervised,
        view_config=view_config,
        use_cache=args.dataset_cache,
    )
    has_real_train_domains = getattr(train_full_clean, "domain_ids", None) is not None
    if args.dg_method != "erm":
        if not args.dg_train_with_transforms and not has_real_train_domains:
            args.dg_train_with_transforms = True
            print(
                "dg_train_with_transforms_auto=1 "
                f"dg_method={args.dg_method} "
                "reason=no_real_domain_ids_in_dataset"
            )
        elif not args.dg_train_with_transforms and has_real_train_domains:
            print(
                "dg_real_domains_enabled=1 "
                f"dg_method={args.dg_method} "
                "using dataset-provided domain_id for IRM/REx."
            )
    train_full = train_full_clean
    if args.dg_train_with_transforms:
        train_full = UEAPretrainDataset(
            args.dataset,
            split="train",
            pad_to_max=args.pad_to_max,
            view_config=view_config,
            base_seed=args.seed,
            source_domain_ids=pretrain_source_domain_ids,
            use_cache=args.dataset_cache,
        )
    max_train_length = int(train_full_clean.lengths.max().item()) if len(train_full_clean) > 0 else 0
    if (use_multiview_supervised or need_pretrain_encoder) and args.n_fft > max_train_length > 0:
        print(
            "stft_nfft_note=1 "
            f"dataset={args.dataset} n_fft={args.n_fft} max_train_length={max_train_length} "
            "consider smaller --n-fft/--hop-length for short sequences."
        )
    test_ds = UEATimeSeriesDataset(
        args.dataset,
        split="test",
        pad_to_max=args.pad_to_max,
        return_freq=use_multiview_supervised,
        view_config=view_config,
        use_cache=args.dataset_cache,
    )

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val-split must be in [0, 1).")
    if len(train_full_clean) < 300 and args.val_split >= 0.2:
        print(
            "small_trainset_valsplit_note=1 "
            f"dataset={args.dataset} train_len={len(train_full_clean)} val_split={args.val_split} "
            "consider --val-split 0.0 or 0.1 for tiny datasets."
        )
    train_label_source = train_full_clean.labels
    train_domain_source = getattr(train_full_clean, "domain_ids", None)
    resolved_val_split_mode = "none"
    if args.val_split > 0.0:
        resolved_val_split_mode = args.val_split_mode
        if resolved_val_split_mode == "auto":
            resolved_val_split_mode = "domain_stratified" if train_domain_source is not None else "label_stratified"
        if resolved_val_split_mode == "domain_stratified":
            if train_domain_source is None:
                print("val_split_mode_fallback=label_stratified reason=no_domain_ids")
                resolved_val_split_mode = "label_stratified"
                train_indices, val_indices = _stratified_split(train_label_source, args.val_split, args.seed)
            else:
                train_indices, val_indices = _domain_stratified_split(
                    train_label_source, train_domain_source, args.val_split, args.seed
                )
        else:
            resolved_val_split_mode = "label_stratified"
            train_indices, val_indices = _stratified_split(train_label_source, args.val_split, args.seed)
        print(f"val_split_mode_resolved={resolved_val_split_mode}")
        train_ds = torch.utils.data.Subset(train_full, train_indices)
        val_ds = torch.utils.data.Subset(train_full_clean, val_indices)
    else:
        train_ds = train_full
        val_ds = None

    input_dim = train_full_clean.data[0].shape[0]
    num_classes = len(train_full_clean.class_labels)
    train_labels = _labels_from_dataset(train_ds, train_label_source).to(dtype=torch.long).cpu()
    class_counts = torch.bincount(train_labels, minlength=num_classes).to(dtype=torch.long)
    class_weights = None
    if args.class_weight_mode == "balanced":
        class_weights = _balanced_class_weights_from_counts(class_counts)
    logit_adjustment = None
    if args.logit_adjustment == "train_prior":
        logit_adjustment = _build_logit_adjustment_from_counts(class_counts, tau=args.logit_adjust_tau)
    train_sampler = None
    if args.train_sampler == "balanced":
        sampler_class_weights = class_weights if class_weights is not None else _balanced_class_weights_from_counts(class_counts)
        sample_weights = sampler_class_weights[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights.to(dtype=torch.double),
            num_samples=int(train_labels.numel()),
            replacement=True,
        )
    if (
        args.class_weight_mode != "none"
        or args.train_sampler != "none"
        or args.logit_adjustment != "none"
        or args.loss_type != "ce"
    ):
        weights_str = (
            ",".join(f"{float(v):.4f}" for v in class_weights.tolist())
            if class_weights is not None
            else "none"
        )
        counts_str = ",".join(str(int(v)) for v in class_counts.tolist())
        logit_adjust_str = (
            ",".join(f"{float(v):.4f}" for v in logit_adjustment.tolist())
            if logit_adjustment is not None
            else "none"
        )
        print(
            f"class_balance_info class_counts=[{counts_str}] "
            f"class_weight_mode={args.class_weight_mode} "
            f"train_sampler={args.train_sampler} "
            f"class_weights=[{weights_str}] "
            f"loss_type={args.loss_type} focal_gamma={args.focal_gamma} "
            f"mixup_alpha={args.mixup_alpha} mixup_prob={args.mixup_prob} "
            f"dg_method={args.dg_method} dg_lambda={args.dg_lambda} "
            f"dg_min_group_size={args.dg_min_group_size} "
            f"logit_adjustment={args.logit_adjustment} logit_adjust_tau={args.logit_adjust_tau} "
            f"logit_adjust_on_eval={int(args.logit_adjust_on_eval)} "
            f"logit_bias=[{logit_adjust_str}]"
        )
    train_logit_adjustment = logit_adjustment
    eval_logit_adjustment = logit_adjustment if args.logit_adjust_on_eval else None

    pretrain_model = None
    if need_pretrain_encoder:
        pretrain_ds = UEAPretrainDataset(
            args.dataset,
            split="train",
            pad_to_max=args.pad_to_max,
            view_config=view_config,
            source_domain_ids=pretrain_source_domain_ids,
            use_cache=args.dataset_cache,
        )
        pretrain_sample = pretrain_ds[0]
        input_dim_freq = pretrain_sample["x_freq"].shape[0] if pretrain_sample["x_freq"].dim() > 1 else 1
        input_dim_tf = pretrain_sample["x_tf"].shape[0] if pretrain_sample["x_tf"].dim() > 1 else 1
        pretrain_model = MultiViewModel(
            input_dim_time=input_dim,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            hidden_dim=args.hidden_dim,
            output_dim=args.embed_dim,
            num_heads=args.num_heads,
            res_blocks=args.res_blocks,
            backbone=args.backbone,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_temporal_attn=args.use_temporal_attn,
            use_shared_qk_attn=args.use_shared_qk_attn,
            shared_qk_heads=args.shared_qk_heads,
            shared_qk_dropout=args.shared_qk_dropout,
            fuse_dropout=args.fuse_dropout,
        ).to(args.device)
        if pretrain_checkpoint_path is not None:
            loaded = torch.load(pretrain_checkpoint_path, map_location=args.device)
            ckpt_cfg = loaded.get("config", {}) if isinstance(loaded, dict) else {}
            if isinstance(ckpt_cfg, dict):
                mismatch = _check_pretrain_compatibility(args, ckpt_cfg)
                if mismatch:
                    raise ValueError(
                        "Pretrain checkpoint architecture mismatch. "
                        "Align model flags with checkpoint or use a matching checkpoint. "
                        f"Details: {mismatch}"
                    )
            state = loaded["model_state"] if isinstance(loaded, dict) and "model_state" in loaded else loaded
            pretrain_model.load_state_dict(state, strict=True)
            print(f"loaded_pretrain_checkpoint={pretrain_checkpoint_path}")

        if args.pretrain_epochs > 0:
            pretrain_loader = build_dataloader(
                pretrain_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                for_pretrain=True,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
                prefetch_factor=args.prefetch_factor,
            )
            color_dim = view_config.color_bands
            aux_heads = {
                "shift": TransformPredictor(args.embed_dim, 1).to(args.device),
                "scale": TransformPredictor(args.embed_dim, 1).to(args.device),
                "color": TransformPredictor(args.embed_dim, color_dim).to(args.device),
            }
            pretrain_optimizer = torch.optim.AdamW(
                list(pretrain_model.parameters())
                + list(aux_heads["shift"].parameters())
                + list(aux_heads["scale"].parameters())
                + list(aux_heads["color"].parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            pretrain_scheduler = None
            if args.use_cosine:
                pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    pretrain_optimizer,
                    T_0=args.cosine_t0,
                    T_mult=args.cosine_t_mult,
                    eta_min=args.cosine_eta_min,
                )

            for epoch in range(1, args.pretrain_epochs + 1):
                ramp = _linear_ramp(epoch, args.aux_ramp_start, args.aux_ramp_epochs)
                pre_stats = run_pretrain_epoch(
                    pretrain_model,
                    pretrain_loader,
                    optimizer=pretrain_optimizer,
                    device=args.device,
                    ta_mode=args.ta_mode,
                    ta_pair_mode=args.ta_pair_mode,
                    ta_shuffle_pairs=args.ta_shuffle_pairs,
                    lambda_md=args.lambda_md * ramp,
                    lambda_ta=args.lambda_ta * ramp,
                    lambda_shift=args.lambda_shift * ramp,
                    lambda_scale=args.lambda_scale * ramp,
                    lambda_color=args.lambda_color * ramp,
                    lambda_attn=args.lambda_attn * ramp,
                    aux_heads=aux_heads,
                )
                if pretrain_scheduler is not None:
                    pretrain_scheduler.step(epoch)
                print(
                    f"pretrain_epoch={epoch} "
                    f"loss={pre_stats['loss']:.4f} md_loss={pre_stats['md_loss']:.4f} "
                    f"ta_loss={pre_stats['ta_loss']:.4f} aux_loss={pre_stats['aux_loss']:.4f} "
                    f"shift_loss={pre_stats['shift_loss']:.4f} scale_loss={pre_stats['scale_loss']:.4f} "
                    f"color_loss={pre_stats['color_loss']:.4f} "
                    f"shift_mae={pre_stats['shift_mae']:.4f} scale_mae={pre_stats['scale_mae']:.4f} "
                    f"color_mae={pre_stats['color_mae']:.4f}"
                )
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                saved_pretrain_path = save_dir / f"{run_name}_pretrain_last.pt"
                torch.save(
                    {
                        "epoch": args.pretrain_epochs,
                        "model_state": pretrain_model.state_dict(),
                        "aux_heads_state": {key: head.state_dict() for key, head in aux_heads.items()},
                        "config": {
                            **vars(args),
                            "run_name": run_name,
                            "input_dim_time": input_dim,
                            "input_dim_freq": input_dim_freq,
                            "input_dim_tf": input_dim_tf,
                            "num_classes": num_classes,
                            "resolved_pretrain_shift_bins": pretrain_shift_bins,
                            "resolved_pretrain_scale_ratios": pretrain_scale_ratios,
                            "resolved_pretrain_color_max_gain_db_levels": view_config.color_max_gain_db_levels,
                            "resolved_pretrain_color_active_bands": pretrain_color_active_bands,
                            "resolved_pretrain_source_domain_ids": pretrain_source_domain_ids,
                            "resolved_pretrain_total_domains": pretrain_ds.total_domains,
                            "pretrain_checkpoint_type": "multiview_pretrain",
                        },
                    },
                    saved_pretrain_path,
                )
                print(f"saved_pretrain_checkpoint={saved_pretrain_path}")

    if use_triview_supervised:
        train_sample = train_full[0]
        input_dim_freq = train_sample["x_freq"].shape[0] if train_sample["x_freq"].dim() > 1 else 1
        input_dim_tf = train_sample["x_tf"].shape[0] if train_sample["x_tf"].dim() > 1 else 1
        model = UEATriViewClassifier(
            input_dim_time=input_dim,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            res_blocks=args.res_blocks,
            backbone=args.backbone,
            use_temporal_attn=args.use_temporal_attn,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_shared_qk_attn=args.use_shared_qk_attn,
            shared_qk_heads=args.shared_qk_heads,
            shared_qk_dropout=args.shared_qk_dropout,
            triview_fusion=args.triview_fusion,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_dropout=args.gate_dropout,
            gate_temperature=args.gate_temperature,
            fuse_dropout=args.fuse_dropout,
            head_dropout=args.head_dropout,
        ).to(args.device)
    elif use_timefreq_supervised:
        train_sample = train_full[0]
        input_dim_freq = train_sample["x_freq"].shape[0] if train_sample["x_freq"].dim() > 1 else 1
        model = UEAFreqViewClassifier(
            input_dim_time=input_dim,
            input_dim_freq=input_dim_freq,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            res_blocks=args.res_blocks,
            backbone=args.backbone,
            use_temporal_attn=args.use_temporal_attn,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_shared_qk_attn=args.use_shared_qk_attn,
            shared_qk_heads=args.shared_qk_heads,
            shared_qk_dropout=args.shared_qk_dropout,
            triview_fusion=args.triview_fusion,
            gate_hidden_dim=args.gate_hidden_dim,
            gate_dropout=args.gate_dropout,
            gate_temperature=args.gate_temperature,
            fuse_dropout=args.fuse_dropout,
            head_dropout=args.head_dropout,
        ).to(args.device)
    else:
        model = UEAClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            res_blocks=args.res_blocks,
            backbone=args.backbone,
            use_temporal_attn=args.use_temporal_attn,
            use_se=args.use_se,
            se_reduction=args.se_reduction,
            use_shared_qk_attn=args.use_shared_qk_attn,
            shared_qk_heads=args.shared_qk_heads,
            shared_qk_dropout=args.shared_qk_dropout,
            fuse_dropout=args.fuse_dropout,
            head_dropout=args.head_dropout,
        ).to(args.device)

    if pretrain_model is not None:
        if use_triview_supervised:
            model.encoder.time_encoder.load_state_dict(pretrain_model.time_encoder.state_dict())
            model.encoder.freq_encoder.load_state_dict(pretrain_model.freq_encoder.state_dict())
            model.encoder.tf_encoder.load_state_dict(pretrain_model.tf_encoder.state_dict())
        elif use_timefreq_supervised:
            model.encoder.time_encoder.load_state_dict(pretrain_model.time_encoder.state_dict())
            model.encoder.freq_encoder.load_state_dict(pretrain_model.freq_encoder.state_dict())
        else:
            model.encoder.load_state_dict(pretrain_model.time_encoder.state_dict())

    if args.eval_only or args.eval_checkpoint:
        checkpoint_path = Path(args.eval_checkpoint) if args.eval_checkpoint else None
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("eval-only requires --eval-checkpoint pointing to a valid checkpoint.")
        load_checkpoint(model, checkpoint_path, device=args.device, strict=True)
        test_loader = build_dataloader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            for_pretrain=False,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
        test_loss, test_acc, test_mf1 = evaluate(
            model=model,
            loader=test_loader,
            device=args.device,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            supervised_views=args.supervised_views,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            logit_adjustment=eval_logit_adjustment,
        )
        print(f"eval_only=1 test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_mf1={test_mf1:.4f}")
        return

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        for_pretrain=False,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        sampler=train_sampler,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            for_pretrain=False,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
    test_loader = build_dataloader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        for_pretrain=False,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor,
    )

    config = vars(args).copy()
    config.update(
        {
            "input_dim": input_dim,
            "supervised_views": args.supervised_views,
            "num_classes": num_classes,
            "train_len": len(train_ds),
            "val_len": len(val_ds) if val_ds is not None else 0,
            "test_len": len(test_ds),
            "resolved_val_split_mode": resolved_val_split_mode,
            "train_class_counts": [int(v) for v in class_counts.tolist()],
            "resolved_class_weights": (
                [float(v) for v in class_weights.tolist()] if class_weights is not None else []
            ),
            "resolved_logit_adjustment": (
                [float(v) for v in logit_adjustment.tolist()] if logit_adjustment is not None else []
            ),
            "resolved_pretrain_shift_bins": pretrain_shift_bins,
            "resolved_pretrain_scale_ratios": pretrain_scale_ratios,
            "resolved_pretrain_color_max_gain_db_levels": view_config.color_max_gain_db_levels,
            "resolved_pretrain_color_active_bands": pretrain_color_active_bands,
            "resolved_pretrain_source_domain_ids": pretrain_source_domain_ids,
            "resolved_pretrain_total_domains": (
                len(view_config.shift_bins) * len(view_config.scale_ratios) * len(view_config.color_max_gain_db_levels)
            ),
            "resolved_pretrain_checkpoint": str(pretrain_checkpoint_path) if pretrain_checkpoint_path is not None else "",
            "resolved_dg_method": args.dg_method,
            "resolved_dg_lambda": args.dg_lambda,
            "resolved_dg_min_group_size": args.dg_min_group_size,
            "resolved_dg_train_with_transforms": bool(args.dg_train_with_transforms),
        }
    )
    _save_config(save_dir, run_name, config)

    metric_higher = args.checkpoint_metric in {"val_acc", "val_mf1"}
    best_metric = -float("inf") if metric_higher else float("inf")
    best_checkpoint_path = None
    best_checkpoints = []
    patience_left = args.patience

    def _metric_value(val_loss, val_acc, val_mf1) -> Optional[float]:
        if args.checkpoint_metric == "val_loss":
            return val_loss
        if args.checkpoint_metric == "val_acc":
            return val_acc
        return val_mf1

    def _is_better(metric: float, ref: float) -> bool:
        if metric_higher:
            return metric > ref + args.min_delta
        return metric < ref - args.min_delta

    def _maybe_save_best(epoch: int, val_loss: Optional[float], val_acc: Optional[float], val_mf1: Optional[float], optimizer_state) -> None:
        nonlocal best_metric, best_checkpoint_path, patience_left, best_checkpoints
        metric = _metric_value(val_loss, val_acc, val_mf1)
        if metric is None:
            return
        if _is_better(metric, best_metric):
            best_metric = metric
            patience_left = args.patience
        else:
            patience_left -= 1

        if save_dir is None or args.save_top_k <= 0:
            return
        save_dir.mkdir(parents=True, exist_ok=True)

        should_save = len(best_checkpoints) < args.save_top_k
        if not should_save:
            worst_metric = min(best_checkpoints, key=lambda x: x["metric"])["metric"] if metric_higher else max(
                best_checkpoints, key=lambda x: x["metric"]
            )["metric"]
            if _is_better(metric, worst_metric):
                should_save = True
        if should_save:
            metric_tag = f"{args.checkpoint_metric}={metric:.4f}"
            ckpt_path = save_dir / f"{run_name}_ep{epoch}_{metric_tag}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "metric": metric,
                    "metric_name": args.checkpoint_metric,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer_state,
                    "config": config,
                },
                ckpt_path,
            )
            best_checkpoints.append({"metric": metric, "path": ckpt_path, "epoch": epoch})
            best_checkpoints.sort(key=lambda x: x["metric"], reverse=metric_higher)
            while len(best_checkpoints) > args.save_top_k:
                removed = best_checkpoints.pop(-1)
                if removed["path"].exists():
                    removed["path"].unlink()
            best_checkpoint_path = best_checkpoints[0]["path"]

    def _log_epoch(
        tag: str,
        epoch: int,
        train_loss,
        train_acc,
        train_mf1,
        val_loss,
        val_acc,
        val_mf1,
        test_loss=None,
        test_acc=None,
        test_mf1=None,
    ):
        message = (
            f"{tag}_epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_mf1={train_mf1:.4f} "
            + (
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_mf1={val_mf1:.4f} "
                if val_loss is not None
                else ""
            )
        )
        if test_loss is not None:
            message += f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_mf1={test_mf1:.4f}"
        print(message)

    if args.freeze_epochs > 0:
        _freeze_module(model.encoder)
        _unfreeze_module(model.classifier)
        freeze_optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
        freeze_scheduler = None
        if args.use_cosine:
            freeze_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                freeze_optimizer,
                T_0=args.cosine_t0,
                T_mult=args.cosine_t_mult,
                eta_min=args.cosine_eta_min,
            )
        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc, train_mf1 = run_epoch(
                model,
                train_loader,
                optimizer=freeze_optimizer,
                device=args.device,
                label_smoothing=args.label_smoothing,
                class_weights=class_weights,
                supervised_views=args.supervised_views,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                logit_adjustment=train_logit_adjustment,
                mixup_alpha=args.mixup_alpha,
                mixup_prob=args.mixup_prob,
                dg_method=args.dg_method,
                dg_lambda=args.dg_lambda,
                dg_min_group_size=args.dg_min_group_size,
            )
            if val_loader is not None:
                val_loss, val_acc, val_mf1 = run_epoch(
                    model,
                    val_loader,
                    optimizer=None,
                    device=args.device,
                    label_smoothing=args.label_smoothing,
                    class_weights=class_weights,
                    supervised_views=args.supervised_views,
                    loss_type=args.loss_type,
                    focal_gamma=args.focal_gamma,
                    logit_adjustment=eval_logit_adjustment,
                    dg_method=args.dg_method,
                    dg_lambda=args.dg_lambda,
                    dg_min_group_size=args.dg_min_group_size,
                )
            else:
                val_loss, val_acc, val_mf1 = None, None, None
            test_loss, test_acc, test_mf1 = None, None, None
            if args.eval_test_each_epoch:
                test_loss, test_acc, test_mf1 = run_epoch(
                    model,
                    test_loader,
                    optimizer=None,
                    device=args.device,
                    label_smoothing=args.label_smoothing,
                    class_weights=class_weights,
                    supervised_views=args.supervised_views,
                    loss_type=args.loss_type,
                    focal_gamma=args.focal_gamma,
                    logit_adjustment=eval_logit_adjustment,
                    dg_method=args.dg_method,
                    dg_lambda=args.dg_lambda,
                    dg_min_group_size=args.dg_min_group_size,
                )
            _log_epoch(
                "freeze",
                epoch,
                train_loss,
                train_acc,
                train_mf1,
                val_loss,
                val_acc,
                val_mf1,
                test_loss,
                test_acc,
                test_mf1,
            )
            if freeze_scheduler is not None:
                freeze_scheduler.step(epoch)
            _maybe_save_best(epoch, val_loss, val_acc, val_mf1, freeze_optimizer.state_dict())

    patience_left = args.patience
    _unfreeze_module(model.classifier)
    if args.freeze_encoder:
        _freeze_module(model.encoder)
        _unfreeze_last_blocks(model.encoder, args.unfreeze_blocks)
    else:
        _unfreeze_module(model.encoder)
    finetune_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    finetune_scheduler = None
    if args.use_cosine:
        finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            finetune_optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_t_mult,
            eta_min=args.cosine_eta_min,
        )
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_lr = args.swa_lr if args.swa_lr is not None else args.head_lr
        swa_scheduler = SWALR(finetune_optimizer, swa_lr=swa_lr, anneal_epochs=args.swa_anneal_epochs)

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc, train_mf1 = run_epoch(
            model,
            train_loader,
            optimizer=finetune_optimizer,
            device=args.device,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            supervised_views=args.supervised_views,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            logit_adjustment=train_logit_adjustment,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            dg_method=args.dg_method,
            dg_lambda=args.dg_lambda,
            dg_min_group_size=args.dg_min_group_size,
        )
        if val_loader is not None:
            val_loss, val_acc, val_mf1 = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=args.device,
                label_smoothing=args.label_smoothing,
                class_weights=class_weights,
                supervised_views=args.supervised_views,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                logit_adjustment=eval_logit_adjustment,
                dg_method=args.dg_method,
                dg_lambda=args.dg_lambda,
                dg_min_group_size=args.dg_min_group_size,
            )
        else:
            val_loss, val_acc, val_mf1 = None, None, None
        test_loss, test_acc, test_mf1 = None, None, None
        if args.eval_test_each_epoch:
            test_loss, test_acc, test_mf1 = run_epoch(
                model,
                test_loader,
                optimizer=None,
                device=args.device,
                label_smoothing=args.label_smoothing,
                class_weights=class_weights,
                supervised_views=args.supervised_views,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                logit_adjustment=eval_logit_adjustment,
                dg_method=args.dg_method,
                dg_lambda=args.dg_lambda,
                dg_min_group_size=args.dg_min_group_size,
            )
        _log_epoch(
            "finetune",
            epoch,
            train_loss,
            train_acc,
            train_mf1,
            val_loss,
            val_acc,
            val_mf1,
            test_loss,
            test_acc,
            test_mf1,
        )

        if args.use_swa and swa_model is not None and epoch >= args.swa_start:
            swa_model.update_parameters(model)
            if swa_scheduler is not None:
                swa_scheduler.step()
        elif finetune_scheduler is not None:
            finetune_scheduler.step(epoch)

        if val_loss is not None or val_acc is not None or val_mf1 is not None:
            _maybe_save_best(epoch + args.freeze_epochs, val_loss, val_acc, val_mf1, finetune_optimizer.state_dict())
            if patience_left <= 0:
                print(f"early_stop=1 best_{args.checkpoint_metric}={best_metric:.4f} epoch={epoch}")
                break

    if best_checkpoints:
        best_checkpoint_path = best_checkpoints[0]["path"]

    if best_checkpoint_path is not None:
        checkpoint = load_checkpoint(model, best_checkpoint_path, device=args.device, strict=True)
        test_loss, test_acc, test_mf1 = evaluate(
            model=model,
            loader=test_loader,
            device=args.device,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            supervised_views=args.supervised_views,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            logit_adjustment=eval_logit_adjustment,
        )
        print(
            f"best_checkpoint_eval=1 epoch={checkpoint.get('epoch')} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_mf1={test_mf1:.4f}"
        )
    else:
        test_loss, test_acc, test_mf1 = evaluate(
            model=model,
            loader=test_loader,
            device=args.device,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            supervised_views=args.supervised_views,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            logit_adjustment=eval_logit_adjustment,
        )
        print(f"final_eval=1 test_loss={test_loss:.4f} test_acc={test_acc:.4f} test_mf1={test_mf1:.4f}")

    if args.use_swa and swa_model is not None and args.finetune_epochs >= args.swa_start:
        _update_bn_from_loader(train_loader, swa_model, device=args.device, supervised_views=args.supervised_views)
        swa_loss, swa_acc, swa_mf1 = run_epoch(
            swa_model,
            test_loader,
            optimizer=None,
            device=args.device,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
            supervised_views=args.supervised_views,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            logit_adjustment=eval_logit_adjustment,
            dg_method=args.dg_method,
            dg_lambda=args.dg_lambda,
            dg_min_group_size=args.dg_min_group_size,
        )
        print(f"swa_eval=1 test_loss={swa_loss:.4f} test_acc={swa_acc:.4f} test_mf1={swa_mf1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="UWaveGestureLibrary")
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="auto",
        choices=["auto", "none", "facedetection", "heartbeat", "handwriting", "hhar"],
        help=(
            "Auto-apply dataset-specific defaults while preserving explicit CLI overrides. "
            "'auto' matches --dataset name."
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--use-se", action="store_true", default=False)
    parser.add_argument("--se-reduction", type=int, default=16)
    parser.add_argument(
        "--backbone",
        type=str,
        default="all",
        choices=["inception", "resnet", "tfc_resnet", "inception_resattn", "timesnet", "tslanet", "all"],
    )
    parser.add_argument(
        "--supervised-views",
        type=str,
        default="time",
        choices=["time", "timefreq", "triview"],
        help="Supervised classifier inputs: time only, fused (time+freq), or fused (time+freq+tf).",
    )
    parser.add_argument(
        "--triview-fusion",
        type=str,
        default="gated",
        choices=["concat", "gated"],
        help="Fusion head used when --supervised-views triview.",
    )
    parser.add_argument(
        "--gate-hidden-dim",
        type=int,
        default=64,
        help="Hidden size for gated triview fusion MLP.",
    )
    parser.add_argument(
        "--gate-dropout",
        type=float,
        default=0.0,
        help="Dropout used inside gated triview fusion MLP.",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for gated triview fusion.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default="",
        help=(
            "Optional multiview pretrain checkpoint (*.pt) used to initialize encoder. "
            "Useful for continuing finetune after pretrain finished."
        ),
    )
    parser.add_argument("--ta-mode", type=str, default="vicreg", choices=["vicreg", "infonce"])
    parser.add_argument(
        "--ta-pair-mode",
        type=str,
        default="full",
        choices=["full", "same_domain", "plain_cfc"],
        help=(
            "TA pairing strategy: "
            "full=transform-aware cross-domain pairs (default), "
            "same_domain=only within-domain transformed/base pairs, "
            "plain_cfc=non-transform-aware clean cross-frequency pairs."
        ),
    )
    parser.add_argument(
        "--ta-shuffle-pairs",
        action="store_true",
        default=False,
        help="Shuffle warped samples in TA pairing to remove per-sample theta conditioning.",
    )
    parser.add_argument("--lambda-tf", type=float, default=1.0)
    parser.add_argument("--lambda-md", type=float, default=None)
    parser.add_argument("--lambda-ta", type=float, default=1.0)
    parser.add_argument("--lambda-shift", type=float, default=1.0)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--lambda-color", type=float, default=1.0)
    parser.add_argument("--lambda-attn", type=float, default=1.0)
    parser.add_argument("--aux-ramp-epochs", type=int, default=0)
    parser.add_argument("--aux-ramp-start", type=int, default=0)
    parser.add_argument("--use-shared-qk-attn", action="store_true", default=False)
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
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
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="per_sample_channel",
        choices=["per_sample_channel", "none"],
        help="Normalization convention shared by train/eval/visualization.",
    )
    parser.add_argument(
        "--pretrain-shift-bins",
        type=str,
        default="3,-3",
        help="Seen shift bins used to sample transform family during pretraining.",
    )
    parser.add_argument(
        "--pretrain-shift-mode",
        type=str,
        default="border",
        choices=["zero", "circular", "border", "reflect"],
        help="Shift implementation used in pretraining transform family.",
    )
    parser.add_argument(
        "--pretrain-scale-ratios",
        type=str,
        default="0.9,1.1",
        help="Seen scale ratios used to sample transform family during pretraining.",
    )
    parser.add_argument(
        "--pretrain-color-bands",
        type=int,
        default=8,
        help="Number of piecewise linear color bands for transform family.",
    )
    parser.add_argument(
        "--pretrain-color-max-gain-db",
        type=float,
        default=6.0,
        help="Max absolute dB gain used in color transform family sampling.",
    )
    parser.add_argument(
        "--pretrain-color-max-gain-db-levels",
        type=str,
        default="",
        help=(
            "Optional ordered color severity levels (comma-separated max dB, e.g. '3,6,9'). "
            "If empty, falls back to --pretrain-color-max-gain-db."
        ),
    )
    parser.add_argument(
        "--pretrain-color-active-bands",
        type=str,
        default="",
        help="Optional seen color band indices (e.g. '0,1,2,3'). Empty means all bands.",
    )
    parser.add_argument(
        "--pretrain-source-domain-ids",
        type=str,
        default="",
        help=(
            "Optional source domain ids for DG pretraining (comma-separated). "
            "Domain id is defined over grid order (shift, scale, color-level). Empty means all domains."
        ),
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--val-split-mode",
        type=str,
        default="auto",
        choices=["auto", "label_stratified", "domain_stratified"],
        help=(
            "Validation split protocol. "
            "auto=domain_stratified when dataset exposes domain_id, else label_stratified."
        ),
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-metric", type=str, default="val_mf1", choices=["val_mf1", "val_acc", "val_loss"])
    parser.add_argument("--save-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--save-dir", type=str, default="time-main/checkpoints")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--freeze-epochs", type=int, default=0)
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=None,
        help="Freeze encoder during finetune (default: True if pretrain_epochs>0, else False).",
    )
    freeze_group.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        default=None,
        help="Do not freeze encoder during finetune.",
    )
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--weight-decay-grid", type=str, default="")
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Optional class weighting strategy for cross-entropy loss.",
    )
    parser.add_argument(
        "--train-sampler",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Optional training sampler strategy (balanced uses WeightedRandomSampler).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Beta(alpha, alpha) mixup strength for supervised training (0 disables mixup).",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.0,
        help="Per-batch probability of applying mixup during supervised training.",
    )
    parser.add_argument(
        "--dg-method",
        type=str,
        default="erm",
        choices=["erm", "irm", "rex"],
        help=(
            "Domain-generalization objective for supervised training. "
            "erm=plain empirical risk, irm=invariant risk minimization, rex=risk extrapolation."
        ),
    )
    parser.add_argument(
        "--dg-lambda",
        type=float,
        default=1.0,
        help="Weight for DG penalty when --dg-method is irm/rex.",
    )
    parser.add_argument(
        "--dg-min-group-size",
        type=int,
        default=2,
        help="Minimum samples per domain group in a batch for IRM/REx penalty.",
    )
    parser.add_argument(
        "--dg-train-with-transforms",
        action="store_true",
        default=False,
        help=(
            "Train supervised phase on transformed-domain samples (UEAPretrainDataset). "
            "When disabled, IRM/REx will use dataset-provided domain_id if available."
        ),
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Classification loss for supervised training.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma used when --loss-type focal.",
    )
    parser.add_argument(
        "--logit-adjustment",
        type=str,
        default="none",
        choices=["none", "train_prior"],
        help="Optional logit bias; train_prior uses -tau*log(train class prior).",
    )
    parser.add_argument(
        "--logit-adjust-tau",
        type=float,
        default=1.0,
        help="Scaling tau for logit adjustment.",
    )
    logit_eval_group = parser.add_mutually_exclusive_group()
    logit_eval_group.add_argument(
        "--logit-adjust-on-eval",
        dest="logit_adjust_on_eval",
        action="store_true",
        default=True,
        help="Apply logit adjustment during validation/test inference.",
    )
    logit_eval_group.add_argument(
        "--no-logit-adjust-on-eval",
        dest="logit_adjust_on_eval",
        action="store_false",
        help="Disable logit adjustment during validation/test inference.",
    )
    parser.add_argument("--use-temporal-attn", action="store_true", default=False)
    parser.add_argument("--fuse-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--eval-test-each-epoch", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--eval-checkpoint", type=str, default="")
    parser.add_argument("--pad-to-max", action="store_true", default=True)
    parser.add_argument("--no-pad-to-max", dest="pad_to_max", action="store_false")
    parser.add_argument(
        "--dataset-cache",
        action="store_true",
        default=True,
        help="Cache parsed UEA .ts files on disk to speed up repeated experiments.",
    )
    parser.add_argument(
        "--no-dataset-cache",
        dest="dataset_cache",
        action="store_false",
        help="Disable on-disk dataset cache.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=0,
        help="DataLoader workers for val/test/eval loaders.",
    )
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=None,
        help="Enable DataLoader pin_memory.",
    )
    pin_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pin_memory.",
    )
    persistent_group = parser.add_mutually_exclusive_group()
    persistent_group.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=None,
        help="Keep DataLoader workers alive across epochs (requires num_workers>0).",
    )
    persistent_group.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches prefetched by each DataLoader worker (num_workers>0 only).",
    )
    parser.add_argument("--use-cosine", action="store_true", default=False)
    parser.add_argument("--cosine-t0", type=int, default=10)
    parser.add_argument("--cosine-t-mult", type=int, default=2)
    parser.add_argument("--cosine-eta-min", type=float, default=1e-6)
    parser.add_argument("--use-swa", action="store_true", default=False)
    parser.add_argument("--swa-start", type=int, default=10)
    parser.add_argument("--swa-lr", type=float, default=None)
    parser.add_argument("--swa-anneal-epochs", type=int, default=5)
    args = parser.parse_args()
    explicit_dests = _explicit_arg_dests(parser)
    applied_profile = _apply_dataset_profile(args, parser, explicit_dests=explicit_dests)
    if applied_profile:
        payload = " ".join(f"{key}={applied_profile[key]}" for key in sorted(applied_profile.keys()))
        print(f"dataset_profile_applied=1 profile={args.dataset_profile} {payload}")

    if args.weight_decay_grid:
        grid = _parse_float_list(args.weight_decay_grid)
        if not grid:
            _train_once(args)
            return
        base_run = args.run_name
        for wd in grid:
            run_args = copy.deepcopy(args)
            run_args.weight_decay = wd
            if base_run.strip():
                run_args.run_name = f"{base_run}_wd{wd:g}"
            else:
                run_args.run_name = f"{args.dataset}_wd{wd:g}"
            _train_once(run_args)
        return

    _train_once(args)
    return

if __name__ == "__main__":
    main()

