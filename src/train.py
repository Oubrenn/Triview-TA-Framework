import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets import SyntheticTimeSeriesDataset, ViewConfig
from losses import multi_domain_consistency_loss, ta_cfc_loss, color_regression_loss
from models import MultiViewModel, TransformPredictor


def collate_fn(batch):
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
        },
    }


def run_epoch(
    model,
    loader,
    optimizer=None,
    device="cpu",
    ta_mode="vicreg",
    lambda_md: float = 1.0,
    lambda_ta: float = 1.0,
    lambda_shift: float = 1.0,
    lambda_scale: float = 1.0,
    lambda_color: float = 1.0,
    lambda_attn: float = 1.0,
    aux_heads=None,
):
    total = 0.0
    if aux_heads is not None:
        if optimizer is None:
            for head in aux_heads.values():
                head.eval()
        else:
            for head in aux_heads.values():
                head.train()
    for batch in loader:
        x_time = batch["x_time"].to(device)
        x_freq = batch["x_freq"].to(device)
        x_tf = batch["x_tf"].to(device)
        x_shift_freq = batch["x_shift_freq"].to(device)
        x_shift_tf = batch["x_shift_tf"].to(device)
        x_scale = batch["x_scale"].to(device)
        x_scale_freq = batch["x_scale_freq"].to(device)
        x_scale_tf = batch["x_scale_tf"].to(device)
        x_color = batch["x_color"].to(device)
        x_color_freq = batch["x_color_freq"].to(device)
        x_color_tf = batch["x_color_tf"].to(device)

        if hasattr(model, "forward_with_attn"):
            (z_time, z_freq, z_tf), attn_info = model.forward_with_attn(x_time, x_freq, x_tf)
        else:
            z_time, z_freq, z_tf = model(x_time, x_freq, x_tf)
            attn_info = None
        z_scale_time, z_scale_freq, z_scale_tf = model(x_scale, x_scale_freq, x_scale_tf)
        z_color_time, z_color_freq, z_color_tf = model(x_color, x_color_freq, x_color_tf)
        _, z_shift_freq, z_shift_tf = model(None, x_shift_freq, x_shift_tf)

        loss_md = multi_domain_consistency_loss(z_time, z_freq, z_tf)
        loss_ta = (
            ta_cfc_loss(z_time, z_scale_time, mode=ta_mode)
            + ta_cfc_loss(z_freq, z_scale_freq, mode=ta_mode)
            + ta_cfc_loss(z_tf, z_scale_tf, mode=ta_mode)
            + ta_cfc_loss(z_time, z_color_time, mode=ta_mode)
            + ta_cfc_loss(z_freq, z_color_freq, mode=ta_mode)
            + ta_cfc_loss(z_tf, z_color_tf, mode=ta_mode)
            + ta_cfc_loss(z_freq, z_shift_freq, mode=ta_mode)
            + ta_cfc_loss(z_tf, z_shift_tf, mode=ta_mode)
            # Cross-domain transform consistency (shifted freq vs shifted TF).
            + ta_cfc_loss(z_shift_freq, z_shift_tf, mode=ta_mode)
            # Cross-domain alignment to time view (shifted freq/TF vs time).
            + ta_cfc_loss(z_time, z_shift_freq, mode=ta_mode)
            + ta_cfc_loss(z_time, z_shift_tf, mode=ta_mode)
        )
        loss_attn = 0.0
        if attn_info is not None:
            loss_attn = _attn_consistency_loss(
                attn_info.get("time"),
                attn_info.get("freq"),
                attn_info.get("tf"),
            )
        loss_aux = 0.0
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

            pred_scale_time = aux_heads["scale"](z_scale_time)
            pred_scale_freq = aux_heads["scale"](z_scale_freq)
            pred_scale_tf = aux_heads["scale"](z_scale_tf)
            loss_scale = (
                F.mse_loss(pred_scale_time, scale_ratio)
                + F.mse_loss(pred_scale_freq, scale_ratio)
                + F.mse_loss(pred_scale_tf, scale_ratio)
            ) / 3.0

            pred_color_time = aux_heads["color"](z_color_time)
            pred_color_freq = aux_heads["color"](z_color_freq)
            pred_color_tf = aux_heads["color"](z_color_tf)
            loss_color = (
                color_regression_loss(pred_color_time, color_gains)
                + color_regression_loss(pred_color_freq, color_gains)
                + color_regression_loss(pred_color_tf, color_gains)
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

        total += loss.item()
    return total / max(1, len(loader))


def _infer_embed_dim(model: nn.Module) -> int:
    proj = getattr(model, "time_projector", None)
    if isinstance(proj, nn.Identity):
        enc_proj = getattr(model.time_encoder, "proj", None)
        if enc_proj is not None and hasattr(enc_proj, "out_features"):
            return enc_proj.out_features
    if hasattr(proj, "net") and hasattr(proj.net[-1], "out_features"):
        return proj.net[-1].out_features
    raise ValueError("Unable to infer embedding dimension from model.")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ta-mode", type=str, default="vicreg", choices=["vicreg", "infonce"])
    parser.add_argument("--lambda-md", type=float, default=1.0)
    parser.add_argument("--lambda-ta", type=float, default=1.0)
    parser.add_argument("--lambda-shift", type=float, default=1.0)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--lambda-color", type=float, default=1.0)
    parser.add_argument("--lambda-attn", type=float, default=1.0)
    parser.add_argument("--use-temporal-attn", action="store_true", default=False)
    parser.add_argument("--use-shared-qk-attn", action="store_true", default=False)
    parser.add_argument("--shared-qk-heads", type=int, default=4)
    parser.add_argument("--shared-qk-dropout", type=float, default=0.0)
    args = parser.parse_args()

    view_config = ViewConfig()
    dataset = SyntheticTimeSeriesDataset(num_samples=200, length=1024, view_config=view_config)
    train_len = int(0.7 * len(dataset))
    val_len = int(0.15 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample = dataset[0]
    x_time = sample["x_time"]
    x_freq = sample["x_freq"]
    x_tf = sample["x_tf"]
    input_dim_time = 1 if x_time.dim() == 1 else x_time.shape[0]
    input_dim_freq = x_freq.shape[0] if x_freq.dim() > 1 else 1
    input_dim_tf = x_tf.shape[0] if x_tf.dim() > 1 else 1
    model = MultiViewModel(
        input_dim_time=input_dim_time,
        input_dim_freq=input_dim_freq,
        input_dim_tf=input_dim_tf,
        use_temporal_attn=args.use_temporal_attn,
        use_shared_qk_attn=args.use_shared_qk_attn,
        shared_qk_heads=args.shared_qk_heads,
        shared_qk_dropout=args.shared_qk_dropout,
    ).to(args.device)
    embed_dim = _infer_embed_dim(model)
    color_dim = view_config.color_bands
    aux_heads = {
        "shift": TransformPredictor(embed_dim, 1).to(args.device),
        "scale": TransformPredictor(embed_dim, 1).to(args.device),
        "color": TransformPredictor(embed_dim, color_dim).to(args.device),
    }
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(aux_heads["shift"].parameters())
        + list(aux_heads["scale"].parameters())
        + list(aux_heads["color"].parameters()),
        lr=1e-3,
    )

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            ta_mode=args.ta_mode,
            lambda_md=args.lambda_md,
            lambda_ta=args.lambda_ta,
            lambda_shift=args.lambda_shift,
            lambda_scale=args.lambda_scale,
            lambda_color=args.lambda_color,
            lambda_attn=args.lambda_attn,
            aux_heads=aux_heads,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            device=args.device,
            ta_mode=args.ta_mode,
            lambda_md=args.lambda_md,
            lambda_ta=args.lambda_ta,
            lambda_shift=args.lambda_shift,
            lambda_scale=args.lambda_scale,
            lambda_color=args.lambda_color,
            lambda_attn=args.lambda_attn,
            aux_heads=aux_heads,
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    test_loss = run_epoch(
        model,
        test_loader,
        device=args.device,
        ta_mode=args.ta_mode,
        lambda_md=args.lambda_md,
        lambda_ta=args.lambda_ta,
        lambda_shift=args.lambda_shift,
        lambda_scale=args.lambda_scale,
        lambda_color=args.lambda_color,
        lambda_attn=args.lambda_attn,
        aux_heads=aux_heads,
    )
    print(f"test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
