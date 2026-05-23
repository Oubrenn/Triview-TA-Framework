import csv
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


def move_to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [move_to_device(value, device) for value in batch]
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def ensure_bct(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must have shape (B, C, T), got {tuple(x.shape)}.")


def flatten_feature(h: torch.Tensor) -> torch.Tensor:
    if h.dim() > 2:
        h = h.reshape(h.size(0), -1)
    return h


def pair_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    h1 = flatten_feature(h1)
    h2 = flatten_feature(h2)
    cos = F.cosine_similarity(h1, h2, dim=1, eps=1e-8)
    return (1.0 - cos.abs()).mean()


def mean_cross_view_distance(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    d_time_freq = pair_distance(features["h_time"], features["h_freq"])
    d_time_tf = pair_distance(features["h_time"], features["h_tf"])
    d_freq_tf = pair_distance(features["h_freq"], features["h_tf"])
    d_mean = (d_time_freq + d_time_tf + d_freq_tf) / 3.0
    return {
        "d_time_freq": d_time_freq,
        "d_time_tf": d_time_tf,
        "d_freq_tf": d_freq_tf,
        "d_mean": d_mean,
    }


def extract_triview_features(
    model: torch.nn.Module,
    x_time: torch.Tensor,
    x_freq: torch.Tensor,
    x_tf: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    x_time = ensure_bct(x_time, "x_time")
    x_freq = ensure_bct(x_freq, "x_freq")
    x_tf = ensure_bct(x_tf, "x_tf")

    encoder = getattr(model, "encoder", model)
    out = encoder(
        x_time=x_time,
        x_freq=x_freq,
        x_tf=x_tf,
        return_intermediates=True,
    )
    return {
        "h_time": out["h_time"],
        "h_freq": out["h_freq"],
        "h_tf": out["h_tf"],
    }


def compute_cross_view_evolution_metrics(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    method_name: str,
    epoch: int,
    perturb_fn: Optional[Callable[[torch.Tensor], Dict[str, torch.Tensor]]] = None,
    max_batches: int = 20,
) -> Dict[str, object]:
    model.eval()
    clean_sum = {key: 0.0 for key in ("d_time_freq", "d_time_tf", "d_freq_tf", "d_mean")}
    pert_sum = {key: 0.0 for key in clean_sum}
    total = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_to_device(batch, device)
            x_time = ensure_bct(batch["x_time"], "x_time")
            x_freq = ensure_bct(batch["x_freq"], "x_freq")
            x_tf = ensure_bct(batch["x_tf"], "x_tf")
            batch_size = int(x_time.size(0))

            clean = mean_cross_view_distance(extract_triview_features(model, x_time, x_freq, x_tf))
            for key, value in clean.items():
                clean_sum[key] += float(value.item()) * batch_size

            if perturb_fn is not None:
                pert_views = perturb_fn(x_time)
                pert = mean_cross_view_distance(
                    extract_triview_features(
                        model,
                        ensure_bct(pert_views["x_time"], "x_time"),
                        ensure_bct(pert_views["x_freq"], "x_freq"),
                        ensure_bct(pert_views["x_tf"], "x_tf"),
                    )
                )
            else:
                pert = clean
            for key, value in pert.items():
                pert_sum[key] += float(value.item()) * batch_size
            total += batch_size

    denom = max(total, 1)
    clean_avg = {key: value / denom for key, value in clean_sum.items()}
    pert_avg = {key: value / denom for key, value in pert_sum.items()}

    return {
        "method": method_name,
        "epoch": int(epoch),
        "n_samples": int(total),
        "clean_d_time_freq": clean_avg["d_time_freq"],
        "clean_d_time_tf": clean_avg["d_time_tf"],
        "clean_d_freq_tf": clean_avg["d_freq_tf"],
        "clean_d_mean": clean_avg["d_mean"],
        "pert_d_time_freq": pert_avg["d_time_freq"],
        "pert_d_time_tf": pert_avg["d_time_tf"],
        "pert_d_freq_tf": pert_avg["d_freq_tf"],
        "pert_d_mean": pert_avg["d_mean"],
        "cv_drift": abs(pert_avg["d_mean"] - clean_avg["d_mean"]),
    }


def write_metrics_csv(csv_path: Path, rows) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
