from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing import PreprocessConfig, build_triview_from_time


def apply_per_sample_channel(x: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if x.dim() == 2:
        return torch.stack([fn(x[i]) for i in range(x.shape[0])], dim=0)
    if x.dim() == 3:
        return torch.stack(
            [torch.stack([fn(x[i, j]) for j in range(x.shape[1])], dim=0) for i in range(x.shape[0])],
            dim=0,
        )
    raise ValueError(f"Expected batch tensor shape (B, T) or (B, C, T), got {tuple(x.shape)}.")


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    label_smoothing: float = 0.0,
    transform_fn: Optional[Callable[[torch.Tensor, int], Any]] = None,
    batch_logger: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
    supervised_views: str = "time",
    preprocess_config: Optional[PreprocessConfig] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x_time"].to(device)
            y = batch["y"].to(device)

            transform_meta = None
            if transform_fn is not None:
                transformed = transform_fn(x, batch_idx)
                if isinstance(transformed, tuple):
                    x, transform_meta = transformed
                else:
                    x = transformed

            if supervised_views == "triview":
                if transform_fn is None and ("x_freq" in batch) and ("x_tf" in batch):
                    x_freq = batch["x_freq"].to(device)
                    x_tf = batch["x_tf"].to(device)
                else:
                    if preprocess_config is None:
                        raise ValueError("preprocess_config is required when recomputing tri-view inputs.")
                    x_freq_list: List[torch.Tensor] = []
                    x_tf_list: List[torch.Tensor] = []
                    for i in range(x.shape[0]):
                        views = build_triview_from_time(x[i], preprocess_config)
                        x_freq_list.append(views["x_freq"])
                        x_tf_list.append(views["x_tf"])
                    x_freq = torch.stack(x_freq_list, dim=0).to(device)
                    x_tf = torch.stack(x_tf_list, dim=0).to(device)
                logits = model(x, x_freq, x_tf)
            elif supervised_views == "timefreq":
                if transform_fn is None and ("x_freq" in batch):
                    x_freq = batch["x_freq"].to(device)
                else:
                    if preprocess_config is None:
                        raise ValueError("preprocess_config is required when recomputing time+freq inputs.")
                    x_freq_list: List[torch.Tensor] = []
                    for i in range(x.shape[0]):
                        views = build_triview_from_time(x[i], preprocess_config)
                        x_freq_list.append(views["x_freq"])
                    x_freq = torch.stack(x_freq_list, dim=0).to(device)
                logits = model(x, x_freq)
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total_count += y.size(0)

            num_classes = logits.size(1)
            if confusion is None:
                confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
            y_cpu = y.view(-1).to(torch.long).cpu()
            preds_cpu = preds.view(-1).to(torch.long).cpu()
            idx = y_cpu * num_classes + preds_cpu
            bins = torch.bincount(idx, minlength=num_classes * num_classes)
            confusion += bins.view(num_classes, num_classes)

            if batch_logger is not None:
                batch_logger(batch_idx, int(y.size(0)), transform_meta)

    acc = total_correct / max(1, total_count)
    mf1 = 0.0
    if confusion is not None:
        conf = confusion.to(dtype=torch.float32)
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        mf1 = f1.mean().item()
    return {
        "loss": total_loss / max(1, total_count),
        "acc": acc,
        "mf1": mf1,
        "count": float(total_count),
    }


def assert_same_training_budget(configs: Sequence[Mapping[str, Any]], required_keys: Iterable[str]) -> None:
    configs = list(configs)
    if len(configs) <= 1:
        return
    baseline = configs[0]
    for idx, cfg in enumerate(configs[1:], start=1):
        for key in required_keys:
            if baseline.get(key) != cfg.get(key):
                raise ValueError(
                    f"Budget mismatch at checkpoint index {idx}: key='{key}', "
                    f"baseline={baseline.get(key)} vs current={cfg.get(key)}"
                )
