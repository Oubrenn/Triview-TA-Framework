import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_reviewer_fast_experiments import (  # noqa: E402
    CkptSpec,
    _load_model,
    _make_dataset,
    _mixed_shift_color,
    _predict_logits,
)
from train_uea import collate_fn  # noqa: E402
from train_uea import _domain_stratified_split, _stratified_split  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "Baseline": ROOT / "outputs_new/hhar6ch_runs/hhar6ch_erm_time_simple_ep18_val_mf1=0.9188.pt",
    "Tri-view": ROOT / "outputs_new/hhar6ch_runs/hhar6ch_triview_no_pretrain_ep9_val_mf1=0.9550.pt",
    "TriView-TA": ROOT / "outputs_new/hhar6ch_runs/hhar6ch_triview_ta_ep17_val_mf1=0.9416.pt",
}


def _macro_f1_from_confusion(confusion: torch.Tensor) -> float:
    conf = confusion.to(dtype=torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(f1.mean().item())


def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    y_t = torch.tensor(list(y_true), dtype=torch.long)
    y_p = torch.tensor(list(y_pred), dtype=torch.long)
    confusion = torch.bincount(y_t * num_classes + y_p, minlength=num_classes * num_classes).view(
        num_classes, num_classes
    )
    return {
        "acc": float((y_t == y_p).to(dtype=torch.float32).mean().item()),
        "mf1": _macro_f1_from_confusion(confusion),
        "count": float(y_t.numel()),
    }


@torch.no_grad()
def collect_predictions(
    loaded,
    loader: DataLoader,
    device: str,
    condition: str,
    perturb_seed: int,
    shift_bins: float,
    color_db: float,
    color_bands: int,
) -> Tuple[List[int], List[int], List[int]]:
    sample_ids: List[int] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    offset = 0

    loaded.model.eval()
    for batch_idx, batch in enumerate(loader):
        x = batch["x_time"].to(device)
        y = batch["y"].to(device)

        if condition == "pert":
            x = _mixed_shift_color(
                x,
                loaded,
                seed=int(perturb_seed) + 1009 * batch_idx,
                shift_bins=float(shift_bins),
                color_db=float(color_db),
                color_bands=int(color_bands),
            )
        elif condition != "clean":
            raise ValueError(f"Unsupported condition: {condition}")

        logits = _predict_logits(loaded, x)
        pred = logits.argmax(dim=1)

        batch_size = int(y.numel())
        sample_ids.extend(range(offset, offset + batch_size))
        y_true_all.extend(int(v) for v in y.detach().cpu().tolist())
        y_pred_all.extend(int(v) for v in pred.detach().cpu().tolist())
        offset += batch_size

    return sample_ids, y_true_all, y_pred_all


def _parse_checkpoint_overrides(raw: Sequence[str]) -> Dict[str, Path]:
    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for item in raw:
        if "=" not in item:
            raise ValueError("--checkpoint overrides must have form Label=path")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("Checkpoint label cannot be empty.")
        checkpoints[label] = Path(path).expanduser()
    return checkpoints


def _dataset_class_labels(dataset) -> List[str]:
    if hasattr(dataset, "class_labels"):
        return [str(label) for label in dataset.class_labels]
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "class_labels"):
        return [str(label) for label in dataset.dataset.class_labels]
    raise AttributeError("Dataset does not expose class_labels.")


def _make_eval_dataset(config: Dict[str, object], split: str):
    if split == "val":
        full_train = _make_dataset("HHAR", config, split="train", return_freq=False)
        val_split = float(config.get("val_split", 0.1))
        seed = int(config.get("seed", 42))
        domain_ids = getattr(full_train, "domain_ids", None)
        if domain_ids is not None:
            _, val_indices = _domain_stratified_split(full_train.labels, domain_ids, val_split, seed)
        else:
            _, val_indices = _stratified_split(full_train.labels, val_split, seed)
        return Subset(full_train, val_indices)
    return _make_dataset("HHAR", config, split=split, return_freq=False)


def export_predictions(args: argparse.Namespace) -> None:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    checkpoints = _parse_checkpoint_overrides(args.checkpoint)
    for label, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {path}")

    loaded_by_label = {}
    for label, path in checkpoints.items():
        spec = CkptSpec("HHAR", label, path)
        print(f"[Load] {label}: {path}")
        load_split = "test" if args.split == "val" else args.split
        loaded_by_label[label] = _load_model(spec, device=device, split=load_split)

    first_loaded = next(iter(loaded_by_label.values()))
    dataset = _make_eval_dataset(first_loaded.config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    rows = []
    summary_rows = []
    class_labels = _dataset_class_labels(dataset)
    num_classes = len(class_labels)
    for label in checkpoints:
        loaded = loaded_by_label[label]
        clean_ids, y_true_clean, y_pred_clean = collect_predictions(
            loaded,
            loader,
            device=device,
            condition="clean",
            perturb_seed=args.perturb_seed,
            shift_bins=args.shift_bins,
            color_db=args.color_db,
            color_bands=args.color_bands,
        )
        pert_ids, y_true_pert, y_pred_pert = collect_predictions(
            loaded,
            loader,
            device=device,
            condition="pert",
            perturb_seed=args.perturb_seed,
            shift_bins=args.shift_bins,
            color_db=args.color_db,
            color_bands=args.color_bands,
        )
        if clean_ids != pert_ids or y_true_clean != y_true_pert:
            raise RuntimeError("Clean and perturbed passes are not aligned. Check shuffle=False and dataloader order.")

        clean_metrics = _classification_metrics(y_true_clean, y_pred_clean, num_classes)
        pert_metrics = _classification_metrics(y_true_pert, y_pred_pert, num_classes)
        summary_rows.extend(
            [
                {
                    "model": label,
                    "condition": "clean",
                    "acc": clean_metrics["acc"],
                    "mf1": clean_metrics["mf1"],
                    "count": clean_metrics["count"],
                },
                {
                    "model": label,
                    "condition": "pert",
                    "acc": pert_metrics["acc"],
                    "mf1": pert_metrics["mf1"],
                    "count": pert_metrics["count"],
                },
            ]
        )
        print(
            f"[Eval] {label}: clean_acc={clean_metrics['acc']:.4f} pert_acc={pert_metrics['acc']:.4f} "
            f"clean_mf1={clean_metrics['mf1']:.4f} pert_mf1={pert_metrics['mf1']:.4f}"
        )

        for sample_id, y_true, y_pred in zip(clean_ids, y_true_clean, y_pred_clean):
            rows.append(
                {
                    "model": label,
                    "condition": "clean",
                    "sample_id": sample_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        for sample_id, y_true, y_pred in zip(pert_ids, y_true_pert, y_pred_pert):
            rows.append(
                {
                    "model": label,
                    "condition": "pert",
                    "sample_id": sample_id,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "condition", "sample_id", "y_true", "y_pred"])
        writer.writeheader()
        writer.writerows(rows)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "condition", "acc", "mf1", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "dataset": "HHAR",
        "split": args.split,
        "class_labels": class_labels,
        "condition": "retained_mixed_shift_color",
        "mixed_shift_bins": float(args.shift_bins),
        "mixed_color_db": float(args.color_db),
        "mixed_color_bands": int(args.color_bands),
        "perturb_seed": int(args.perturb_seed),
        "device": device,
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "output_csv": str(args.output_csv),
        "summary_csv": str(args.summary_csv),
        "note": "Uses the reviewer Safe-A retained mixed setting: STFT band shift followed by spectral coloring.",
    }
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[Saved] {args.output_csv}")
    print(f"[Saved] {args.summary_csv}")
    print(f"[Saved] {args.meta_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HHAR clean/perturbed predictions for confusion diagnostics.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    parser.add_argument("--perturb-seed", type=int, default=42)
    parser.add_argument("--shift-bins", type=float, default=0.25)
    parser.add_argument("--color-db", type=float, default=3.0)
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Optional override/addition in the form Label=path. Defaults to Baseline/Tri-view/TriView-TA.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/hhar_confusion_predictions.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/hhar_confusion_summary.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("outputs/hhar_confusion_predictions_meta.json"))
    return parser.parse_args()


if __name__ == "__main__":
    export_predictions(parse_args())
