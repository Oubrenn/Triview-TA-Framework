import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


MODEL_ORDER = ["Baseline", "Tri-view", "TriView-TA"]
MODEL_COLORS = {
    "Baseline": "#4C78A8",
    "Tri-view": "#F58518",
    "TriView-TA": "#54A24B",
}


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_class_names(meta_json: Path, n_classes: int) -> List[str]:
    if meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        labels = meta.get("class_labels")
        if isinstance(labels, list) and len(labels) == n_classes:
            return [str(label) for label in labels]
    return [str(i) for i in range(n_classes)]


def _confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < n_classes and 0 <= yp < n_classes:
            cm[yt, yp] += 1
    return cm


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)


def _per_class_recall(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    cm = _confusion_matrix(y_true, y_pred, n_classes)
    denom = cm.sum(axis=1).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)
    return np.divide(diag, denom, out=np.zeros_like(diag), where=denom != 0)


def _series(rows: Sequence[Dict[str, str]], model: str, condition: str) -> Tuple[List[int], List[int]]:
    selected = [row for row in rows if row["model"] == model and row["condition"] == condition]
    selected.sort(key=lambda row: int(row["sample_id"]))
    if not selected:
        raise ValueError(f"No rows for model={model!r}, condition={condition!r}")
    y_true = [int(row["y_true"]) for row in selected]
    y_pred = [int(row["y_pred"]) for row in selected]
    return y_true, y_pred


def _accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p)) if y_t.size else 0.0


def plot_hhar_confusion(args: argparse.Namespace) -> None:
    rows = _read_rows(args.csv_path)
    all_labels = [int(row["y_true"]) for row in rows] + [int(row["y_pred"]) for row in rows]
    n_classes = max(all_labels) + 1
    class_names = _load_class_names(args.meta_json, n_classes)

    model_order = [model for model in MODEL_ORDER if any(row["model"] == model for row in rows)]
    if not model_order:
        raise ValueError("No expected models found in prediction CSV.")

    fig, axes = plt.subplots(
        1,
        len(model_order) + 1,
        figsize=(18, 4.8),
        gridspec_kw={"width_ratios": [1.0] * len(model_order) + [1.35]},
        constrained_layout=True,
    )

    delta_recalls: Dict[str, np.ndarray] = {}
    metric_rows: List[Dict[str, object]] = []
    image = None
    for ax_idx, model_name in enumerate(model_order):
        y_true_clean, y_pred_clean = _series(rows, model_name, "clean")
        y_true_pert, y_pred_pert = _series(rows, model_name, "pert")
        if y_true_clean != y_true_pert:
            raise ValueError(f"Clean/pert true-label order mismatch for {model_name}.")

        cm_pert = _confusion_matrix(y_true_pert, y_pred_pert, n_classes)
        cm_pert_norm = _row_normalize(cm_pert)
        recall_clean = _per_class_recall(y_true_clean, y_pred_clean, n_classes)
        recall_pert = _per_class_recall(y_true_pert, y_pred_pert, n_classes)
        delta_recalls[model_name] = recall_pert - recall_clean

        metric_rows.append(
            {
                "model": model_name,
                "clean_acc": _accuracy(y_true_clean, y_pred_clean),
                "pert_acc": _accuracy(y_true_pert, y_pred_pert),
                "mean_delta_recall": float(delta_recalls[model_name].mean()),
                "min_delta_recall": float(delta_recalls[model_name].min()),
            }
        )

        ax = axes[ax_idx]
        image = ax.imshow(cm_pert_norm, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Predicted label")
        if ax_idx == 0:
            ax.set_ylabel("True label")

        ticks = np.arange(n_classes)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

        for i in range(n_classes):
            for j in range(n_classes):
                value = cm_pert_norm[i, j]
                color = "white" if value >= 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    if image is not None:
        cbar = fig.colorbar(image, ax=list(axes[: len(model_order)]), shrink=0.82)
        cbar.set_label("Row-normalized frequency")

    ax_bar = axes[-1]
    x = np.arange(n_classes)
    width = min(0.25, 0.8 / max(1, len(model_order)))
    center = (len(model_order) - 1) / 2.0
    for idx, model_name in enumerate(model_order):
        ax_bar.bar(
            x + (idx - center) * width,
            -delta_recalls[model_name],
            width=width,
            label=model_name,
            color=MODEL_COLORS.get(model_name),
        )

    ax_bar.axhline(0.0, color="#1f1f1f", linewidth=1.0)
    ax_bar.set_title("Per-class recall drop", fontsize=11)
    ax_bar.set_ylabel(r"$R_k^{clean} - R_k^{pert}$")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax_bar.legend(fontsize=8, frameon=False)
    ax_bar.grid(axis="y", alpha=0.2, linewidth=0.8)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    pdf_path = args.output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "clean_acc", "pert_acc", "mean_delta_recall", "min_delta_recall"],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"[Saved] {args.output_path}")
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {args.metrics_csv}")
    for row in metric_rows:
        print(
            f"[Metrics] {row['model']}: clean_acc={row['clean_acc']:.4f} pert_acc={row['pert_acc']:.4f} "
            f"mean_delta_recall={row['mean_delta_recall']:.4f} min_delta_recall={row['min_delta_recall']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HHAR confusion matrices and per-class recall shifts.")
    parser.add_argument("--csv-path", type=Path, default=Path("outputs/hhar_confusion_predictions.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("outputs/hhar_confusion_predictions_meta.json"))
    parser.add_argument("--output-path", type=Path, default=Path("figs/fig_hhar_confusion_mixed.png"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("outputs/hhar_confusion_plot_metrics.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    plot_hhar_confusion(parse_args())
