import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_reviewer_fast_experiments import (  # noqa: E402
    CkptSpec,
    LoadedModel,
    _load_model,
    _make_dataset,
    _mixed_shift_color,
    _predict_logits,
)
from train_uea import _domain_stratified_split, _stratified_split, collate_fn  # noqa: E402


MODEL_ORDER = ["Baseline", "Tri-view", "TriView-TA"]
MODEL_COLORS = {
    "Baseline": "#4C78A8",
    "Tri-view": "#F58518",
    "TriView-TA": "#54A24B",
}
HEATMAP_CMAP_NAME = "iceblue"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    HEATMAP_CMAP_NAME,
    [
        "#E6F4FF",
        "#B7DDF2",
        "#6BAED6",
        "#2171B5",
        "#08306B",
    ],
)


@dataclass(frozen=True)
class ModelSet:
    dataset: str
    tag: str
    split: str
    checkpoints: Dict[str, Path]
    priority: int


@dataclass(frozen=True)
class MixedCondition:
    severity_id: int
    shift_bins: float
    color_db: float


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _fmt_float(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:g}"


def _existing(path: str) -> Path:
    return ROOT / path


def _best_pt(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    scored = []
    for path in matches:
        name = path.name
        score = -math.inf
        for key in ("val_mf1=", "val_acc="):
            if key in name:
                try:
                    score = float(name.split(key, 1)[1].split(".pt", 1)[0])
                except ValueError:
                    score = -math.inf
                break
        scored.append((score, path))
    scored.sort(key=lambda item: (item[0], str(item[1])))
    return scored[-1][1]


def _stft_model_set(dataset: str, n_fft: int, priority: int) -> Optional[ModelSet]:
    base = ROOT / "outputs_46/stft_sensitivity_full_run2/checkpoints" / dataset / f"nfft{n_fft}" / "seed42"
    paths = {
        "Baseline": _best_pt(base / "baseline", f"stftsens_{dataset}_baseline_nfft{n_fft}_seed42_ep*_val_mf1=*.pt"),
        "Tri-view": _best_pt(base / "triview", f"stftsens_{dataset}_triview_nfft{n_fft}_seed42_ep*_val_mf1=*.pt"),
        "TriView-TA": _best_pt(base / "full", f"stftsens_{dataset}_full_nfft{n_fft}_seed42_ep*_val_mf1=*.pt"),
    }
    if any(path is None for path in paths.values()):
        return None
    return ModelSet(
        dataset=dataset,
        tag=f"{dataset}_stft_nfft{n_fft}",
        split="test",
        checkpoints={key: path for key, path in paths.items() if path is not None},
        priority=priority,
    )


def default_model_sets() -> List[ModelSet]:
    sets: List[ModelSet] = []
    hhar_main = ModelSet(
        dataset="HHAR",
        tag="HHAR_hhar6ch_val",
        split="val",
        checkpoints={
            "Baseline": _existing("outputs_new/hhar6ch_runs/hhar6ch_erm_time_simple_ep18_val_mf1=0.9188.pt"),
            "Tri-view": _existing("outputs_new/hhar6ch_runs/hhar6ch_triview_no_pretrain_ep9_val_mf1=0.9550.pt"),
            "TriView-TA": _existing("outputs_new/hhar6ch_runs/hhar6ch_triview_ta_ep17_val_mf1=0.9416.pt"),
        },
        priority=1,
    )
    sets.append(hhar_main)

    uwave_main = ModelSet(
        dataset="UWaveGestureLibrary",
        tag="UWaveGestureLibrary_paper_main",
        split="test",
        checkpoints={
            "Baseline": _existing("checkpoints/uwave_baseline_no_ta_ep16_val_mf1=0.8405.pt"),
            "Tri-view": _existing("checkpoints/uwave_triview_no_ta_ep17_val_mf1=0.7286.pt"),
            "TriView-TA": _existing("checkpoints/uwave_triview_ta_full_ep20_val_mf1=0.6807.pt"),
        },
        priority=2,
    )
    sets.append(uwave_main)

    for seed in (1, 7, 13):
        sets.append(
            ModelSet(
                dataset="UWaveGestureLibrary",
                tag=f"UWaveGestureLibrary_seedstab_s{seed}",
                split="test",
                checkpoints={
                    "Baseline": _best_pt(ROOT / "checkpoints", f"uwave_seedstab_baseline_s{seed}_ep*_val_mf1=*.pt")
                    or ROOT / "_missing_baseline.pt",
                    "Tri-view": _best_pt(ROOT / "checkpoints", f"uwave_seedstab_triview_s{seed}_ep*_val_mf1=*.pt")
                    or ROOT / "_missing_triview.pt",
                    "TriView-TA": _best_pt(ROOT / "checkpoints", f"uwave_seedstab_ta_s{seed}_ep*_val_mf1=*.pt")
                    or ROOT / "_missing_ta.pt",
                },
                priority=2,
            )
        )

    priority_by_dataset = {
        "HHAR": 1,
        "UWaveGestureLibrary": 2,
        "Handwriting": 3,
    }
    for dataset, nffts in {
        "HHAR": [32, 64, 128],
        "UWaveGestureLibrary": [128, 256, 512],
        "Handwriting": [16, 32, 64],
    }.items():
        for n_fft in nffts:
            model_set = _stft_model_set(dataset, n_fft, priority=priority_by_dataset[dataset])
            if model_set is not None:
                sets.append(model_set)

    filtered = []
    for model_set in sets:
        missing = [name for name, path in model_set.checkpoints.items() if not path.exists()]
        if missing:
            print(f"[Skip] {model_set.tag}: missing {', '.join(missing)}")
            continue
        filtered.append(model_set)
    return filtered


def build_conditions(shift_bins: Sequence[float], color_db: Sequence[float]) -> List[MixedCondition]:
    conditions: List[MixedCondition] = []
    for shift_id, bins in enumerate(shift_bins):
        for color_id, db in enumerate(color_db):
            severity_id = shift_id * len(color_db) + color_id
            conditions.append(MixedCondition(severity_id=severity_id, shift_bins=float(bins), color_db=float(db)))
    return conditions


def _make_eval_dataset(dataset_name: str, config: Dict[str, object], split: str):
    if split == "val":
        full_train = _make_dataset(dataset_name, config, split="train", return_freq=False)
        val_split = float(config.get("val_split", 0.1))
        seed = int(config.get("seed", 42))
        domain_ids = getattr(full_train, "domain_ids", None)
        if domain_ids is not None:
            _, val_indices = _domain_stratified_split(full_train.labels, domain_ids, val_split, seed)
        else:
            _, val_indices = _stratified_split(full_train.labels, val_split, seed)
        return Subset(full_train, val_indices)
    return _make_dataset(dataset_name, config, split=split, return_freq=False)


def _class_labels(dataset, loaded: LoadedModel) -> List[str]:
    def clean_label(label: object) -> str:
        text = str(label)
        try:
            value = float(text)
        except ValueError:
            return text
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return text

    if hasattr(dataset, "class_labels"):
        return [clean_label(label) for label in dataset.class_labels]
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "class_labels"):
        return [clean_label(label) for label in dataset.dataset.class_labels]
    return [clean_label(label) for label in loaded.class_labels]


@torch.no_grad()
def collect_predictions(
    loaded: LoadedModel,
    loader: DataLoader,
    device: str,
    condition: Optional[MixedCondition],
    perturb_loaded: LoadedModel,
    perturb_seed: int,
    color_bands: int,
) -> Tuple[List[int], List[int], List[int]]:
    sample_ids: List[int] = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    offset = 0
    loaded.model.eval()
    for batch in loader:
        x = batch["x_time"].to(device)
        y = batch["y"].to(device)
        if condition is not None:
            x = _mixed_shift_color(
                x,
                perturb_loaded,
                seed=int(perturb_seed) + int(condition.severity_id) * 10000,
                shift_bins=float(condition.shift_bins),
                color_db=float(condition.color_db),
                color_bands=int(color_bands),
            )
        logits = _predict_logits(loaded, x)
        pred = logits.argmax(dim=1)
        batch_size = int(y.numel())
        sample_ids.extend(range(offset, offset + batch_size))
        y_true_all.extend(int(v) for v in y.detach().cpu().tolist())
        y_pred_all.extend(int(v) for v in pred.detach().cpu().tolist())
        offset += batch_size
    return sample_ids, y_true_all, y_pred_all


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    index = {int(label): idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if int(yt) in index and int(yp) in index:
            cm[index[int(yt)], index[int(yp)]] += 1
    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)


def per_class_recall(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels)
    denom = cm.sum(axis=1).astype(np.float64)
    diag = np.diag(cm).astype(np.float64)
    return np.divide(diag, denom, out=np.zeros_like(diag), where=denom != 0)


def compute_metrics(
    y_true_clean: Sequence[int],
    y_pred_clean: Sequence[int],
    y_true_pert: Sequence[int],
    y_pred_pert: Sequence[int],
    labels: Sequence[int],
) -> Dict[str, object]:
    cm_pert = confusion_matrix(y_true_pert, y_pred_pert, labels)
    cm_norm = row_normalize(cm_pert)
    recall_clean = per_class_recall(y_true_clean, y_pred_clean, labels)
    recall_pert = per_class_recall(y_true_pert, y_pred_pert, labels)
    macro_recall_pert = float(np.mean(recall_pert))
    mean_recall_drop = float(np.mean(recall_clean - recall_pert))
    offdiag_mass = float(1.0 - np.mean(np.diag(cm_norm)))
    clean_acc = float(np.mean(np.asarray(y_true_clean) == np.asarray(y_pred_clean)))
    pert_acc = float(np.mean(np.asarray(y_true_pert) == np.asarray(y_pred_pert)))
    return {
        "macro_recall_pert": macro_recall_pert,
        "mean_recall_drop": mean_recall_drop,
        "offdiag_mass": offdiag_mass,
        "clean_acc": clean_acc,
        "pert_acc": pert_acc,
        "cm_norm": cm_norm,
        "recall_clean": recall_clean,
        "recall_pert": recall_pert,
    }


def score_condition(metrics: Dict[str, Dict[str, object]]) -> float:
    tri = metrics["Tri-view"]
    ta = metrics["TriView-TA"]
    return float(
        (ta["macro_recall_pert"] - tri["macro_recall_pert"])
        + (tri["mean_recall_drop"] - ta["mean_recall_drop"])
        + (tri["offdiag_mass"] - ta["offdiag_mass"])
    )


def is_valid_condition(metrics: Dict[str, Dict[str, object]], min_gap: float) -> bool:
    tri = metrics["Tri-view"]
    ta = metrics["TriView-TA"]
    return bool(
        ta["macro_recall_pert"] > tri["macro_recall_pert"] + min_gap
        and tri["mean_recall_drop"] > ta["mean_recall_drop"]
        and tri["offdiag_mass"] > ta["offdiag_mass"]
    )


def is_baseline_consistent_condition(metrics: Dict[str, Dict[str, object]], min_gap: float) -> bool:
    if not is_valid_condition(metrics, min_gap=min_gap):
        return False
    base = metrics["Baseline"]
    tri = metrics["Tri-view"]
    ta = metrics["TriView-TA"]
    return bool(
        ta["macro_recall_pert"] > base["macro_recall_pert"]
        and ta["mean_recall_drop"] < base["mean_recall_drop"]
        and ta["offdiag_mass"] < base["offdiag_mass"]
        and ta["clean_acc"] > base["clean_acc"]
        and tri["macro_recall_pert"] < base["macro_recall_pert"]
        and tri["clean_acc"] < base["clean_acc"]
    )


def is_compound_condition(condition: MixedCondition) -> bool:
    return abs(float(condition.shift_bins)) > 1e-10 and abs(float(condition.color_db)) > 1e-10


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: List[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _model_label_for_spec(model_name: str) -> str:
    return {"Baseline": "Baseline", "Tri-view": "Tri-view", "TriView-TA": "TriView-TA"}[model_name]


def _load_model_set(model_set: ModelSet, device: str) -> Dict[str, LoadedModel]:
    loaded: Dict[str, LoadedModel] = {}
    load_split = "test" if model_set.split == "val" else model_set.split
    for model_name in MODEL_ORDER:
        spec = CkptSpec(model_set.dataset, _model_label_for_spec(model_name), model_set.checkpoints[model_name])
        print(f"[Load] {model_set.tag}/{model_name}: {spec.path}")
        loaded[model_name] = _load_model(spec, device=device, split=load_split)
    return loaded


def _row_for_metrics(
    model_set: ModelSet,
    condition: MixedCondition,
    model_name: str,
    metrics: Dict[str, object],
    score: float,
    valid: bool,
    baseline_consistent: bool,
    compound: bool,
) -> Dict[str, object]:
    return {
        "dataset": model_set.dataset,
        "candidate": model_set.tag,
        "split": model_set.split,
        "priority": model_set.priority,
        "severity_id": condition.severity_id,
        "b_bins": condition.shift_bins,
        "g_db": condition.color_db,
        "model": model_name,
        "macro_recall_pert": metrics["macro_recall_pert"],
        "mean_recall_drop": metrics["mean_recall_drop"],
        "offdiag_mass": metrics["offdiag_mass"],
        "clean_acc": metrics["clean_acc"],
        "pert_acc": metrics["pert_acc"],
        "score": score,
        "valid": int(valid),
        "baseline_consistent": int(baseline_consistent),
        "compound_mixed": int(compound),
    }


def scan(args: argparse.Namespace) -> Dict[str, object]:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    model_sets = default_model_sets()
    if args.datasets.strip():
        allowed = {item.strip() for item in args.datasets.split(",") if item.strip()}
        model_sets = [model_set for model_set in model_sets if model_set.dataset in allowed]
    if args.candidates.strip():
        allowed_tags = {item.strip() for item in args.candidates.split(",") if item.strip()}
        model_sets = [model_set for model_set in model_sets if model_set.tag in allowed_tags]

    conditions = build_conditions(_parse_float_list(args.mixed_shift_bins), _parse_float_list(args.mixed_color_db))
    if not conditions:
        raise ValueError("No mixed conditions were provided.")

    print(f"[Scan] candidates={len(model_sets)} conditions={len(conditions)} min_gap={args.min_gap:g}")
    scan_rows: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    fallback_best: Optional[Dict[str, object]] = None

    for model_set in model_sets:
        print(f"[Candidate] {model_set.tag} ({model_set.dataset}, split={model_set.split})")
        try:
            loaded = _load_model_set(model_set, device=device)
            reference = loaded["Baseline"]
            dataset = _make_eval_dataset(model_set.dataset, reference.config, model_set.split)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            class_labels = _class_labels(dataset, reference)
            labels = list(range(len(class_labels)))

            clean_cache: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
            for model_name in MODEL_ORDER:
                clean_cache[model_name] = collect_predictions(
                    loaded[model_name],
                    loader,
                    device=device,
                    condition=None,
                    perturb_loaded=reference,
                    perturb_seed=args.perturb_seed,
                    color_bands=args.color_bands,
                )
                print(
                    f"[Clean] {model_set.tag}/{model_name}: "
                    f"acc={np.mean(np.asarray(clean_cache[model_name][1]) == np.asarray(clean_cache[model_name][2])):.4f}"
                )

            for condition in conditions:
                metrics_by_model: Dict[str, Dict[str, object]] = {}
                prediction_cache: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
                aligned = True
                for model_name in MODEL_ORDER:
                    pert = collect_predictions(
                        loaded[model_name],
                        loader,
                        device=device,
                        condition=condition,
                        perturb_loaded=reference,
                        perturb_seed=args.perturb_seed,
                        color_bands=args.color_bands,
                    )
                    prediction_cache[model_name] = pert
                    clean_ids, y_true_clean, y_pred_clean = clean_cache[model_name]
                    pert_ids, y_true_pert, y_pred_pert = pert
                    if clean_ids != pert_ids or y_true_clean != y_true_pert:
                        aligned = False
                    metrics_by_model[model_name] = compute_metrics(
                        y_true_clean,
                        y_pred_clean,
                        y_true_pert,
                        y_pred_pert,
                        labels,
                    )
                if not aligned:
                    raise RuntimeError(f"Clean/pert sample order mismatch for {model_set.tag}.")
                score = score_condition(metrics_by_model)
                valid = is_valid_condition(metrics_by_model, min_gap=args.min_gap)
                baseline_consistent = is_baseline_consistent_condition(metrics_by_model, min_gap=args.min_gap)
                compound = is_compound_condition(condition)
                for model_name in MODEL_ORDER:
                    scan_rows.append(
                        _row_for_metrics(
                            model_set,
                            condition,
                            model_name,
                            metrics_by_model[model_name],
                            score,
                            valid,
                            baseline_consistent,
                            compound,
                        )
                    )

                tri = metrics_by_model["Tri-view"]
                ta = metrics_by_model["TriView-TA"]
                print(
                    f"[Cond] {model_set.tag} sid={condition.severity_id} "
                    f"b={_fmt_float(condition.shift_bins)} g={_fmt_float(condition.color_db)} "
                    f"TA-Tri={ta['macro_recall_pert'] - tri['macro_recall_pert']:+.4f} "
                    f"drop_gap={tri['mean_recall_drop'] - ta['mean_recall_drop']:+.4f} "
                    f"offdiag_gap={tri['offdiag_mass'] - ta['offdiag_mass']:+.4f} "
                    f"score={score:+.4f} valid={int(valid)} strong={int(baseline_consistent)} compound={int(compound)}"
                )

                candidate_payload = {
                    "model_set": model_set,
                    "condition": condition,
                    "metrics": metrics_by_model,
                    "predictions": prediction_cache,
                    "clean": clean_cache,
                    "class_labels": class_labels,
                    "labels": labels,
                    "score": score,
                    "valid": valid,
                    "baseline_consistent": baseline_consistent,
                    "compound": compound,
                    "device": device,
                }
                if fallback_best is None or score > float(fallback_best["score"]):
                    fallback_best = candidate_payload
                if valid and (
                    best is None
                    or (int(baseline_consistent and compound), int(baseline_consistent), score)
                    > (
                        int(bool(best.get("baseline_consistent", False)) and bool(best.get("compound", False))),
                        int(bool(best.get("baseline_consistent", False))),
                        float(best["score"]),
                    )
                ):
                    best = candidate_payload
        except Exception as exc:  # Keep scanning other candidates if one checkpoint family is incompatible.
            print(f"[Skip] {model_set.tag}: {type(exc).__name__}: {exc}")
            scan_rows.append(
                {
                    "dataset": model_set.dataset,
                    "candidate": model_set.tag,
                    "split": model_set.split,
                    "priority": model_set.priority,
                    "status": "skipped_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    _write_csv(args.scan_csv, scan_rows)
    print(f"[Saved] {args.scan_csv}")
    if best is None:
        if fallback_best is not None:
            print(
                "[Result] No condition satisfied the strict validity rule. "
                f"Best fallback score={float(fallback_best['score']):.4f}, but no final figure was generated."
            )
        raise SystemExit(2)
    return best


def write_selected_outputs(best: Dict[str, object], args: argparse.Namespace) -> None:
    model_set: ModelSet = best["model_set"]
    condition: MixedCondition = best["condition"]
    class_labels: List[str] = best["class_labels"]
    metrics_by_model: Dict[str, Dict[str, object]] = best["metrics"]
    clean_cache = best["clean"]
    prediction_cache = best["predictions"]

    pred_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        clean_ids, y_true_clean, y_pred_clean = clean_cache[model_name]
        pert_ids, y_true_pert, y_pred_pert = prediction_cache[model_name]
        for sample_id, yt, yp in zip(clean_ids, y_true_clean, y_pred_clean):
            pred_rows.append(
                {
                    "dataset": model_set.dataset,
                    "candidate": model_set.tag,
                    "model": model_name,
                    "condition": "clean",
                    "sample_id": sample_id,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )
        for sample_id, yt, yp in zip(pert_ids, y_true_pert, y_pred_pert):
            pred_rows.append(
                {
                    "dataset": model_set.dataset,
                    "candidate": model_set.tag,
                    "model": model_name,
                    "condition": "pert",
                    "sample_id": sample_id,
                    "y_true": yt,
                    "y_pred": yp,
                    "severity_id": condition.severity_id,
                    "b_bins": condition.shift_bins,
                    "g_db": condition.color_db,
                }
            )
        metric = metrics_by_model[model_name]
        metric_rows.append(
            {
                "dataset": model_set.dataset,
                "candidate": model_set.tag,
                "split": model_set.split,
                "model": model_name,
                "severity_id": condition.severity_id,
                "b_bins": condition.shift_bins,
                "g_db": condition.color_db,
                "macro_recall_pert": metric["macro_recall_pert"],
                "mean_recall_drop": metric["mean_recall_drop"],
                "offdiag_mass": metric["offdiag_mass"],
                "clean_acc": metric["clean_acc"],
                "pert_acc": metric["pert_acc"],
                "score": best["score"],
            }
        )

    _write_csv(args.predictions_csv, pred_rows)
    _write_csv(args.metrics_csv, metric_rows)

    meta = {
        "dataset": model_set.dataset,
        "candidate": model_set.tag,
        "split": model_set.split,
        "class_labels": class_labels,
        "selection_rule": {
            "min_gap": args.min_gap,
            "validity": [
                "TriView-TA perturbed macro recall > Tri-view perturbed macro recall + min_gap",
                "Tri-view mean recall drop > TriView-TA mean recall drop",
                "Tri-view off-diagonal mass > TriView-TA off-diagonal mass",
            ],
            "baseline_consistent_preference": [
                "TriView-TA perturbed macro recall > Baseline perturbed macro recall",
                "TriView-TA mean recall drop < Baseline mean recall drop",
                "TriView-TA off-diagonal mass < Baseline off-diagonal mass",
                "TriView-TA clean accuracy > Baseline clean accuracy",
                "Tri-view perturbed macro recall < Baseline perturbed macro recall",
                "Tri-view clean accuracy < Baseline clean accuracy",
            ],
            "compound_mixed_preference": "Prefer retained mixed conditions with both nonzero shift and nonzero coloring.",
            "score": "(TA macro recall pert - Tri macro recall pert) + (Tri mean recall drop - TA mean recall drop) + (Tri offdiag mass - TA offdiag mass)",
        },
        "selected_condition": {
            "severity_id": condition.severity_id,
            "mixed_shift_bins": condition.shift_bins,
            "mixed_color_db": condition.color_db,
            "mixed_color_bands": args.color_bands,
            "perturb_seed": args.perturb_seed,
        },
        "checkpoints": {model: str(path) for model, path in model_set.checkpoints.items()},
        "outputs": {
            "predictions_csv": str(args.predictions_csv),
            "metrics_csv": str(args.metrics_csv),
            "scan_csv": str(args.scan_csv),
            "table_tex": str(args.table_tex),
            "figure_png": str(args.output_path),
            "figure_pdf": str(args.output_path.with_suffix(".pdf")),
        },
    }
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"[Saved] {args.predictions_csv}")
    print(f"[Saved] {args.metrics_csv}")
    print(f"[Saved] {args.meta_json}")


def write_latex_table(best: Dict[str, object], args: argparse.Namespace) -> None:
    metrics_by_model: Dict[str, Dict[str, object]] = best["metrics"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Aggregate prediction-level diagnostic under the selected retained mixed perturbation condition on UWaveGestureLibrary. Macro Recall is computed under the perturbed condition. Mean Recall Drop denotes the average class-wise recall decrease from clean to perturbed inputs. Offdiag Mass denotes the off-diagonal mass of the row-normalized confusion matrix.}",
        r"\label{tab:prediction_diag}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Macro Recall $\uparrow$ & Mean Recall Drop $\downarrow$ & Offdiag Mass $\downarrow$ \\",
        r"\midrule",
    ]
    for model_name in MODEL_ORDER:
        metric = metrics_by_model[model_name]
        lines.append(
            f"{model_name} & {float(metric['macro_recall_pert']):.4f} & "
            f"{float(metric['mean_recall_drop']):.4f} & {float(metric['offdiag_mass']):.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    args.table_tex.parent.mkdir(parents=True, exist_ok=True)
    args.table_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {args.table_tex}")


def plot_selected(best: Dict[str, object], args: argparse.Namespace) -> None:
    metrics_by_model: Dict[str, Dict[str, object]] = best["metrics"]

    n_classes = len(best["class_labels"])
    class_labels = [f"C{idx + 1}" for idx in range(n_classes)]
    tick_font = 8 if n_classes <= 10 else 6
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.65),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
        constrained_layout=True,
    )

    image = None
    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes[idx]
        cm_norm = metrics_by_model[model_name]["cm_norm"]
        image = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap=HEATMAP_CMAP)
        ax.set_title(model_name, fontsize=10.5)
        ax.set_xlabel("Predicted label")
        if idx == 0:
            ax.set_ylabel("True label")
        ticks = np.arange(n_classes)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=tick_font)
        ax.set_yticklabels(class_labels, fontsize=tick_font)
        if n_classes <= 12:
            for i in range(n_classes):
                for j in range(n_classes):
                    value = float(cm_norm[i, j])
                    color = "white" if value >= 0.55 else "black"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    if image is not None:
        cbar = fig.colorbar(image, ax=list(axes), shrink=0.82)
        cbar.set_label("Row-normalized frequency")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    pdf_path = args.output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {args.output_path}")
    print(f"[Saved] {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and plot one prediction-level mixed-perturbation diagnostic."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--perturb-seed", type=int, default=42)
    parser.add_argument("--mixed-shift-bins", type=str, default="-0.1,0,0.1")
    parser.add_argument("--mixed-color-db", type=str, default="0,3,6")
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--scan-csv", type=Path, default=Path("outputs/confusion_mixed_selected_scan.csv"))
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/confusion_mixed_selected_predictions.csv"),
    )
    parser.add_argument("--metrics-csv", type=Path, default=Path("outputs/confusion_mixed_selected_metrics.csv"))
    parser.add_argument("--meta-json", type=Path, default=Path("outputs/confusion_mixed_selected_meta.json"))
    parser.add_argument("--table-tex", type=Path, default=Path("outputs/confusion_mixed_selected_table.tex"))
    parser.add_argument("--output-path", type=Path, default=Path("figs/fig_prediction_confusion_uwave_selected.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = scan(args)
    write_selected_outputs(best, args)
    write_latex_table(best, args)
    plot_selected(best, args)
    model_set: ModelSet = best["model_set"]
    condition: MixedCondition = best["condition"]
    print(
        f"[Selected] {model_set.tag} dataset={model_set.dataset} split={model_set.split} "
        f"severity_id={condition.severity_id} b={_fmt_float(condition.shift_bins)} "
        f"g={_fmt_float(condition.color_db)} score={float(best['score']):.4f}"
    )
    for model_name in MODEL_ORDER:
        metric = best["metrics"][model_name]
        print(
            f"[Metric] {model_name}: macro_recall_pert={float(metric['macro_recall_pert']):.4f} "
            f"mean_recall_drop={float(metric['mean_recall_drop']):.4f} "
            f"offdiag_mass={float(metric['offdiag_mass']):.4f}"
        )


if __name__ == "__main__":
    main()
