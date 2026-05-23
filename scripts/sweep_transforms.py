import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import UEATimeSeriesDataset, ViewConfig  # noqa: E402
from eval_utils import apply_per_sample_channel, assert_same_training_budget, evaluate_classifier  # noqa: E402
from output_utils import build_tag, ensure_output_dirs, stable_hash, write_csv, write_run_meta  # noqa: E402
from train_uea import UEAClassifier, UEAFreqViewClassifier, UEATriViewClassifier, collate_fn  # noqa: E402
from transforms import (  # noqa: E402
    band_shift_time,
    band_shift_time_stft,
    frequency_scale_time,
    make_coloring_gains,
    spectral_coloring,
)

_BUDGET_KEYS = (
    "pretrain_epochs",
    "freeze_epochs",
    "finetune_epochs",
    "patience",
    "min_delta",
    "val_split",
    "batch_size",
    "lr",
    "encoder_lr",
    "head_lr",
    "weight_decay",
    "use_cosine",
    "use_swa",
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


def _resolve_supervised_views(config: Dict[str, object]) -> str:
    raw = str(config.get("supervised_views", "time")).strip().lower()
    if raw in {"time", "timefreq", "triview"}:
        return raw
    return "time"


def _build_model(
    config: Dict[str, object],
    input_dim: int,
    num_classes: int,
    device: str,
    input_dim_freq: int,
    input_dim_tf: int,
):
    supervised_views = _resolve_supervised_views(config)
    common_kwargs = dict(
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
    )
    if supervised_views == "triview":
        model = UEATriViewClassifier(
            input_dim_time=input_dim,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
            triview_fusion=str(config.get("triview_fusion", "gated")),
            gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
            gate_dropout=float(config.get("gate_dropout", 0.0)),
            gate_temperature=float(config.get("gate_temperature", 1.0)),
            **common_kwargs,
        ).to(device)
        return model, supervised_views
    if supervised_views == "timefreq":
        model = UEAFreqViewClassifier(
            input_dim_time=input_dim,
            input_dim_freq=input_dim_freq,
            triview_fusion=str(config.get("triview_fusion", "gated")),
            gate_hidden_dim=int(config.get("gate_hidden_dim", 64)),
            gate_dropout=float(config.get("gate_dropout", 0.0)),
            gate_temperature=float(config.get("gate_temperature", 1.0)),
            **common_kwargs,
        ).to(device)
        return model, supervised_views
    model = UEAClassifier(
        input_dim=input_dim,
        **common_kwargs,
    ).to(device)
    return model, supervised_views


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _x_key_for_transform(transform_name: str) -> Optional[str]:
    if transform_name == "shift":
        return "b_bins"
    if transform_name == "scale":
        return "rho"
    if transform_name == "color":
        return "g_db"
    if transform_name == "mixed_shift_color":
        return "severity_id"
    return None


def _x_value(row: Dict[str, object], transform_name: str) -> float:
    x_key = _x_key_for_transform(transform_name)
    if x_key is None:
        return 0.0
    default = 1.0 if x_key == "rho" else 0.0
    return _safe_float(row.get(x_key, default), default)


def _curve_points(rows: List[Dict[str, object]], transform_name: str) -> List[Dict[str, object]]:
    x_key = _x_key_for_transform(transform_name)
    if x_key is None:
        return []
    per_x: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        x_val = _x_value(row, transform_name)
        per_x.setdefault(x_val, []).append(row)

    ordered = []
    for x_val in sorted(per_x.keys()):
        candidates = per_x[x_val]
        mean_row = next((r for r in candidates if str(r.get("trial")) == "mean"), None)
        if mean_row is not None:
            ordered.append(mean_row)
            continue
        zero_row = next((r for r in candidates if str(r.get("trial")) in {"0", "0.0"}), None)
        if zero_row is not None:
            ordered.append(zero_row)
            continue
        ordered.append(candidates[0])
    return ordered


def _aggregate_curve_by_group(
    rows: List[Dict[str, object]],
    group_name: str,
    transform_name: str,
) -> Tuple[List[float], List[float], List[float], int]:
    rows_group = [
        row
        for row in rows
        if str(row.get("checkpoint", "")) == group_name and str(row.get("transform", "")) == transform_name
    ]
    if not rows_group:
        return [], [], [], 0

    per_run: Dict[str, List[Dict[str, object]]] = {}
    for row in rows_group:
        run_id = str(row.get("run_id", "run0"))
        per_run.setdefault(run_id, []).append(row)

    x_to_values: Dict[float, List[float]] = {}
    for run_rows in per_run.values():
        points = _curve_points(run_rows, transform_name)
        for point in points:
            x_val = _x_value(point, transform_name)
            x_to_values.setdefault(x_val, []).append(_safe_float(point.get("acc", 0.0), 0.0))

    xs = sorted(x_to_values.keys())
    means: List[float] = []
    stds: List[float] = []
    for x in xs:
        vals = x_to_values[x]
        if not vals:
            means.append(0.0)
            stds.append(0.0)
            continue
        t = torch.tensor(vals, dtype=torch.float32)
        means.append(float(t.mean().item()))
        stds.append(float(t.std(unbiased=False).item()) if len(vals) > 1 else 0.0)
    return xs, means, stds, len(per_run)


def _normalized_auc(xs: List[float], ys: List[float]) -> float:
    if not ys:
        return 0.0
    if len(xs) <= 1 or (xs[-1] - xs[0]) <= 1e-12:
        return float(ys[0])
    auc = 0.0
    for i in range(len(xs) - 1):
        width = xs[i + 1] - xs[i]
        auc += width * (ys[i] + ys[i + 1]) * 0.5
    return float(auc / (xs[-1] - xs[0]))


def _slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    x_t = torch.tensor(xs, dtype=torch.float32)
    y_t = torch.tensor(ys, dtype=torch.float32)
    x_center = x_t - x_t.mean()
    denom = float((x_center * x_center).sum().item())
    if denom <= 1e-12:
        return 0.0
    y_center = y_t - y_t.mean()
    return float(((x_center * y_center).sum().item()) / denom)


def _run_level_metrics(
    rows: List[Dict[str, object]],
    group_name: str,
    transform_name: str,
) -> List[Dict[str, float]]:
    rows_group = [
        row
        for row in rows
        if str(row.get("checkpoint", "")) == group_name and str(row.get("transform", "")) == transform_name
    ]
    if not rows_group:
        return []

    per_run: Dict[str, List[Dict[str, object]]] = {}
    for row in rows_group:
        run_id = str(row.get("run_id", "run0"))
        per_run.setdefault(run_id, []).append(row)

    metrics = []
    for run_id, run_rows in per_run.items():
        points = _curve_points(run_rows, transform_name)
        if not points:
            continue
        xs = [_x_value(p, transform_name) for p in points]
        acc_vals = [_safe_float(p.get("acc", 0.0), 0.0) for p in points]
        mf1_vals = [_safe_float(p.get("mf1", 0.0), 0.0) for p in points]

        clean_row = next(
            (
                row
                for row in rows
                if str(row.get("checkpoint", "")) == group_name
                and str(row.get("run_id", "run0")) == run_id
                and str(row.get("transform", "")) == "clean"
            ),
            None,
        )
        clean_acc = _safe_float(clean_row.get("acc", 0.0), 0.0) if clean_row is not None else 0.0
        clean_mf1 = _safe_float(clean_row.get("mf1", 0.0), 0.0) if clean_row is not None else 0.0

        avg_acc = sum(acc_vals) / len(acc_vals)
        worst_acc = min(acc_vals)
        rauc_acc = _normalized_auc(xs, acc_vals)
        slope_acc = _slope(xs, acc_vals)

        avg_mf1 = sum(mf1_vals) / len(mf1_vals)
        worst_mf1 = min(mf1_vals)
        rauc_mf1 = _normalized_auc(xs, mf1_vals)
        slope_mf1 = _slope(xs, mf1_vals)

        metrics.append(
            {
                "run_id": run_id,
                "clean_acc": clean_acc,
                "avg_acc": avg_acc,
                "worst_acc": worst_acc,
                "drop_avg_acc": clean_acc - avg_acc,
                "drop_worst_acc": clean_acc - worst_acc,
                "rauc_acc": rauc_acc,
                "slope_acc": slope_acc,
                "clean_mf1": clean_mf1,
                "avg_mf1": avg_mf1,
                "worst_mf1": worst_mf1,
                "drop_avg_mf1": clean_mf1 - avg_mf1,
                "drop_worst_mf1": clean_mf1 - worst_mf1,
                "rauc_mf1": rauc_mf1,
                "slope_mf1": slope_mf1,
            }
        )
    return metrics


def _interval_from_values(values: List[float]) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    return (min(values), max(values))


def _load_safe_ranges_from_summary(path: Path, threshold: float) -> Dict[str, List[float]]:
    safe = {"shift": [], "scale": [], "color": [], "mixed_shift_color": []}
    if not path.exists():
        raise FileNotFoundError(f"Safe summary csv not found: {path}")
    by_transform = {"shift": set(), "scale": set(), "color": set(), "mixed_shift_color": set()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            transform = str(row.get("transform", "")).strip()
            if transform not in by_transform:
                continue
            has_safe_flag = "safe" in row and str(row.get("safe", "")).strip() != ""
            if has_safe_flag:
                is_safe = int(_safe_float(row.get("safe", 0.0), 0.0)) == 1
            else:
                agreement = _safe_float(row.get("agreement", 0.0), 0.0)
                is_safe = agreement >= threshold
            if not is_safe:
                continue
            if transform == "shift":
                by_transform["shift"].add(_safe_float(row.get("b_bins", 0.0), 0.0))
            elif transform == "scale":
                by_transform["scale"].add(_safe_float(row.get("rho", 1.0), 1.0))
            elif transform == "color":
                by_transform["color"].add(_safe_float(row.get("g_db", 0.0), 0.0))
            elif transform == "mixed_shift_color":
                by_transform["mixed_shift_color"].add(_safe_float(row.get("severity_id", 0.0), 0.0))
    for key in safe:
        safe[key] = sorted(by_transform[key])
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Optional comma-separated labels for checkpoints (same order as --checkpoints).",
    )
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--severity-source",
        type=str,
        default="fixed",
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
    parser.add_argument("--scale-ratios", type=str, default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--color-max-db", type=str, default="0,3,6,9")
    parser.add_argument("--color-bands", type=int, default=8)
    parser.add_argument("--color-active-bands", type=str, default="")
    parser.add_argument("--color-trials", type=int, default=1)
    parser.add_argument("--enable-mixed", action="store_true", default=False)
    parser.add_argument(
        "--mixed-shift-bins",
        type=str,
        default="",
        help="Optional shift bins for mixed shift+color sweep. Empty uses --shift-bins.",
    )
    parser.add_argument(
        "--mixed-color-max-db",
        type=str,
        default="",
        help="Optional color strengths for mixed shift+color sweep. Empty uses --color-max-db.",
    )
    parser.add_argument("--mixed-trials", type=int, default=1)
    parser.add_argument("--max-param-rows", type=int, default=1000)
    parser.add_argument(
        "--safe-shift-csv",
        type=str,
        default="",
        help="Optional path to teacher_agreement *_safe_shift.csv; if set, only rows with safe=1 are used.",
    )
    parser.add_argument(
        "--safe-agreement-summary-csv",
        type=str,
        default="",
        help="Optional teacher_agreement *_summary.csv used to derive safe ranges for shift/scale/color.",
    )
    parser.add_argument(
        "--safe-agreement-threshold",
        type=float,
        default=0.95,
        help="Agreement threshold used when parsing --safe-agreement-summary-csv.",
    )
    parser.add_argument(
        "--safe-signal-summary-csv",
        type=str,
        default="",
        help="Optional safe_signal *_summary.csv used to derive safe ranges for shift/scale/color.",
    )
    parser.add_argument("--enforce-same-budget", action="store_true", default=True)
    parser.add_argument("--no-enforce-same-budget", dest="enforce_same_budget", action="store_false")
    parser.add_argument("--output-root", type=str, default="outputs")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(args.seed)

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    loaded = [_load_checkpoint(path, device) for path in checkpoint_paths]
    labels = [ckpt_path.stem for ckpt_path in checkpoint_paths]
    if args.labels.strip():
        parsed_labels = [item.strip() for item in args.labels.split(",") if item.strip()]
        if len(parsed_labels) != len(checkpoint_paths):
            raise ValueError("--labels count must match --checkpoints count.")
        labels = parsed_labels
    configs = [cfg for _, cfg, _ in loaded]
    if args.enforce_same_budget:
        assert_same_training_budget(configs, _BUDGET_KEYS)

    dataset_name = args.dataset or str(configs[0].get("dataset", ""))
    if not dataset_name:
        raise ValueError("Dataset name must be provided via --dataset or checkpoint config.")
    for cfg in configs:
        if str(cfg.get("dataset", dataset_name)) != dataset_name:
            raise ValueError("All checkpoints must target the same dataset for fair comparison.")

    ref_cfg = configs[0]
    view_config = ViewConfig(
        n_fft=args.n_fft if args.n_fft is not None else int(ref_cfg.get("n_fft", 256)),
        hop_length=args.hop_length if args.hop_length is not None else int(ref_cfg.get("hop_length", 64)),
        win_length=args.stft_win_length if args.stft_win_length is not None else ref_cfg.get("stft_win_length"),
        window_name=args.stft_window or str(ref_cfg.get("stft_window", "hann")),
        center=bool(ref_cfg.get("stft_center", args.stft_center)),
        magnitude_power=(
            args.stft_magnitude_power
            if args.stft_magnitude_power is not None
            else float(ref_cfg.get("stft_magnitude_power", 1.0))
        ),
        tf_log1p=bool(ref_cfg.get("tf_log1p", args.tf_log1p)),
        tf_flatten=bool(ref_cfg.get("tf_flatten", args.tf_flatten)),
        normalize_mode=args.normalize_mode,
        shift_mode=args.shift_fill,
    )
    preprocess_config = view_config.to_preprocess_config()
    supervised_views_all = [_resolve_supervised_views(cfg) for cfg in configs]
    need_freq_views = any(v in {"timefreq", "triview"} for v in supervised_views_all)

    dataset = UEATimeSeriesDataset(
        dataset_name,
        split=args.split,
        pad_to_max=args.pad_to_max,
        return_freq=need_freq_views,
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
    input_dim_freq = 1
    input_dim_tf = 1
    if need_freq_views:
        probe = dataset[0]
        if "x_freq" not in probe or "x_tf" not in probe:
            raise ValueError("Dataset is missing x_freq/x_tf needed for timefreq/triview evaluation.")
        x_freq_probe = probe["x_freq"]
        x_tf_probe = probe["x_tf"]
        input_dim_freq = int(x_freq_probe.shape[0]) if x_freq_probe.dim() > 1 else 1
        input_dim_tf = int(x_tf_probe.shape[0]) if x_tf_probe.dim() > 1 else 1
    num_classes = len(dataset.class_labels)

    shift_bins = _parse_float_list(args.shift_bins)
    safe_ranges: Dict[str, List[float]] = {"shift": [], "scale": [], "color": [], "mixed_shift_color": []}
    safe_shift_source = ""
    safe_agreement_source = ""
    safe_signal_source = ""
    if args.safe_agreement_summary_csv.strip():
        safe_summary_path = Path(args.safe_agreement_summary_csv)
        safe_from_agreement = _load_safe_ranges_from_summary(safe_summary_path, float(args.safe_agreement_threshold))
        for key in safe_ranges:
            if safe_from_agreement.get(key):
                safe_ranges[key] = safe_from_agreement[key]
        safe_agreement_source = str(safe_summary_path)
        if safe_ranges["shift"]:
            shift_bins = sorted(set(safe_ranges["shift"]))
            safe_shift_source = str(safe_summary_path)
    if args.safe_signal_summary_csv.strip():
        safe_signal_path = Path(args.safe_signal_summary_csv)
        safe_from_signal = _load_safe_ranges_from_summary(safe_signal_path, float(args.safe_agreement_threshold))
        for key in ("shift", "scale", "color", "mixed_shift_color"):
            if safe_from_signal.get(key):
                safe_ranges[key] = safe_from_signal[key]
        safe_signal_source = str(safe_signal_path)
        if safe_ranges["shift"]:
            shift_bins = sorted(set(safe_ranges["shift"]))
            safe_shift_source = str(safe_signal_path)
    if args.safe_shift_csv.strip():
        safe_path = Path(args.safe_shift_csv)
        if not safe_path.exists():
            raise FileNotFoundError(f"--safe-shift-csv not found: {safe_path}")
        loaded_safe_bins = []
        with safe_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    is_safe = int(float(str(row.get("safe", "0")))) == 1
                except ValueError:
                    is_safe = False
                if not is_safe:
                    continue
                loaded_safe_bins.append(float(str(row.get("b_bins", "0"))))
        if loaded_safe_bins and not safe_ranges["shift"]:
            shift_bins = sorted(set(loaded_safe_bins))
            safe_ranges["shift"] = shift_bins[:]
            safe_shift_source = str(safe_path)
    if not shift_bins:
        raise ValueError("No shift bins available for sweep. Provide --shift-bins or safe shift sources.")
    scale_ratios = _parse_float_list(args.scale_ratios)
    color_levels = _parse_float_list(args.color_max_db)
    mixed_shift_bins = _parse_float_list(args.mixed_shift_bins) or shift_bins
    mixed_color_levels = _parse_float_list(args.mixed_color_max_db) or color_levels
    mixed_trials = max(1, int(args.mixed_trials))
    color_active_bands = _parse_band_list(args.color_active_bands)
    color_trials = max(1, int(args.color_trials))
    num_bins = dataset.data[0].shape[-1] // 2 + 1
    safe_mixed_ids = {int(v) for v in safe_ranges["mixed_shift_color"]}

    summary_rows = []
    param_rows = []

    def _maybe_log_params(
        ckpt_tag: str,
        transform_name: str,
        batch_idx: int,
        batch_size: int,
        meta: Dict[str, object],
    ) -> None:
        if len(param_rows) >= args.max_param_rows:
            return
        row = {
            "checkpoint": ckpt_tag,
            "transform": transform_name,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "severity_id": meta.get("severity_id", 0),
            "rho": meta.get("rho", 1.0),
            "g_db": meta.get("g_db", 0.0),
            "b_bins": meta.get("b_bins", 0),
            "seed": meta.get("seed", args.seed),
            "shift_fill": meta.get("shift_fill", "na"),
            "run_id": meta.get("run_id", ""),
            "checkpoint_path": meta.get("checkpoint_path", ""),
            "split": args.split,
        }
        param_rows.append(row)

    for run_index, ((checkpoint, config, state), ckpt_path, ckpt_tag, supervised_views) in enumerate(
        zip(loaded, checkpoint_paths, labels, supervised_views_all)
    ):
        run_id = f"{run_index:03d}:{ckpt_path.stem}"
        model, _ = _build_model(
            config,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device,
            input_dim_freq=input_dim_freq,
            input_dim_tf=input_dim_tf,
        )
        model.load_state_dict(state, strict=True)

        baseline = evaluate_classifier(
            model,
            loader,
            device=device,
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            supervised_views=supervised_views,
            preprocess_config=preprocess_config,
        )
        summary_rows.append(
            {
                "checkpoint": ckpt_tag,
                "transform": "clean",
                "severity_id": 0,
                "rho": 1.0,
                "g_db": 0.0,
                "b_bins": 0,
                "trial": 0,
                "acc": baseline["acc"],
                "mf1": baseline["mf1"],
                "loss": baseline["loss"],
                "count": baseline["count"],
                "split": args.split,
                "severity_source": args.severity_source,
                "shift_fill": "na",
                "run_id": run_id,
                "checkpoint_path": str(ckpt_path),
            }
        )

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
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt_path),
                }

            metrics = evaluate_classifier(
                model,
                loader,
                device=device,
                label_smoothing=float(config.get("label_smoothing", 0.0)),
                transform_fn=_shift_transform,
                batch_logger=lambda batch_idx, batch_size, meta, ck=ckpt_tag: _maybe_log_params(
                    ck, "shift", batch_idx, batch_size, meta or {}
                ),
                supervised_views=supervised_views,
                preprocess_config=preprocess_config,
            )
            summary_rows.append(
                {
                    "checkpoint": ckpt_tag,
                    "transform": "shift",
                    "severity_id": severity_id,
                    "rho": 1.0,
                    "g_db": 0.0,
                    "b_bins": bins,
                    "trial": 0,
                    "acc": metrics["acc"],
                    "mf1": metrics["mf1"],
                    "loss": metrics["loss"],
                    "count": metrics["count"],
                    "split": args.split,
                    "severity_source": args.severity_source,
                    "shift_fill": args.shift_fill,
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt_path),
                }
            )

        for severity_id, ratio in enumerate(scale_ratios):
            def _scale_transform(x: torch.Tensor, _batch_idx: int, r: float = ratio, sid: int = severity_id):
                out = apply_per_sample_channel(x, lambda s: frequency_scale_time(s, r))
                return out, {
                    "severity_id": sid,
                    "rho": r,
                    "g_db": 0.0,
                    "b_bins": 0,
                    "seed": args.seed,
                    "shift_fill": "na",
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt_path),
                }

            metrics = evaluate_classifier(
                model,
                loader,
                device=device,
                label_smoothing=float(config.get("label_smoothing", 0.0)),
                transform_fn=_scale_transform,
                batch_logger=lambda batch_idx, batch_size, meta, ck=ckpt_tag: _maybe_log_params(
                    ck, "scale", batch_idx, batch_size, meta or {}
                ),
                supervised_views=supervised_views,
                preprocess_config=preprocess_config,
            )
            summary_rows.append(
                {
                    "checkpoint": ckpt_tag,
                    "transform": "scale",
                    "severity_id": severity_id,
                    "rho": ratio,
                    "g_db": 0.0,
                    "b_bins": 0,
                    "trial": 0,
                    "acc": metrics["acc"],
                    "mf1": metrics["mf1"],
                    "loss": metrics["loss"],
                    "count": metrics["count"],
                    "split": args.split,
                    "severity_source": args.severity_source,
                    "shift_fill": "na",
                    "run_id": run_id,
                    "checkpoint_path": str(ckpt_path),
                }
            )

        for severity_id, max_db in enumerate(color_levels):
            trial_metrics = []
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
                    current_seed: int = trial_seed,
                    db: float = max_db,
                ):
                    out = apply_per_sample_channel(x, lambda s: spectral_coloring(s, g))
                    return out, {
                        "severity_id": sid,
                        "rho": 1.0,
                        "g_db": db,
                        "b_bins": 0,
                        "seed": current_seed,
                        "shift_fill": "na",
                        "run_id": run_id,
                        "checkpoint_path": str(ckpt_path),
                    }

                metrics = evaluate_classifier(
                    model,
                    loader,
                    device=device,
                    label_smoothing=float(config.get("label_smoothing", 0.0)),
                    transform_fn=_color_transform,
                    batch_logger=lambda batch_idx, batch_size, meta, ck=ckpt_tag: _maybe_log_params(
                        ck, "color", batch_idx, batch_size, meta or {}
                    ),
                    supervised_views=supervised_views,
                    preprocess_config=preprocess_config,
                )
                trial_metrics.append(metrics)
                summary_rows.append(
                    {
                        "checkpoint": ckpt_tag,
                        "transform": "color",
                        "severity_id": severity_id,
                        "rho": 1.0,
                        "g_db": max_db,
                        "b_bins": 0,
                        "trial": trial,
                        "acc": metrics["acc"],
                        "mf1": metrics["mf1"],
                        "loss": metrics["loss"],
                        "count": metrics["count"],
                        "split": args.split,
                        "severity_source": args.severity_source,
                        "shift_fill": "na",
                        "run_id": run_id,
                        "checkpoint_path": str(ckpt_path),
                    }
                )
            if color_trials > 1:
                summary_rows.append(
                    {
                        "checkpoint": ckpt_tag,
                        "transform": "color",
                        "severity_id": severity_id,
                        "rho": 1.0,
                        "g_db": max_db,
                        "b_bins": 0,
                        "trial": "mean",
                        "acc": sum(m["acc"] for m in trial_metrics) / len(trial_metrics),
                        "mf1": sum(m["mf1"] for m in trial_metrics) / len(trial_metrics),
                        "loss": sum(m["loss"] for m in trial_metrics) / len(trial_metrics),
                        "count": sum(m["count"] for m in trial_metrics) / len(trial_metrics),
                        "split": args.split,
                        "severity_source": args.severity_source,
                        "shift_fill": "na",
                        "run_id": run_id,
                        "checkpoint_path": str(ckpt_path),
                    }
                )

        if args.enable_mixed and mixed_shift_bins and mixed_color_levels:
            for shift_id, bins in enumerate(mixed_shift_bins):
                for color_id, max_db in enumerate(mixed_color_levels):
                    severity_id = shift_id * len(mixed_color_levels) + color_id
                    if safe_mixed_ids and severity_id not in safe_mixed_ids:
                        continue
                    trial_metrics = []
                    for trial in range(mixed_trials):
                        trial_seed = args.seed + severity_id * 10000 + trial
                        gains = make_coloring_gains(
                            num_bins=num_bins,
                            bands=args.color_bands,
                            max_gain_db=max_db,
                            active_bands=color_active_bands if color_active_bands else None,
                            generator=torch.Generator().manual_seed(trial_seed),
                        )

                        def _mixed_transform(
                            x: torch.Tensor,
                            _batch_idx: int,
                            b: float = bins,
                            g: torch.Tensor = gains,
                            sid: int = severity_id,
                            current_seed: int = trial_seed,
                            db: float = max_db,
                        ):
                            if args.shift_mode == "stft":
                                x_shift = apply_per_sample_channel(
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
                                x_shift = apply_per_sample_channel(x, lambda s: band_shift_time(s, b, shift_mode=args.shift_fill))
                            out = apply_per_sample_channel(x_shift, lambda s: spectral_coloring(s, g))
                            return out, {
                                "severity_id": sid,
                                "rho": 1.0,
                                "g_db": db,
                                "b_bins": b,
                                "seed": current_seed,
                                "shift_fill": args.shift_fill,
                                "run_id": run_id,
                                "checkpoint_path": str(ckpt_path),
                            }

                        metrics = evaluate_classifier(
                            model,
                            loader,
                            device=device,
                            label_smoothing=float(config.get("label_smoothing", 0.0)),
                            transform_fn=_mixed_transform,
                            batch_logger=lambda batch_idx, batch_size, meta, ck=ckpt_tag: _maybe_log_params(
                                ck, "mixed_shift_color", batch_idx, batch_size, meta or {}
                            ),
                            supervised_views=supervised_views,
                            preprocess_config=preprocess_config,
                        )
                        trial_metrics.append(metrics)
                        summary_rows.append(
                            {
                                "checkpoint": ckpt_tag,
                                "transform": "mixed_shift_color",
                                "severity_id": severity_id,
                                "rho": 1.0,
                                "g_db": max_db,
                                "b_bins": bins,
                                "trial": trial,
                                "acc": metrics["acc"],
                                "mf1": metrics["mf1"],
                                "loss": metrics["loss"],
                                "count": metrics["count"],
                                "split": args.split,
                                "severity_source": args.severity_source,
                                "shift_fill": args.shift_fill,
                                "run_id": run_id,
                                "checkpoint_path": str(ckpt_path),
                            }
                        )
                    if mixed_trials > 1 and trial_metrics:
                        summary_rows.append(
                            {
                                "checkpoint": ckpt_tag,
                                "transform": "mixed_shift_color",
                                "severity_id": severity_id,
                                "rho": 1.0,
                                "g_db": max_db,
                                "b_bins": bins,
                                "trial": "mean",
                                "acc": sum(m["acc"] for m in trial_metrics) / len(trial_metrics),
                                "mf1": sum(m["mf1"] for m in trial_metrics) / len(trial_metrics),
                                "loss": sum(m["loss"] for m in trial_metrics) / len(trial_metrics),
                                "count": sum(m["count"] for m in trial_metrics) / len(trial_metrics),
                                "split": args.split,
                                "severity_source": args.severity_source,
                                "shift_fill": args.shift_fill,
                                "run_id": run_id,
                                "checkpoint_path": str(ckpt_path),
                            }
                        )

    n_cols = 4 if args.enable_mixed else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.5))
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]
    ckpt_tags = sorted({row["checkpoint"] for row in summary_rows})
    shift_interval = _interval_from_values(safe_ranges["shift"])
    scale_interval = _interval_from_values(safe_ranges["scale"])
    color_interval = _interval_from_values(safe_ranges["color"])
    mixed_interval = _interval_from_values(safe_ranges["mixed_shift_color"])
    if shift_interval is not None:
        axes[0].axvspan(shift_interval[0], shift_interval[1], color="tab:green", alpha=0.12, label="safe range")
    if scale_interval is not None:
        axes[1].axvspan(scale_interval[0], scale_interval[1], color="tab:green", alpha=0.12, label="safe range")
    if color_interval is not None:
        axes[2].axvspan(color_interval[0], color_interval[1], color="tab:green", alpha=0.12, label="safe range")
    if args.enable_mixed and mixed_interval is not None:
        axes[3].axvspan(mixed_interval[0], mixed_interval[1], color="tab:green", alpha=0.12, label="safe range")

    robustness_rows = []
    interval_by_transform = {
        "shift": shift_interval,
        "scale": scale_interval,
        "color": color_interval,
        "mixed_shift_color": mixed_interval,
    }
    transform_order = ["shift", "scale", "color"] + (["mixed_shift_color"] if args.enable_mixed else [])
    for ckpt_tag in ckpt_tags:
        shift_x, shift_mean, shift_std, shift_runs = _aggregate_curve_by_group(summary_rows, ckpt_tag, "shift")
        if len(shift_x) <= 1:
            axes[0].scatter(shift_x, shift_mean, marker="o", label=ckpt_tag)
        else:
            axes[0].plot(shift_x, shift_mean, marker="o", label=ckpt_tag)
            if shift_runs > 1:
                lower = [m - s for m, s in zip(shift_mean, shift_std)]
                upper = [m + s for m, s in zip(shift_mean, shift_std)]
                axes[0].fill_between(shift_x, lower, upper, alpha=0.18)

        scale_x, scale_mean, scale_std, scale_runs = _aggregate_curve_by_group(summary_rows, ckpt_tag, "scale")
        if len(scale_x) <= 1:
            axes[1].scatter(scale_x, scale_mean, marker="o", label=ckpt_tag)
        else:
            axes[1].plot(scale_x, scale_mean, marker="o", label=ckpt_tag)
            if scale_runs > 1:
                lower = [m - s for m, s in zip(scale_mean, scale_std)]
                upper = [m + s for m, s in zip(scale_mean, scale_std)]
                axes[1].fill_between(scale_x, lower, upper, alpha=0.18)

        color_x, color_mean, color_std, color_runs = _aggregate_curve_by_group(summary_rows, ckpt_tag, "color")
        if len(color_x) <= 1:
            axes[2].scatter(color_x, color_mean, marker="o", label=ckpt_tag)
        else:
            axes[2].plot(color_x, color_mean, marker="o", label=ckpt_tag)
            if color_runs > 1:
                lower = [m - s for m, s in zip(color_mean, color_std)]
                upper = [m + s for m, s in zip(color_mean, color_std)]
                axes[2].fill_between(color_x, lower, upper, alpha=0.18)

        if args.enable_mixed:
            mixed_x, mixed_mean, mixed_std, mixed_runs = _aggregate_curve_by_group(
                summary_rows, ckpt_tag, "mixed_shift_color"
            )
            if len(mixed_x) <= 1:
                axes[3].scatter(mixed_x, mixed_mean, marker="o", label=ckpt_tag)
            else:
                axes[3].plot(mixed_x, mixed_mean, marker="o", label=ckpt_tag)
                if mixed_runs > 1:
                    lower = [m - s for m, s in zip(mixed_mean, mixed_std)]
                    upper = [m + s for m, s in zip(mixed_mean, mixed_std)]
                    axes[3].fill_between(mixed_x, lower, upper, alpha=0.18)

        for transform_name in transform_order:
            per_run = _run_level_metrics(summary_rows, ckpt_tag, transform_name)
            if not per_run:
                continue
            metric_names = [
                "clean_acc",
                "avg_acc",
                "worst_acc",
                "drop_avg_acc",
                "drop_worst_acc",
                "rauc_acc",
                "slope_acc",
                "clean_mf1",
                "avg_mf1",
                "worst_mf1",
                "drop_avg_mf1",
                "drop_worst_mf1",
                "rauc_mf1",
                "slope_mf1",
            ]
            row = {
                "checkpoint": ckpt_tag,
                "transform": transform_name,
                "n_runs": len(per_run),
                "safe_low": interval_by_transform[transform_name][0] if interval_by_transform[transform_name] is not None else "",
                "safe_high": interval_by_transform[transform_name][1] if interval_by_transform[transform_name] is not None else "",
            }
            for metric_name in metric_names:
                values = [float(item[metric_name]) for item in per_run]
                values_t = torch.tensor(values, dtype=torch.float32)
                row[f"{metric_name}_mean"] = float(values_t.mean().item())
                row[f"{metric_name}_std"] = float(values_t.std(unbiased=False).item()) if len(values) > 1 else 0.0
            robustness_rows.append(row)

    axes[0].set_title("Shift Robustness")
    axes[0].set_xlabel("b (shift bins)")
    axes[0].set_ylabel("ACC")
    axes[1].set_title("Scale Robustness")
    axes[1].set_xlabel("rho (scale)")
    axes[1].set_ylabel("ACC")
    axes[2].set_title("Color Robustness")
    axes[2].set_xlabel("g (max dB)")
    axes[2].set_ylabel("ACC")
    if args.enable_mixed:
        axes[3].set_title("Mixed Robustness")
        axes[3].set_xlabel("mixed severity id")
        axes[3].set_ylabel("ACC")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    if args.safe_shift_csv.strip() and not any(abs(float(b)) > 1e-8 for b in shift_bins):
        axes[0].text(
            0.02,
            0.04,
            "No safe shift found under threshold.",
            transform=axes[0].transAxes,
            fontsize=8,
            color="tab:red",
            ha="left",
            va="bottom",
        )
    if len({row.get("run_id") for row in summary_rows}) > len(ckpt_tags):
        fig.text(0.5, 0.01, "mean+/-std over repeated runs (same label)", ha="center", fontsize=9)
    plt.tight_layout()

    output_root = Path(args.output_root)
    figs_dir, csv_dir = ensure_output_dirs(output_root)
    stft_hash = preprocess_config.hash()
    sev_hash = stable_hash(
        {
            "shift_bins": shift_bins,
            "scale_ratios": scale_ratios,
            "color_max_db": color_levels,
            "enable_mixed": args.enable_mixed,
            "mixed_shift_bins": mixed_shift_bins if args.enable_mixed else [],
            "mixed_color_max_db": mixed_color_levels if args.enable_mixed else [],
            "mixed_trials": mixed_trials if args.enable_mixed else 0,
            "shift_mode": args.shift_mode,
            "safe_shift_source": safe_shift_source,
            "safe_agreement_source": safe_agreement_source,
            "safe_signal_source": safe_signal_source,
            "safe_agreement_threshold": args.safe_agreement_threshold,
            "color_bands": args.color_bands,
            "color_active_bands": color_active_bands,
            "color_trials": color_trials,
        }
    )
    stem = build_tag(
        "sweep",
        dataset_name,
        args.split,
        f"seed{args.seed}",
        f"stft{stft_hash}",
        f"sh{len(shift_bins)}-sc{len(scale_ratios)}-co{len(color_levels)}-mx{len(mixed_shift_bins) * len(mixed_color_levels) if args.enable_mixed else 0}",
        f"sev{sev_hash}",
    )
    fig_path = figs_dir / f"{stem}.png"
    summary_csv = csv_dir / f"{stem}_summary.csv"
    params_csv = csv_dir / f"{stem}_params.csv"
    robustness_csv = csv_dir / f"{stem}_robustness_metrics.csv"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    write_csv(summary_csv, summary_rows)
    write_csv(params_csv, param_rows)
    write_csv(robustness_csv, robustness_rows)

    meta_path = write_run_meta(
        output_root=output_root,
        script_name="scripts/sweep_transforms.py",
        device=device,
        config=vars(args),
        extra={
            "figure": str(fig_path),
            "summary_csv": str(summary_csv),
            "params_csv": str(params_csv),
            "robustness_csv": str(robustness_csv),
            "labels": labels,
            "stft_hash": stft_hash,
            "severity_hash": sev_hash,
            "shift_fill": args.shift_fill,
            "safe_shift_source": safe_shift_source,
            "safe_agreement_source": safe_agreement_source,
            "safe_signal_source": safe_signal_source,
            "safe_agreement_threshold": args.safe_agreement_threshold,
            "mixed_enabled": args.enable_mixed,
            "mixed_shift_bins": mixed_shift_bins if args.enable_mixed else [],
            "mixed_color_max_db": mixed_color_levels if args.enable_mixed else [],
            "mixed_trials": mixed_trials if args.enable_mixed else 0,
            "fair_budget_keys": list(_BUDGET_KEYS),
            "leakage_guard": "No severity tuning is performed on split=test.",
        },
    )
    print(f"saved_figure={fig_path}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_params_csv={params_csv}")
    print(f"saved_robustness_csv={robustness_csv}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()

