import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


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


def _x_label_for_transform(transform_name: str) -> str:
    if transform_name == "shift":
        return "b (shift bins)"
    if transform_name == "scale":
        return "rho (scale)"
    if transform_name == "color":
        return "g (max dB)"
    if transform_name == "mixed_shift_color":
        return "mixed severity id (shift+color)"
    return "severity"


def _x_value(row: Dict[str, object], transform_name: str) -> float:
    x_key = _x_key_for_transform(transform_name)
    if x_key is None:
        return 0.0
    default = 1.0 if x_key == "rho" else 0.0
    return _safe_float(row.get(x_key, default), default)


def _preferred_row(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    mean_row = next((r for r in candidates if str(r.get("trial")) == "mean"), None)
    if mean_row is not None:
        return mean_row
    zero_row = next((r for r in candidates if str(r.get("trial")) in {"0", "0.0"}), None)
    if zero_row is not None:
        return zero_row
    return candidates[0]


def _curve_points(rows: List[Dict[str, object]], transform_name: str) -> List[Dict[str, object]]:
    per_x: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        x_val = _x_value(row, transform_name)
        per_x.setdefault(x_val, []).append(row)
    ordered = []
    for x_val in sorted(per_x.keys()):
        ordered.append(_preferred_row(per_x[x_val]))
    return ordered


def _aggregate_metric_curve(
    rows: List[Dict[str, object]],
    method: str,
    transform_name: str,
    metric_name: str,
) -> Tuple[List[float], List[float], List[float], int]:
    rows_group = [
        row
        for row in rows
        if str(row.get("checkpoint", "")) == method and str(row.get("transform", "")) == transform_name
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
            x_to_values.setdefault(x_val, []).append(_safe_float(point.get(metric_name, 0.0), 0.0))

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


def _parse_list(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_label_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in _parse_list(raw):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        key = k.strip()
        val = v.strip()
        if key and val:
            mapping[key] = val
    return mapping


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _safe_key(transform_name: str, value: float):
    if transform_name == "mixed_shift_color":
        return int(round(value))
    return round(float(value), 6)


def _load_safe_values(path: Path) -> Dict[str, List[float]]:
    safe: Dict[str, set] = {"shift": set(), "scale": set(), "color": set(), "mixed_shift_color": set()}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            transform_name = str(row.get("transform", "")).strip()
            if transform_name not in safe:
                continue
            if "safe" not in row:
                continue
            is_safe = int(_safe_float(row.get("safe", 0.0), 0.0)) == 1
            if not is_safe:
                continue
            if transform_name == "shift":
                safe["shift"].add(_safe_key("shift", _safe_float(row.get("b_bins", 0.0), 0.0)))
            elif transform_name == "scale":
                safe["scale"].add(_safe_key("scale", _safe_float(row.get("rho", 1.0), 1.0)))
            elif transform_name == "color":
                safe["color"].add(_safe_key("color", _safe_float(row.get("g_db", 0.0), 0.0)))
            elif transform_name == "mixed_shift_color":
                safe["mixed_shift_color"].add(_safe_key("mixed_shift_color", _safe_float(row.get("severity_id", 0.0), 0.0)))
    return {k: sorted(list(v)) for k, v in safe.items()}


def _filter_rows_by_safe(
    rows: List[Dict[str, object]],
    safe_values: Dict[str, List[float]],
    policy: str,
) -> List[Dict[str, object]]:
    safe_sets = {k: set(v) for k, v in safe_values.items()}
    out: List[Dict[str, object]] = []
    for row in rows:
        transform_name = str(row.get("transform", ""))
        if transform_name not in safe_sets:
            out.append(row)
            continue
        safe_set = safe_sets[transform_name]
        if not safe_set:
            if policy == "fallback":
                out.append(row)
            continue
        x_val = _x_value(row, transform_name)
        if _safe_key(transform_name, x_val) in safe_set:
            out.append(row)
    return out


def _format_values(values: List[float], transform_name: str) -> str:
    if not values:
        return "none"
    if transform_name == "mixed_shift_color":
        return ", ".join(str(int(round(v))) for v in sorted(values))
    return ", ".join(f"{v:.2f}" for v in sorted(values))


def _collect_used_values(rows: List[Dict[str, object]], transform_name: str) -> List[float]:
    vals = {_safe_key(transform_name, _x_value(row, transform_name)) for row in rows if str(row.get("transform", "")) == transform_name}
    if transform_name == "mixed_shift_color":
        return [float(int(v)) for v in sorted(vals)]
    return [float(v) for v in sorted(vals)]


def _auto_ylim(values: List[float], pad: float = 0.04, min_span: float = 0.16) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo = max(0.0, min(values) - pad)
    hi = min(1.0, max(values) + pad)
    if hi - lo < min_span:
        center = 0.5 * (hi + lo)
        lo = max(0.0, center - 0.5 * min_span)
        hi = min(1.0, center + 0.5 * min_span)
    return lo, hi


def _plot_factor_main(
    rows: List[Dict[str, object]],
    methods: List[str],
    method_label_map: Dict[str, str],
    out_path: Path,
    curve_rows: List[Dict[str, object]],
    safe_label: str,
    used_values: Dict[str, List[float]],
    annotate_safe_values: bool,
) -> None:
    transforms = ["shift", "scale", "color"]
    titles = {
        "shift": f"Shift Robustness ({safe_label})",
        "scale": f"Scale Robustness ({safe_label})",
        "color": f"Color Robustness ({safe_label})",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)

    for method_idx, method in enumerate(methods):
        color = palette[method_idx % len(palette)]
        for ax, transform_name in zip(axes, transforms):
            for metric_name, linestyle, marker in (("acc", "-", "o"), ("mf1", "--", "^")):
                xs, means, stds, n_runs = _aggregate_metric_curve(rows, method, transform_name, metric_name)
                if not xs:
                    continue
                if len(xs) <= 1:
                    ax.scatter(xs, means, color=color, marker=marker, s=28)
                else:
                    ax.plot(xs, means, color=color, linestyle=linestyle, marker=marker, linewidth=1.8, markersize=5)
                if n_runs > 1 and len(xs) > 1:
                    lower = [m - s for m, s in zip(means, stds)]
                    upper = [m + s for m, s in zip(means, stds)]
                    ax.fill_between(xs, lower, upper, color=color, alpha=0.08)
                for x, m, s in zip(xs, means, stds):
                    curve_rows.append(
                        {
                            "figure": "factor_main",
                            "transform": transform_name,
                            "metric": metric_name,
                            "checkpoint": method,
                            "label": method_label_map.get(method, method),
                            "x": x,
                            "mean": m,
                            "std": s,
                            "n_runs": n_runs,
                        }
                    )

    for ax, transform_name in zip(axes, transforms):
        ax.set_title(titles[transform_name])
        ax.set_xlabel(_x_label_for_transform(transform_name))
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        if transform_name == "color":
            ax.set_xlim(left=0.0)
        if annotate_safe_values:
            ax.text(
                0.02,
                0.96,
                f"used {transform_name}: {_format_values(used_values.get(transform_name, []), transform_name)}",
                transform=ax.transAxes,
                fontsize=8,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
            )
        ax.grid(True, alpha=0.3)

    method_handles = []
    for method_idx, method in enumerate(methods):
        color = palette[method_idx % len(palette)]
        method_handles.append(Line2D([0], [0], color=color, lw=2.2, label=method_label_map.get(method, method)))
    metric_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", marker="o", label="ACC"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", marker="^", label="MF1"),
    ]
    if method_handles:
        legend_methods = fig.legend(
            handles=method_handles,
            loc="upper center",
            bbox_to_anchor=(0.45, 1.06),
            ncol=min(4, max(1, len(method_handles))),
            frameon=False,
            title="Methods",
        )
        fig.add_artist(legend_methods)
    fig.legend(
        handles=metric_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.06),
        ncol=2,
        frameon=False,
        title="Metrics",
    )
    fig.text(
        0.5,
        0.005,
        f"{safe_label} severities only; curves show degradation across selected severities.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.91])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_mixed_mapping(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    mapping: Dict[int, Dict[str, object]] = {}
    for row in rows:
        if str(row.get("transform", "")) != "mixed_shift_color":
            continue
        sid = int(round(_safe_float(row.get("severity_id", 0.0), 0.0)))
        if sid in mapping:
            continue
        mapping[sid] = {
            "severity_id": sid,
            "b_bins": _safe_float(row.get("b_bins", 0.0), 0.0),
            "g_db": _safe_float(row.get("g_db", 0.0), 0.0),
        }
    return [mapping[k] for k in sorted(mapping.keys())]


def _plot_mixed_main(
    rows: List[Dict[str, object]],
    methods: List[str],
    method_label_map: Dict[str, str],
    out_path: Path,
    curve_rows: List[Dict[str, object]],
    safe_label: str,
    used_values: Dict[str, List[float]],
    mixed_ids_location: str,
) -> bool:
    has_mixed = any(str(row.get("transform", "")) == "mixed_shift_color" for row in rows)
    if not has_mixed:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), sharex=True)
    ax_acc, ax_mf1 = axes
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    metric_to_ax = {"acc": ax_acc, "mf1": ax_mf1}
    metric_values: Dict[str, List[float]] = {"acc": [], "mf1": []}

    for method_idx, method in enumerate(methods):
        color = palette[method_idx % len(palette)]
        for metric_name, linestyle, marker, lw in (("acc", "-", "o", 2.3), ("mf1", "--", "^", 2.0)):
            xs, means, stds, n_runs = _aggregate_metric_curve(rows, method, "mixed_shift_color", metric_name)
            if not xs:
                continue
            ax = metric_to_ax[metric_name]
            if len(xs) <= 1:
                ax.scatter(xs, means, color=color, marker=marker, s=28)
            else:
                ax.plot(xs, means, color=color, linestyle=linestyle, marker=marker, linewidth=lw, markersize=5.8)
            if n_runs > 1 and len(xs) > 1:
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.08)
            metric_values[metric_name].extend(means)
            for x, m, s in zip(xs, means, stds):
                curve_rows.append(
                    {
                        "figure": "mixed_main",
                        "transform": "mixed_shift_color",
                        "metric": metric_name,
                        "checkpoint": method,
                        "label": method_label_map.get(method, method),
                        "x": x,
                        "mean": m,
                        "std": s,
                        "n_runs": n_runs,
                    }
                )

    fig.suptitle(f"Mixed Robustness ({safe_label}): Shift + Color", y=0.98, fontsize=15)
    ax_acc.set_title("ACC", fontsize=12)
    ax_mf1.set_title("MF1", fontsize=12)
    for ax in (ax_acc, ax_mf1):
        ax.set_xlabel(_x_label_for_transform("mixed_shift_color"))
        ax.grid(True, alpha=0.25)
    ax_acc.set_ylabel("Score")
    acc_lo, acc_hi = _auto_ylim(metric_values["acc"])
    mf1_lo, mf1_hi = _auto_ylim(metric_values["mf1"])
    ax_acc.set_ylim(acc_lo, acc_hi)
    ax_mf1.set_ylim(mf1_lo, mf1_hi)
    if mixed_ids_location == "plot":
        ax_acc.text(
            0.02,
            0.96,
            f"used mixed ids: {_format_values(used_values.get('mixed_shift_color', []), 'mixed_shift_color')}",
            transform=ax_acc.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
        )

    method_handles = []
    for method_idx, method in enumerate(methods):
        color = palette[method_idx % len(palette)]
        method_handles.append(Line2D([0], [0], color=color, lw=2.2, label=method_label_map.get(method, method)))
    if method_handles:
        methods_legend = ax_mf1.legend(
            handles=method_handles,
            loc="upper right",
            frameon=True,
            fontsize=9,
            title="Methods",
        )
        methods_legend.get_frame().set_alpha(0.86)
    if mixed_ids_location == "caption":
        caption = "Mixed ids denote predefined shift-color perturbation combinations."
    elif mixed_ids_location == "none":
        caption = "Mixed ids denote predefined shift-color perturbation combinations."
    else:
        caption = "Mixed ids denote predefined shift-color perturbation combinations."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.92])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def _plot_appendix_quad_acc(
    rows: List[Dict[str, object]],
    methods: List[str],
    method_label_map: Dict[str, str],
    out_path: Path,
) -> bool:
    transforms = ["shift", "scale", "color"]
    has_mixed = any(str(row.get("transform", "")) == "mixed_shift_color" for row in rows)
    if has_mixed:
        transforms.append("mixed_shift_color")
    n_cols = len(transforms)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.3))
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    else:
        axes = [axes]

    titles = {
        "shift": "Shift Robustness (ACC, Full Sweep)",
        "scale": "Scale Robustness (ACC, Full Sweep)",
        "color": "Color Robustness (ACC, Full Sweep)",
        "mixed_shift_color": "Mixed Robustness (ACC, Full Sweep)",
    }
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)

    for method_idx, method in enumerate(methods):
        color = palette[method_idx % len(palette)]
        for ax, transform_name in zip(axes, transforms):
            xs, means, stds, n_runs = _aggregate_metric_curve(rows, method, transform_name, "acc")
            if not xs:
                continue
            label = method_label_map.get(method, method)
            if len(xs) <= 1:
                ax.scatter(xs, means, marker="o", color=color, label=label)
            else:
                ax.plot(xs, means, marker="o", color=color, label=label)
                if n_runs > 1:
                    lower = [m - s for m, s in zip(means, stds)]
                    upper = [m + s for m, s in zip(means, stds)]
                    ax.fill_between(xs, lower, upper, color=color, alpha=0.10)

    for ax, transform_name in zip(axes, transforms):
        ax.set_title(titles[transform_name])
        ax.set_xlabel(_x_label_for_transform(transform_name))
        ax.set_ylabel("ACC")
        if transform_name == "color":
            ax.set_xlim(left=0.0)
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Appendix figure: ACC-only full evaluated sweep (not protocol-locked safe filtering).",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to sweep_*_summary.csv")
    parser.add_argument(
        "--safe-summary-csv",
        type=str,
        default="",
        help="Optional safe *_summary.csv with safe=1 flags; when set, plotting strictly follows selected severities.",
    )
    parser.add_argument(
        "--safe-policy",
        type=str,
        default="strict",
        choices=["strict", "fallback"],
        help="strict: keep only safe points per transform; fallback: keep all points when a transform has no safe points.",
    )
    parser.add_argument(
        "--safe-label",
        type=str,
        default="Safe-B",
        help="Label shown in figure titles when --safe-summary-csv is provided.",
    )
    parser.add_argument("--methods", type=str, default="", help="Optional ordered methods: m1,m2,m3")
    parser.add_argument(
        "--method-labels",
        type=str,
        default="",
        help="Optional label mapping: raw1=Pretty1,raw2=Pretty2",
    )
    parser.add_argument("--annotate-safe-values", action="store_true", default=True)
    parser.add_argument("--no-annotate-safe-values", dest="annotate_safe_values", action="store_false")
    parser.add_argument(
        "--min-points-per-transform",
        type=int,
        default=2,
        help="Warn if selected severities for shift/scale/color are fewer than this value.",
    )
    parser.add_argument(
        "--mixed-ids-location",
        type=str,
        default="none",
        choices=["caption", "plot", "none"],
        help="Where to show used mixed ids for mixed plot.",
    )
    parser.add_argument("--appendix-quad", action="store_true", default=True)
    parser.add_argument("--no-appendix-quad", dest="appendix_quad", action="store_false")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--factor-name", type=str, default="factor_main")
    parser.add_argument("--mixed-name", type=str, default="mixed_main")
    parser.add_argument("--appendix-quad-name", type=str, default="appendix_quad_acc")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        rows_all = list(csv.DictReader(f))
    if not rows_all:
        raise ValueError(f"summary csv is empty: {summary_path}")

    safe_summary_path: Optional[Path] = None
    rows_plot = rows_all
    safe_values = {"shift": [], "scale": [], "color": [], "mixed_shift_color": []}
    if args.safe_summary_csv.strip():
        safe_summary_path = Path(args.safe_summary_csv)
        if not safe_summary_path.exists():
            raise FileNotFoundError(f"safe summary csv not found: {safe_summary_path}")
        safe_values = _load_safe_values(safe_summary_path)
        rows_plot = _filter_rows_by_safe(rows_all, safe_values=safe_values, policy=args.safe_policy)

    safe_label = args.safe_label if safe_summary_path is not None else "Evaluated"
    used_values = {
        "shift": _collect_used_values(rows_plot, "shift"),
        "scale": _collect_used_values(rows_plot, "scale"),
        "color": _collect_used_values(rows_plot, "color"),
        "mixed_shift_color": _collect_used_values(rows_plot, "mixed_shift_color"),
    }
    for transform_name in ["shift", "scale", "color"]:
        if len(used_values[transform_name]) < max(1, int(args.min_points_per_transform)):
            print(
                f"warning={transform_name} has only {len(used_values[transform_name])} selected points under {safe_label}; "
                f"used={_format_values(used_values[transform_name], transform_name)}"
            )

    available_methods = sorted(
        {
            str(row.get("checkpoint", "")).strip()
            for row in rows_plot
            if str(row.get("checkpoint", "")).strip() and str(row.get("transform", "")) in {"shift", "scale", "color", "mixed_shift_color"}
        }
    )
    if not available_methods:
        raise ValueError("No methods found in summary csv.")

    requested = _parse_list(args.methods)
    if requested:
        missing = [m for m in requested if m not in available_methods]
        if missing:
            raise ValueError(f"Requested methods not found in summary: {missing}")
        methods = requested
    else:
        methods = available_methods

    method_label_map = _parse_label_map(args.method_labels)
    output_root = Path(args.output_root)
    figs_dir = output_root / "figs"
    csv_dir = output_root / "csv"
    figs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    stem = summary_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    factor_path = figs_dir / f"{stem}_{args.factor_name}.png"
    mixed_path = figs_dir / f"{stem}_{args.mixed_name}.png"
    appendix_quad_path = figs_dir / f"{stem}_{args.appendix_quad_name}.png"
    curves_csv = csv_dir / f"{stem}_paper_curve_points.csv"
    mixed_map_csv = csv_dir / f"{stem}_mixed_severity_map.csv"
    protocol_csv = csv_dir / f"{stem}_paper_protocol_used.csv"

    curve_rows: List[Dict[str, object]] = []
    _plot_factor_main(
        rows_plot,
        methods,
        method_label_map,
        factor_path,
        curve_rows,
        safe_label=safe_label,
        used_values=used_values,
        annotate_safe_values=args.annotate_safe_values,
    )
    has_mixed = _plot_mixed_main(
        rows_plot,
        methods,
        method_label_map,
        mixed_path,
        curve_rows,
        safe_label=safe_label,
        used_values=used_values,
        mixed_ids_location=args.mixed_ids_location,
    )
    has_appendix_quad = False
    if args.appendix_quad:
        has_appendix_quad = _plot_appendix_quad_acc(
            rows_all,
            methods,
            method_label_map,
            appendix_quad_path,
        )
    _write_csv(curves_csv, curve_rows)
    _write_csv(mixed_map_csv, _build_mixed_mapping(rows_plot))
    protocol_rows = []
    for transform_name in ["shift", "scale", "color", "mixed_shift_color"]:
        protocol_rows.append(
            {
                "transform": transform_name,
                "n_used_values": len(used_values.get(transform_name, [])),
                "used_values": _format_values(used_values.get(transform_name, []), transform_name),
                "safe_label": safe_label,
                "safe_policy": args.safe_policy if safe_summary_path is not None else "off",
                "safe_summary_csv": str(safe_summary_path) if safe_summary_path is not None else "",
                "safe_values_from_summary": _format_values(safe_values.get(transform_name, []), transform_name),
            }
        )
    _write_csv(protocol_csv, protocol_rows)

    print(f"saved_factor_figure={factor_path}")
    if has_mixed:
        print(f"saved_mixed_figure={mixed_path}")
    else:
        print("saved_mixed_figure=NA (no mixed_shift_color rows in summary)")
    print(f"saved_curve_points_csv={curves_csv}")
    print(f"saved_mixed_map_csv={mixed_map_csv}")
    print(f"saved_protocol_csv={protocol_csv}")
    if has_appendix_quad:
        print(f"saved_appendix_quad_figure={appendix_quad_path}")


if __name__ == "__main__":
    main()
