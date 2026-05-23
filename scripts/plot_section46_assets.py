import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _preferred_trial(rows: List[Dict[str, str]]) -> Dict[str, str]:
    mean_row = next((r for r in rows if str(r.get("trial", "")) == "mean"), None)
    if mean_row is not None:
        return mean_row
    zero_row = next((r for r in rows if str(r.get("trial", "")) in {"0", "0.0"}), None)
    if zero_row is not None:
        return zero_row
    return rows[0]


def _canonicalize_eval_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row.get("checkpoint", "")),
            str(row.get("transform", "")),
            str(row.get("severity_id", "")),
            str(row.get("rho", "")),
            str(row.get("g_db", "")),
            str(row.get("b_bins", "")),
            str(row.get("run_id", "")),
        )
        grouped.setdefault(key, []).append(row)
    return [_preferred_trial(candidates) for candidates in grouped.values()]


def _parse_lambda_from_label(label: str) -> Tuple[float, float]:
    match = re.fullmatch(r"lam([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)", label.strip())
    if not match:
        raise ValueError(f"checkpoint label must be like lam1.0_0.5, got: {label}")
    return float(match.group(1)), float(match.group(2))


def _calc_mixed_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    clean_rows = [r for r in rows if str(r.get("transform", "")) == "clean"]
    transformed = [r for r in rows if str(r.get("transform", "")) != "clean"]
    if not clean_rows:
        raise ValueError("missing clean rows for one checkpoint group")
    if not transformed:
        raise ValueError("missing transformed rows for one checkpoint group")
    clean_acc = _safe_float(clean_rows[0].get("acc", 0.0), 0.0)
    clean_mf1 = _safe_float(clean_rows[0].get("mf1", 0.0), 0.0)
    acc_vals = [_safe_float(r.get("acc", 0.0), 0.0) for r in transformed]
    mf1_vals = [_safe_float(r.get("mf1", 0.0), 0.0) for r in transformed]
    mixed_avg_acc = float(np.mean(acc_vals))
    mixed_avg_mf1 = float(np.mean(mf1_vals))
    mixed_worst_acc = float(np.min(acc_vals))
    mixed_worst_mf1 = float(np.min(mf1_vals))
    return {
        "clean_acc": clean_acc,
        "clean_mf1": clean_mf1,
        "mixed_avg_acc": mixed_avg_acc,
        "mixed_avg_mf1": mixed_avg_mf1,
        "mixed_worst_acc": mixed_worst_acc,
        "mixed_worst_mf1": mixed_worst_mf1,
        "drop_acc": clean_acc - mixed_avg_acc,
        "drop_mf1": clean_mf1 - mixed_avg_mf1,
        "n_mixed_points": float(len(transformed)),
    }


def _build_checkpoint_metrics(summary_rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    per_checkpoint: Dict[str, List[Dict[str, str]]] = {}
    for row in summary_rows:
        name = str(row.get("checkpoint", "")).strip()
        if not name:
            continue
        per_checkpoint.setdefault(name, []).append(row)

    out: List[Dict[str, float]] = []
    for checkpoint in sorted(per_checkpoint.keys()):
        lm, lt = _parse_lambda_from_label(checkpoint)
        metrics = _calc_mixed_metrics(per_checkpoint[checkpoint])
        out.append(
            {
                "setting": checkpoint,
                "lambda_md": lm,
                "lambda_ta": lt,
                **metrics,
            }
        )
    out.sort(key=lambda r: (r["lambda_md"], r["lambda_ta"]))
    return out


def _parse_float_list(raw: str, field_name: str, expected_len: Optional[int] = None) -> List[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(
            f"{field_name} must have exactly {expected_len} values, got {len(values)}: {raw}"
        )
    return values


def _to_pct(value: float) -> float:
    if math.isnan(value):
        return value
    # Accept either [0,1] values or already-percent values.
    return value * 100.0 if abs(value) <= 1.5 else value


def _parse_lambda_series(raw: str, lambda_ta_vals: List[float]) -> Dict[float, List[float]]:
    series: Dict[float, List[float]] = {}
    blocks = [b.strip() for b in str(raw).split(";") if b.strip()]
    if not blocks:
        raise ValueError("lambda series is empty")
    for block in blocks:
        if ":" not in block:
            raise ValueError(
                f"invalid lambda series block: '{block}'. Expected format 'md:v1,v2,...'"
            )
        md_text, vals_text = block.split(":", 1)
        md = float(md_text.strip())
        vals = _parse_float_list(
            vals_text,
            field_name=f"lambda series for lambda_md={md:g}",
            expected_len=len(lambda_ta_vals),
        )
        series[md] = vals
    return series


def _build_lambda_series_from_metrics(
    rows: List[Dict[str, float]],
    lambda_md_vals: List[float],
    lambda_ta_vals: List[float],
    metric: str = "mixed_worst_acc",
) -> Dict[float, List[float]]:
    lookup: Dict[Tuple[float, float], float] = {}
    for row in rows:
        key = (float(row["lambda_md"]), float(row["lambda_ta"]))
        lookup[key] = float(row[metric])

    out: Dict[float, List[float]] = {}
    for lm in lambda_md_vals:
        values: List[float] = []
        for lt in lambda_ta_vals:
            key = (float(lm), float(lt))
            if key not in lookup:
                raise ValueError(f"missing lambda combination for line plot: lambda_md={lm:g}, lambda_ta={lt:g}")
            values.append(_to_pct(lookup[key]))
        out[float(lm)] = values
    return out


def _plot_gamma_stacked(
    ax_count: plt.Axes,
    ax_acc: plt.Axes,
    gamma_rows: List[Dict[str, float]],
    title: str,
) -> None:
    x = [float(r["gamma"]) for r in gamma_rows]
    y_worst = [_to_pct(float(r["mixed_worst_acc"])) for r in gamma_rows]
    n_points = [float(r["n_safe_points"]) for r in gamma_rows]

    if len(x) > 1:
        min_delta = min(abs(x[i + 1] - x[i]) for i in range(len(x) - 1))
        bar_width = min_delta * 0.30
    else:
        bar_width = 0.018

    bars = ax_count.bar(
        x,
        n_points,
        width=bar_width,
        color="#eff6ff",
        edgecolor="#bfdbfe",
        linewidth=0.9,
        alpha=0.55,
        zorder=2,
    )
    ax_count.set_ylabel("Retained severity count", fontsize=12, color="#4b5563")
    ax_count.tick_params(axis="x", labelbottom=False)
    ax_count.tick_params(axis="y", labelsize=10, colors="#4b5563")
    ax_count.set_title(title, fontsize=13, pad=6)
    ax_count.grid(axis="y", alpha=0.20, linestyle="-", zorder=1)
    if n_points:
        n_max = max(n_points)
        ax_count.set_ylim(0.0, n_max + max(1.0, 0.22 * n_max))

    for rect, value in zip(bars, n_points):
        x_mid = rect.get_x() + rect.get_width() / 2.0
        y_top = rect.get_height()
        if abs(value - round(value)) < 1e-6:
            text = f"{int(round(value))}"
        else:
            text = f"{value:.1f}"
        ax_count.annotate(
            text,
            xy=(x_mid, y_top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#374151",
        )

    ax_acc.plot(
        x,
        y_worst,
        color="#1f3f6f",
        marker="o",
        markersize=6.6,
        linewidth=2.7,
        zorder=3,
    )
    ax_acc.set_ylabel("Mixed worst-case ACC (%)", fontsize=12, color="#1f3f6f")
    ax_acc.set_xlabel(r"Teacher agreement threshold $\gamma$", fontsize=12)
    ax_acc.tick_params(axis="x", labelsize=10)
    ax_acc.tick_params(axis="y", labelsize=10, colors="#1f3f6f")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([f"{v:.2f}" for v in x])
    ax_acc.grid(axis="y", alpha=0.22, linestyle="-", zorder=1)

    valid_y = [v for v in y_worst if not math.isnan(v)]
    if valid_y:
        y_min = min(valid_y)
        y_max = max(valid_y)
        pad = max(0.8, (y_max - y_min) * 0.35)
        ax_acc.set_ylim(y_min - pad, y_max + pad)

    max_idx = None
    max_val = -float("inf")
    for idx, y_i in enumerate(y_worst):
        if not math.isnan(y_i) and y_i > max_val:
            max_val = y_i
            max_idx = idx
    if max_idx is not None:
        ax_acc.annotate(
            f"{y_worst[max_idx]:.1f}",
            xy=(x[max_idx], y_worst[max_idx]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#1f3f6f",
        )

    legend_handles = [
        Patch(facecolor="#eff6ff", edgecolor="#bfdbfe", alpha=0.55, label="Retained count"),
        Line2D([0], [0], color="#1f3f6f", marker="o", linewidth=2.0, markersize=4.8, label="Worst-case ACC"),
    ]
    ax_acc.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        fontsize=8,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.2,
        borderaxespad=0.1,
    )


def _plot_lambda_grouped_lines(
    ax: plt.Axes,
    lambda_ta_vals: List[float],
    lambda_series: Dict[float, List[float]],
    title: str,
) -> None:
    style_map = {
        0.0: {"linestyle": "-", "marker": "o", "color": "#1f4e79"},
        0.5: {"linestyle": "--", "marker": "s", "color": "#2f855a"},
        1.0: {"linestyle": "-.", "marker": "^", "color": "#b45309"},
    }
    fallback_colors = ["#1f4e79", "#2f855a", "#b45309", "#7c3aed"]

    ordered_md = sorted(lambda_series.keys())
    for idx, md in enumerate(ordered_md):
        ys = [_to_pct(v) for v in lambda_series[md]]
        style = next((v for k, v in style_map.items() if abs(md - k) < 1e-8), None)
        if style is None:
            style = {
                "linestyle": "-",
                "marker": "o",
                "color": fallback_colors[idx % len(fallback_colors)],
            }
        ax.plot(
            lambda_ta_vals,
            ys,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=6.5,
            linewidth=2.6,
            color=style["color"],
            label=rf"$\lambda_{{md}}={md:g}$",
        )

    ax.set_xlabel(r"$\lambda_{ta}$", fontsize=13)
    ax.set_ylabel("Mixed worst-case ACC (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(lambda_ta_vals)
    ax.set_xticklabels([f"{x:g}" for x in lambda_ta_vals])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.24, linestyle="-")
    ax.legend(frameon=False, fontsize=10, loc="best")


def _plot_gamma_sensitivity_figure(
    gamma_rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    fig, (ax_count, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06},
    )
    _plot_gamma_stacked(
        ax_count=ax_count,
        ax_acc=ax_acc,
        gamma_rows=gamma_rows,
        title="Safe-A threshold",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.12, hspace=0.06)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _plot_lambda_sensitivity(
    lambda_ta_vals: List[float],
    lambda_series: Dict[float, List[float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    _plot_lambda_grouped_lines(
        ax,
        lambda_ta_vals=lambda_ta_vals,
        lambda_series=lambda_series,
        title="Loss-weight sensitivity",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _plot_combined_sensitivity(
    gamma_rows: List[Dict[str, float]],
    lambda_ta_vals: List[float],
    lambda_series: Dict[float, List[float]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(11.8, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[2, 3], wspace=0.26)

    left_grid = outer[0, 0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.06)
    ax_count = fig.add_subplot(left_grid[0, 0])
    ax_acc = fig.add_subplot(left_grid[1, 0], sharex=ax_count)
    _plot_gamma_stacked(
        ax_count=ax_count,
        ax_acc=ax_acc,
        gamma_rows=gamma_rows,
        title="(a) Safe-A threshold",
    )

    right_ax = fig.add_subplot(outer[0, 1])
    _plot_lambda_grouped_lines(
        right_ax,
        lambda_ta_vals=lambda_ta_vals,
        lambda_series=lambda_series,
        title="(b) Loss-weight sensitivity",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.12, wspace=0.26)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _safe_values_by_gamma(teacher_rows: List[Dict[str, str]], gamma: float) -> Dict[str, set]:
    safe = {"shift": set(), "scale": set(), "color": set()}
    for row in teacher_rows:
        transform = str(row.get("transform", ""))
        if transform not in safe:
            continue
        agreement = _safe_float(row.get("agreement", 0.0), 0.0)
        if agreement + 1e-12 < gamma:
            continue
        if transform == "shift":
            safe["shift"].add(round(_safe_float(row.get("b_bins", 0.0), 0.0), 6))
        elif transform == "scale":
            safe["scale"].add(round(_safe_float(row.get("rho", 1.0), 1.0), 6))
        elif transform == "color":
            safe["color"].add(round(_safe_float(row.get("g_db", 0.0), 0.0), 6))
    return safe


def _apply_safe_filter(rows: List[Dict[str, str]], safe_values: Dict[str, set]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        transform = str(row.get("transform", ""))
        if transform == "clean":
            out.append(row)
            continue
        if transform not in safe_values:
            continue
        if transform == "shift":
            x = round(_safe_float(row.get("b_bins", 0.0), 0.0), 6)
        elif transform == "scale":
            x = round(_safe_float(row.get("rho", 1.0), 1.0), 6)
        else:
            x = round(_safe_float(row.get("g_db", 0.0), 0.0), 6)
        if x in safe_values[transform]:
            out.append(row)
    return out


def _compute_gamma_sensitivity(
    summary_rows: List[Dict[str, str]],
    teacher_rows: List[Dict[str, str]],
    method_label: str,
    gammas: List[float],
) -> List[Dict[str, float]]:
    method_rows = [r for r in summary_rows if str(r.get("checkpoint", "")) == method_label]
    if not method_rows:
        raise ValueError(f"method '{method_label}' not found in summary csv")

    curve_rows: List[Dict[str, float]] = []
    for gamma in gammas:
        safe_vals = _safe_values_by_gamma(teacher_rows, gamma)
        filtered = _apply_safe_filter(method_rows, safe_vals)
        transformed_count = len([r for r in filtered if str(r.get("transform", "")) != "clean"])
        if transformed_count == 0:
            curve_rows.append(
                {
                    "gamma": gamma,
                    "mixed_avg_acc": math.nan,
                    "mixed_worst_acc": math.nan,
                    "n_safe_points": 0.0,
                }
            )
            continue
        m = _calc_mixed_metrics(filtered)
        curve_rows.append(
            {
                "gamma": gamma,
                "mixed_avg_acc": m["mixed_avg_acc"],
                "mixed_worst_acc": m["mixed_worst_acc"],
                "n_safe_points": float(transformed_count),
            }
        )

    return curve_rows


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "NA"
    return f"{v:.4f}"


def _write_markdown_table(path: Path, rows: List[Dict[str, float]]) -> None:
    headers = [
        "setting",
        "lambda_md",
        "lambda_ta",
        "clean_acc",
        "mixed_avg_acc",
        "mixed_worst_acc",
        "drop_acc",
        "clean_mf1",
        "mixed_avg_mf1",
        "mixed_worst_mf1",
        "drop_mf1",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        vals = [
            str(r["setting"]),
            _fmt(float(r["lambda_md"])),
            _fmt(float(r["lambda_ta"])),
            _fmt(float(r["clean_acc"])),
            _fmt(float(r["mixed_avg_acc"])),
            _fmt(float(r["mixed_worst_acc"])),
            _fmt(float(r["drop_acc"])),
            _fmt(float(r["clean_mf1"])),
            _fmt(float(r["mixed_avg_mf1"])),
            _fmt(float(r["mixed_worst_mf1"])),
            _fmt(float(r["drop_mf1"])),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--teacher-summary-csv", type=str, required=True)
    parser.add_argument("--gamma-method", type=str, default="lam1.0_1.0")
    parser.add_argument("--gamma-values", type=str, default="0.90,0.95,0.98")
    parser.add_argument("--paper-values", dest="paper_values", action="store_true", default=True)
    parser.add_argument("--no-paper-values", dest="paper_values", action="store_false")
    parser.add_argument("--gamma-worst-acc-values", type=str, default="64.4,64.4,67.5")
    parser.add_argument("--gamma-retained-count-values", type=str, default="11,11,5")
    parser.add_argument(
        "--lambda-series-values",
        type=str,
        default="0.0:31.2,69.7,71.2,72.5;0.5:59.4,56.2,70.6,71.9;1.0:61.4,56.4,62.5,64.4",
    )
    parser.add_argument("--output-root", type=str, default="time-main/outputs_46")
    parser.add_argument("--figure-prefix", type=str, default="fig46")
    parser.add_argument("--combined-suffix", type=str, default="sensitivity_panel")
    parser.add_argument("--table-prefix", type=str, default="table46")
    parser.add_argument("--lambda-md-grid", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--lambda-ta-grid", type=str, default="0.0,0.2,0.5,1.0")
    parser.add_argument("--fig-only", action="store_true", default=False)
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    teacher_path = Path(args.teacher_summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher summary csv not found: {teacher_path}")

    summary_rows = _canonicalize_eval_rows(_read_csv(summary_path))
    teacher_rows = _canonicalize_eval_rows(_read_csv(teacher_path)) if not args.paper_values else []
    metrics_rows = _build_checkpoint_metrics(summary_rows)

    output_root = Path(args.output_root)
    figs_dir = output_root / "figs"
    tables_dir = output_root / "tables"
    csv_dir = output_root / "csv"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    lambda_md_grid = _parse_float_list(args.lambda_md_grid, field_name="lambda-md-grid")
    lambda_ta_grid = _parse_float_list(args.lambda_ta_grid, field_name="lambda-ta-grid")
    gamma_values = _parse_float_list(args.gamma_values, field_name="gamma-values")

    if args.paper_values:
        gamma_worst_acc_vals = _parse_float_list(
            args.gamma_worst_acc_values,
            field_name="gamma-worst-acc-values",
            expected_len=len(gamma_values),
        )
        gamma_retained_vals = _parse_float_list(
            args.gamma_retained_count_values,
            field_name="gamma-retained-count-values",
            expected_len=len(gamma_values),
        )
        gamma_rows: List[Dict[str, float]] = []
        for g, w_acc, n_keep in zip(gamma_values, gamma_worst_acc_vals, gamma_retained_vals):
            gamma_rows.append(
                {
                    "gamma": float(g),
                    "mixed_avg_acc": math.nan,
                    "mixed_worst_acc": float(w_acc) / 100.0,
                    "n_safe_points": float(n_keep),
                }
            )
        lambda_series = _parse_lambda_series(args.lambda_series_values, lambda_ta_grid)
    else:
        gamma_rows = _compute_gamma_sensitivity(
            summary_rows=summary_rows,
            teacher_rows=teacher_rows,
            method_label=args.gamma_method,
            gammas=gamma_values,
        )
        lambda_series = _build_lambda_series_from_metrics(
            metrics_rows,
            lambda_md_vals=lambda_md_grid,
            lambda_ta_vals=lambda_ta_grid,
            metric="mixed_worst_acc",
        )

    lambda_path = figs_dir / f"{args.figure_prefix}_lambda_heatmap.png"
    lambda_curve_path = figs_dir / f"{args.figure_prefix}_lambda_curve.png"
    gamma_path = figs_dir / f"{args.figure_prefix}_gamma_curve.png"
    panel_path = figs_dir / f"{args.figure_prefix}_{args.combined_suffix}.png"
    _plot_lambda_sensitivity(
        lambda_ta_vals=lambda_ta_grid,
        lambda_series=lambda_series,
        out_path=lambda_path,
    )
    _plot_lambda_sensitivity(
        lambda_ta_vals=lambda_ta_grid,
        lambda_series=lambda_series,
        out_path=lambda_curve_path,
    )
    _plot_gamma_sensitivity_figure(
        gamma_rows=gamma_rows,
        out_path=gamma_path,
    )
    _plot_combined_sensitivity(
        gamma_rows=gamma_rows,
        lambda_ta_vals=lambda_ta_grid,
        lambda_series=lambda_series,
        out_path=panel_path,
    )

    print(f"saved_heatmap={lambda_path}")
    print(f"saved_lambda_curve={lambda_curve_path}")
    print(f"saved_gamma_curve={gamma_path}")
    print(f"saved_sensitivity_panel={panel_path}")
    if not args.fig_only:
        table_csv = tables_dir / f"{args.table_prefix}_stability_summary.csv"
        _write_csv(table_csv, metrics_rows)
        table_md = tables_dir / f"{args.table_prefix}_stability_summary.md"
        _write_markdown_table(table_md, metrics_rows)
        gamma_csv = csv_dir / f"{args.table_prefix}_gamma_curve.csv"
        _write_csv(gamma_csv, gamma_rows)
        print(f"saved_table_csv={table_csv}")
        print(f"saved_table_md={table_md}")
        print(f"saved_gamma_csv={gamma_csv}")


if __name__ == "__main__":
    main()
