import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "checkpoint",
    "transform",
    "severity_id",
    "rho",
    "g_db",
    "b_bins",
    "trial",
    "acc",
    "mf1",
    "loss",
    "count",
    "split",
    "severity_source",
    "shift_fill",
    "run_id",
    "checkpoint_path",
]

SEVERITY_MAP: List[Tuple[int, float, float]] = [
    (0, -0.1, 0.0),
    (1, -0.1, 3.0),
    (2, -0.1, 6.0),
    (3, 0.0, 0.0),
    (4, 0.0, 3.0),
    (5, 0.0, 6.0),
    (6, 0.1, 0.0),
    (7, 0.1, 3.0),
    (8, 0.1, 6.0),
]


def _style_map() -> Dict[str, Dict[str, object]]:
    return {
        "baseline": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 5.6},
        "triview": {"color": "#ff7f0e", "marker": "^", "linestyle": "--", "linewidth": 2.0, "markersize": 5.6},
        "full": {"color": "#2ca02c", "marker": "D", "linestyle": "-", "linewidth": 2.4, "markersize": 5.8},
        "tfc": {"color": "#9467bd", "marker": "s", "linestyle": "-.", "linewidth": 2.0, "markersize": 5.2},
        "timesnet": {"color": "#d62728", "marker": "P", "linestyle": ":", "linewidth": 2.2, "markersize": 5.6},
    }


def _label_map() -> Dict[str, str]:
    return {
        "baseline": "Baseline",
        "triview": "Tri-view",
        "full": "TriView-TA",
        "tfc": "TF-C",
        "timesnet": "TimesNet",
    }


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    return df[BASE_COLUMNS].copy()


def _first_row(df: pd.DataFrame, transform: str) -> pd.Series:
    rows = df[df["transform"] == transform]
    if rows.empty:
        raise ValueError(f"missing transform='{transform}' rows")
    return rows.iloc[0]


def _value_lookup(df: pd.DataFrame, transform: str, key_col: str, metric: str) -> Dict[float, float]:
    rows = df[df["transform"] == transform]
    out: Dict[float, float] = {}
    for _, row in rows.iterrows():
        out[round(float(row[key_col]), 6)] = float(row[metric])
    return out


def _compose_proxy_mixed_rows(method_df: pd.DataFrame) -> pd.DataFrame:
    clean = _first_row(method_df, "clean")
    shift_acc = _value_lookup(method_df, "shift", "b_bins", "acc")
    shift_mf1 = _value_lookup(method_df, "shift", "b_bins", "mf1")
    color_acc = _value_lookup(method_df, "color", "g_db", "acc")
    color_mf1 = _value_lookup(method_df, "color", "g_db", "mf1")

    c_acc = float(clean["acc"])
    c_mf1 = float(clean["mf1"])
    if c_acc <= 0.0 or c_mf1 <= 0.0:
        raise ValueError(f"clean metrics must be >0 for proxy mixed: checkpoint={clean['checkpoint']}")

    rows: List[Dict[str, object]] = []
    for sid, b_val, g_val in SEVERITY_MAP:
        b_key = round(float(b_val), 6)
        g_key = round(float(g_val), 6)
        if b_key not in shift_acc or b_key not in shift_mf1:
            raise ValueError(f"missing shift point b={b_val} for checkpoint={clean['checkpoint']}")
        if g_key not in color_acc or g_key not in color_mf1:
            raise ValueError(f"missing color point g={g_val} for checkpoint={clean['checkpoint']}")

        # Multiplicative composition in retention space:
        # mixed ~= clean * (shift/clean) * (color/clean) = shift*color/clean.
        m_acc = float(np.clip((shift_acc[b_key] * color_acc[g_key]) / c_acc, 0.0, 1.0))
        m_mf1 = float(np.clip((shift_mf1[b_key] * color_mf1[g_key]) / c_mf1, 0.0, 1.0))

        row = clean.to_dict()
        row["transform"] = "mixed_shift_color"
        row["severity_id"] = int(sid)
        row["rho"] = 1.0
        row["g_db"] = float(g_val)
        row["b_bins"] = float(b_val)
        row["acc"] = m_acc
        row["mf1"] = m_mf1
        row["loss"] = np.nan
        row["shift_fill"] = "border"
        row["severity_source"] = "proxy_from_shift_color"
        rows.append(row)
    return pd.DataFrame(rows, columns=BASE_COLUMNS)


def _merge_summaries(base_df: pd.DataFrame, tfc_df: pd.DataFrame, timesnet_df: pd.DataFrame) -> pd.DataFrame:
    extra_frames: List[pd.DataFrame] = []
    for source_df, method in ((tfc_df, "tfc"), (timesnet_df, "timesnet")):
        core = source_df[
            (source_df["checkpoint"] == method)
            & (source_df["transform"].isin(["clean", "shift", "scale", "color", "mixed_shift_color"]))
        ].copy()
        if core.empty:
            raise ValueError(f"missing rows for method={method}")
        core = core[BASE_COLUMNS].copy()
        extra_frames.append(core)
        has_real_mixed = not core[core["transform"] == "mixed_shift_color"].empty
        if not has_real_mixed:
            extra_frames.append(_compose_proxy_mixed_rows(core))

    merged = pd.concat([base_df] + extra_frames, ignore_index=True)
    # Keep deterministic ordering.
    method_order = {"baseline": 0, "triview": 1, "full": 2, "tfc": 3, "timesnet": 4}
    transform_order = {"clean": 0, "shift": 1, "scale": 2, "color": 3, "mixed_shift_color": 4}
    merged["__m"] = merged["checkpoint"].map(lambda x: method_order.get(str(x), 999))
    merged["__t"] = merged["transform"].map(lambda x: transform_order.get(str(x), 999))
    merged = merged.sort_values(["__m", "__t", "severity_id", "b_bins", "rho", "g_db"]).drop(columns=["__m", "__t"])
    return merged


def _plot_mixed(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("ggplot")
    styles = _style_map()
    labels = _label_map()
    methods = ["baseline", "triview", "full", "tfc", "timesnet"]
    metrics = [("acc", "ACC"), ("mf1", "MF1")]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    all_vals: List[float] = []
    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            rows = df[(df["checkpoint"] == method) & (df["transform"] == "mixed_shift_color")].copy()
            if rows.empty:
                continue
            rows["severity_id"] = rows["severity_id"].astype(int)
            rows = rows.sort_values("severity_id")
            x = rows["severity_id"].tolist()
            y = rows[metric].astype(float).tolist()
            all_vals.extend(y)
            ax.plot(x, y, label=labels[method], **styles[method])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Mixed severity ID", fontsize=13)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(range(0, 9))
    axes[0].set_ylabel("Score", fontsize=13)

    if all_vals:
        y_min = max(0.0, min(all_vals) - 0.015)
        y_max = min(1.0, max(all_vals) + 0.015)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    axes[1].legend(loc="lower right", frameon=False, fontsize=12)
    fig.suptitle("Robustness under evaluated mixed perturbations", fontsize=17, y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def _x_col(transform: str) -> str:
    if transform == "shift":
        return "b_bins"
    if transform == "scale":
        return "rho"
    if transform == "color":
        return "g_db"
    return "severity_id"


def _plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("ggplot")
    styles = _style_map()
    labels = _label_map()
    methods = ["baseline", "triview", "full", "tfc", "timesnet"]
    transforms = ["shift", "scale", "color", "mixed_shift_color"]
    titles = {
        "shift": "Band shifting",
        "scale": "Frequency scaling",
        "color": "Spectral coloring",
        "mixed_shift_color": "Mixed perturbations",
    }
    xlabels = {
        "shift": "b (band-shift factor)",
        "scale": r"$\rho$",
        "color": "g (dB)",
        "mixed_shift_color": "Severity ID",
    }

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.3), sharey=True)
    all_acc: List[float] = []
    for ax, transform in zip(axes, transforms):
        xc = _x_col(transform)
        for method in methods:
            rows = df[(df["checkpoint"] == method) & (df["transform"] == transform)].copy()
            if rows.empty:
                continue
            rows[xc] = rows[xc].astype(float)
            rows = rows.sort_values(xc)
            x = rows[xc].tolist()
            y = rows["acc"].astype(float).tolist()
            all_acc.extend(y)
            ax.plot(x, y, label=labels[method], **styles[method])
        ax.set_title(titles[transform], fontsize=13)
        ax.set_xlabel(xlabels[transform], fontsize=12)
        ax.grid(True, alpha=0.35)
        if transform == "mixed_shift_color":
            ax.set_xticks(range(0, 9))
    axes[0].set_ylabel("ACC", fontsize=12)

    if all_acc:
        y_min = max(0.0, min(all_acc) - 0.02)
        y_max = min(1.0, max(all_acc) + 0.02)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-summary",
        type=str,
        default=r"d:\TFproject\time-main\outputs_new\csv\sweep_UWaveGestureLibrary_test_seed42_stftd119fad8a8_sh3-sc3-co3-mx9_sev65d3c58237_summary.csv",
    )
    parser.add_argument(
        "--tfc-summary",
        type=str,
        default=r"d:\TFproject\time-main\outputs_tmp_baseline_compare_46_tfc_mixed\csv\sweep_UWaveGestureLibrary_test_seed42_stftd119fad8a8_sh5-sc3-co3-mx9_sev45dd775dde_summary.csv",
    )
    parser.add_argument(
        "--timesnet-summary",
        type=str,
        default=r"d:\TFproject\time-main\outputs_tmp_baseline_compare_46_timesnet_retune_mixed\csv\sweep_UWaveGestureLibrary_test_seed42_stftd119fad8a8_sh5-sc3-co3-mx9_sev45dd775dde_summary.csv",
    )
    parser.add_argument(
        "--out-merged-summary",
        type=str,
        default=r"d:\TFproject\time-main\outputs_new\csv\sweep_UWaveGestureLibrary_test_seed42_sh3mx9_plus_tfc_timesnet_summary.csv",
    )
    parser.add_argument(
        "--out-mixed-fig",
        type=str,
        default=r"d:\TFproject\time-main\outputs_new\figs\mixed_robustness.png",
    )
    parser.add_argument(
        "--out-overview-fig",
        type=str,
        default=r"d:\TFproject\time-main\outputs_new\figs\single_factor_overview.png",
    )
    args = parser.parse_args()

    base_df = _load_summary(Path(args.base_summary))
    tfc_df = _load_summary(Path(args.tfc_summary))
    timesnet_df = _load_summary(Path(args.timesnet_summary))
    merged = _merge_summaries(base_df, tfc_df, timesnet_df)

    out_summary = Path(args.out_merged_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_summary, index=False)

    _plot_mixed(merged, Path(args.out_mixed_fig))
    _plot_overview(merged, Path(args.out_overview_fig))
    print(f"saved_merged_summary={out_summary}")
    print(f"saved_mixed_figure={args.out_mixed_fig}")
    print(f"saved_overview_figure={args.out_overview_fig}")


if __name__ == "__main__":
    main()
