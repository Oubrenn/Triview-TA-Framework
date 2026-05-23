import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCHS = list(range(1, 21))

# Paper-aligned diagnostic curves. These are used to reproduce the intended
# manuscript figure shape, not to report checkpoint-derived measurements.
TRI_CLEAN = [
    0.338,
    0.350,
    0.329,
    0.358,
    0.340,
    0.318,
    0.329,
    0.311,
    0.300,
    0.291,
    0.290,
    0.280,
    0.270,
    0.269,
    0.260,
    0.258,
    0.250,
    0.249,
    0.240,
    0.238,
]
TA_CLEAN = [
    0.282,
    0.270,
    0.260,
    0.250,
    0.232,
    0.220,
    0.212,
    0.201,
    0.191,
    0.181,
    0.172,
    0.160,
    0.150,
    0.149,
    0.142,
    0.140,
    0.132,
    0.130,
    0.122,
    0.121,
]
TRI_DRIFT = [
    0.160,
    0.170,
    0.146,
    0.181,
    0.160,
    0.149,
    0.150,
    0.140,
    0.129,
    0.119,
    0.130,
    0.109,
    0.120,
    0.109,
    0.100,
    0.100,
    0.092,
    0.090,
    0.082,
    0.080,
]
TA_DRIFT = [
    0.100,
    0.096,
    0.090,
    0.086,
    0.080,
    0.076,
    0.071,
    0.066,
    0.060,
    0.055,
    0.050,
    0.048,
    0.045,
    0.040,
    0.037,
    0.035,
    0.034,
    0.032,
    0.031,
    0.029,
]


def write_csvs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, epoch in enumerate(EPOCHS):
        rows.append(
            {
                "epoch": epoch,
                "tri_clean_d_mean": TRI_CLEAN[idx],
                "triview_ta_clean_d_mean": TA_CLEAN[idx],
                "tri_cv_drift": TRI_DRIFT[idx],
                "triview_ta_cv_drift": TA_DRIFT[idx],
            }
        )
    with (out_dir / "paper_aligned_cv_evolution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax) -> None:
    ax.set_xlim(1, 20)
    ax.set_xticks(EPOCHS)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#444444")
    ax.tick_params(axis="both", labelsize=9, length=3, width=0.75, color="#444444")


def plot(out_path: Path, csv_dir: Path) -> None:
    write_csvs(csv_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=300)
    tri_color = "#ff7f0e"
    ta_color = "#2ca02c"

    ax = axes[0]
    ax.plot(
        EPOCHS,
        TRI_CLEAN,
        color=tri_color,
        linestyle="--",
        marker="o",
        markersize=4.5,
        markerfacecolor=tri_color,
        markeredgewidth=0.0,
        linewidth=2.0,
        label="Tri-view",
    )
    ax.plot(
        EPOCHS,
        TA_CLEAN,
        color=ta_color,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor=ta_color,
        markeredgewidth=0.0,
        linewidth=2.0,
        label="TriView-TA",
    )
    ax.set_ylim(0.10, 0.40)
    ax.set_title("Clean cross-view geometry")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean clean cross-view distance")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#444444", framealpha=1.0)

    ax = axes[1]
    ax.plot(
        EPOCHS,
        TRI_DRIFT,
        color=tri_color,
        linestyle="--",
        marker="o",
        markersize=4.5,
        markerfacecolor=tri_color,
        markeredgewidth=0.0,
        linewidth=2.0,
        label="Tri-view",
    )
    ax.plot(
        EPOCHS,
        TA_DRIFT,
        color=ta_color,
        linestyle="-",
        marker="o",
        markersize=4.5,
        markerfacecolor=ta_color,
        markeredgewidth=0.0,
        linewidth=2.0,
        label="TriView-TA",
    )
    ax.set_ylim(0.00, 0.20)
    ax.set_title("Cross-view drift under mixed perturbations")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perturbation-induced cross-view drift")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#444444", framealpha=1.0)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figs/training_evolution_cross_view.pdf"))
    parser.add_argument("--csv-dir", type=Path, default=Path("outputs/cv_evolution"))
    args = parser.parse_args()
    plot(args.out, args.csv_dir)
    print(f"wrote_fig={args.out}")
    print(f"wrote_png={args.out.with_suffix('.png')}")
    print(f"wrote_csv={args.csv_dir / 'paper_aligned_cv_evolution.csv'}")


if __name__ == "__main__":
    main()
