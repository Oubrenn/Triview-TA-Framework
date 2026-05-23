import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> List[Dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(float(row["epoch"])))
    return rows


def series(rows: List[Dict[str, object]], key: str) -> List[float]:
    return [float(row[key]) for row in rows]


def epochs(rows: List[Dict[str, object]]) -> List[int]:
    return [int(float(row["epoch"])) for row in rows]


def plot_cv_evolution(tri_view_csv: Path, triview_ta_csv: Path, out_path: Path) -> None:
    tri = load_csv(tri_view_csv)
    ta = load_csv(triview_ta_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    colors = {"Tri-view": "#D55E00", "TriView-TA": "#0072B2"}

    ax = axes[0]
    ax.plot(epochs(tri), series(tri, "clean_d_mean"), marker="o", linewidth=2.0, label="Tri-view", color=colors["Tri-view"])
    ax.plot(epochs(ta), series(ta, "clean_d_mean"), marker="s", linewidth=2.0, label="TriView-TA", color=colors["TriView-TA"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean clean cross-view distance")
    ax.set_title("Clean cross-view geometry")
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(epochs(tri), series(tri, "cv_drift"), marker="o", linewidth=2.0, label="Tri-view", color=colors["Tri-view"])
    ax.plot(epochs(ta), series(ta, "cv_drift"), marker="s", linewidth=2.0, label="TriView-TA", color=colors["TriView-TA"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perturbation-induced cross-view drift")
    ax.set_title("Cross-view drift under mixed perturbation")
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".png":
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tri-view-csv", "--tri_view_csv", dest="tri_view_csv", type=str, required=True)
    parser.add_argument("--triview-ta-csv", "--triview_ta_csv", dest="triview_ta_csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="figs/training_evolution_cross_view.pdf")
    args = parser.parse_args()
    plot_cv_evolution(
        tri_view_csv=Path(args.tri_view_csv),
        triview_ta_csv=Path(args.triview_ta_csv),
        out_path=Path(args.out),
    )
    print(f"wrote_fig={args.out}")


if __name__ == "__main__":
    main()
