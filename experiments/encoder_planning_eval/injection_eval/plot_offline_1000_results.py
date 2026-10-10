"""Make a slide-ready AUROC comparison for the 200/800 SSv2 offline run."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
INPUT = RESULTS / "ssv2_1000_cal200_eval800_headline_comparison.csv"
OUT = RESULTS / "threshold_examples_1000"
EDITS = [
    ("swap", "Swap"), ("shuffle", "Shuffle"), ("drop", "Drop"),
    ("speed_up", "Speed-up"), ("slow_down", "Slow-down"),
    ("splice_other", "Other-class splice"), ("splice_same", "Same-class splice"),
]
READOUTS = [
    ("DINOv2 CLS", "#a153ad", -0.23, "o"),
    ("DINOv2 patch mean", "#2584b3", 0, "s"),
    ("VICREG patch concat", "#328d3d", 0.23, "D"),
]


def main():
    data = pd.read_csv(INPUT).set_index(["edit", "representation"])
    fig, ax = plt.subplots(figsize=(10.8, 5.9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for row, (edit, label) in enumerate(EDITS):
        if row % 2 == 0:
            ax.axhspan(row - .48, row + .48, color="#f3f6f8", zorder=0)
    for name, color, offset, marker in READOUTS:
        centers = np.array([data.loc[(edit, name), "auroc"] for edit, _ in EDITS])
        lows = np.array([data.loc[(edit, name), "ci_lo"] for edit, _ in EDITS])
        highs = np.array([data.loc[(edit, name), "ci_hi"] for edit, _ in EDITS])
        y = np.arange(len(EDITS)) + offset
        ax.errorbar(centers, y, xerr=np.vstack([centers - lows, highs - centers]),
                    fmt=marker, color=color, ecolor=color, elinewidth=1.7,
                    capsize=3, markersize=7.2, markeredgecolor="white",
                    markeredgewidth=.7, linestyle="none", label=name, zorder=3)
    ax.axvline(.5, color="#83919b", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(.505, -.56, "chance", color="#697984", fontsize=9)
    ax.set_yticks(range(len(EDITS)), [label for _, label in EDITS])
    ax.invert_yaxis()
    ax.set_xlim(.48, .985)
    ax.set_xticks(np.arange(.5, 1.01, .1))
    ax.set_xlabel("AUROC (higher is better)", fontsize=11)
    ax.set_title("Offline detection of synthetic edits", loc="left", fontsize=17, weight="bold", pad=18)
    ax.text(0, 1.015, "SSv2 · 800 held-out clips per edit · 200 clean calibration clips · geometry scores",
            transform=ax.transAxes, fontsize=10, color="#53616b")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d8e0e5", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors="#53616b")
    ax.grid(axis="x", color="#e1e7eb", zorder=0)
    ax.set_axisbelow(True)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = OUT / f"offline_auroc_comparison.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
