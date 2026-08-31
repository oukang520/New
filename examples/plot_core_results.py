"""Optional example figures from already computed result tables.

This file is intentionally isolated from all statistical workflows.  Deleting
the ``examples`` directory does not change a single numerical result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = ["#B5AED5", "#B2E6FD", "#B8D2CC", "#E8B2A7", "#FEEBB9"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_dwell_gradient(result_root: Path, output: Path) -> None:
    """Show true dwell levels against recovered R* with repeat uncertainty."""

    scores = pd.read_csv(result_root / "tables" / "repeat_state_scores.tsv", sep="\t")
    summary = (
        scores.groupby("D_true", as_index=False)["log2_R_star"]
        .agg(median="median", q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75))
        .sort_values("D_true")
    )
    x = np.log2(summary["D_true"].to_numpy(dtype=float))
    y = summary["median"].to_numpy(dtype=float)
    low = y - summary["q1"].to_numpy(dtype=float)
    high = summary["q3"].to_numpy(dtype=float) - y
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.plot([-2.2, 2.2], [-2.2, 2.2], color="#9A9A9A", lw=0.8, ls="--", zorder=1)
    ax.errorbar(x, y, yerr=[low, high], fmt="none", color="#4E5A5E", lw=0.9, capsize=2, zorder=2)
    ax.scatter(x, y, s=34, c=PALETTE, edgecolor="#263238", linewidth=0.5, zorder=3)
    ax.set(xlabel="True relative dwell, log2(D)", ylabel="Recovered relative dwell, median log2(R*)")
    ax.set_xticks(x)
    ax.set_xlim(-2.3, 2.3)
    ax.grid(axis="both", color="#E6E6E6", lw=0.45, zorder=0)
    fig.tight_layout(pad=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_longitudinal(result_root: Path, output: Path) -> None:
    """Show cohort-specific longitudinal AUC and uncertainty."""

    metrics = pd.read_csv(result_root / "tables" / "longitudinal_metrics.tsv", sep="\t")
    metrics = metrics.sort_values("auc").reset_index(drop=True)
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.axvline(0.5, color="#9A9A9A", lw=0.8, ls="--")
    ax.errorbar(
        metrics["auc"],
        y,
        xerr=[metrics["auc"] - metrics["auc_ci_low"], metrics["auc_ci_high"] - metrics["auc"]],
        fmt="none",
        color="#4E5A5E",
        lw=0.9,
        capsize=2,
    )
    ax.scatter(metrics["auc"], y, s=32, c=PALETTE[: len(metrics)], edgecolor="#263238", linewidth=0.5)
    ax.set_yticks(y, metrics["study_id"])
    ax.set(xlabel="Held-out persistence ROC AUC", xlim=(0, 1))
    ax.grid(axis="x", color="#E6E6E6", lw=0.45)
    fig.tight_layout(pad=0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["simulation", "longitudinal"])
    parser.add_argument("result_root")
    parser.add_argument("output")
    args = parser.parse_args()
    _style()
    function = plot_dwell_gradient if args.kind == "simulation" else plot_longitudinal
    function(Path(args.result_root), Path(args.output))


if __name__ == "__main__":
    main()
