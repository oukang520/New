"""Render the selected project palette for Rel-ObsTQ-MHN figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style


PALETTES = [
    {
        "id": "Selected",
        "name": "User Pastel Top-Journal",
        "source": "User-provided project default",
        "categorical": ["#B5AED5", "#B2E6FD", "#B8D2CC", "#E8B2A7", "#FEEBB9"],
        "sequential": ["#FEEBB9", "#B8D2CC", "#B2E6FD", "#B5AED5"],
        "diverging": ["#E8B2A7", "#FEEBB9", "#B2E6FD"],
    },
]


def draw_gradient(ax: plt.Axes, colors: list[str], label: str) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(label, colors)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set_axis_off()
    ax.text(0, -0.75, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color="#444444")


def draw_swatch_row(ax: plt.Axes, colors: list[str]) -> None:
    ax.set_axis_off()
    for idx, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((idx, 0), 0.82, 1, facecolor=color, edgecolor="white", linewidth=0.6))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)


def draw_demo_matrix(ax: plt.Axes, sequential: list[str], diverging: list[str]) -> None:
    rho = np.array(
        [
            [0.68, 0.20, 0.40, 0.44],
            [0.73, 0.11, 0.34, 0.19],
            [0.66, 0.44, 0.44, 0.21],
            [0.73, 0.39, 0.30, 0.42],
        ]
    )
    delta = np.array(
        [
            [0.14, 0.11, 0.23, 0.39],
            [-0.04, 0.25, 0.16, 0.23],
            [-0.07, 0.18, 0.15, 0.24],
            [0.06, 0.12, 0.28, 0.48],
        ]
    )
    seq_cmap = mcolors.LinearSegmentedColormap.from_list("seq_demo", sequential)
    div_cmap = mcolors.LinearSegmentedColormap.from_list("div_demo", diverging)
    ax.imshow(rho, cmap=seq_cmap, norm=mcolors.Normalize(vmin=-0.25, vmax=1.0), aspect="auto")
    div_norm = mcolors.Normalize(vmin=-0.5, vmax=0.5)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            ax.add_patch(
                plt.Rectangle((j - 0.5, i + 0.22), 1.0, 0.28, facecolor=div_cmap(div_norm(delta[i, j])), edgecolor="none")
            )
            color = "white" if rho[i, j] >= 0.58 else "#222222"
            ax.text(j, i - 0.08, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color=color)
            ax.text(j, i + 0.36, f"Δ{delta[i, j]:+.2f}", ha="center", va="center", fontsize=4.8, color="#222222")
    for x in np.arange(-0.5, rho.shape[1] + 0.5, 1):
        ax.axvline(x, color="white", lw=0.45)
    for y in np.arange(-0.5, rho.shape[0] + 0.5, 1):
        ax.axhline(y, color="white", lw=0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def write_palette_table(output: Path) -> None:
    rows = []
    for palette in PALETTES:
        rows.append(
            {
                "id": palette["id"],
                "name": palette["name"],
                "source": palette["source"],
                "categorical": ",".join(palette["categorical"]),
                "sequential": ",".join(palette["sequential"]),
                "diverging": ",".join(palette["diverging"]),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    config = {"plot_style_config": "configs/figure_style.yaml", "plot": {"font_family": "Arial", "base_font_size": 8, "dpi": 600}}
    figure_style.configure_matplotlib(config)
    output_dir = Path("results/figure_style_palette_options")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_palette_table(output_dir / "palette_options.csv")

    fig = plt.figure(figsize=(10.8, 2.5))
    grid = fig.add_gridspec(
        len(PALETTES) + 1,
        5,
        width_ratios=[1.65, 1.55, 1.20, 1.20, 1.55],
        height_ratios=[0.42] + [1] * len(PALETTES),
        left=0.04,
        right=0.985,
        top=0.78,
        bottom=0.20,
        hspace=0.38,
        wspace=0.28,
    )
    fig.suptitle("Selected project palette for Rel-ObsTQ-MHN figures", x=0.04, ha="left", fontsize=12, fontweight="bold")
    headers = ["Palette", "Categorical anchors", "rho sequential", "delta rho diverging", "Matrix preview"]
    for col, header in enumerate(headers):
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        ax.text(0, 0.5, header, ha="left", va="center", fontsize=8.2, fontweight="bold", color="#222222")

    for row, palette in enumerate(PALETTES, start=1):
        ax_text = fig.add_subplot(grid[row, 0])
        ax_text.axis("off")
        ax_text.text(0, 0.74, f"{palette['id']}. {palette['name']}", fontsize=9.1, fontweight="bold", color="#222222", ha="left", va="center")
        ax_text.text(0, 0.40, palette["source"], fontsize=6.8, color="#555555", ha="left", va="center", wrap=True)

        ax_swatches = fig.add_subplot(grid[row, 1])
        draw_swatch_row(ax_swatches, palette["categorical"])

        ax_seq = fig.add_subplot(grid[row, 2])
        draw_gradient(ax_seq, palette["sequential"], "rho")

        ax_div = fig.add_subplot(grid[row, 3])
        draw_gradient(ax_div, palette["diverging"], "delta rho")

        ax_demo = fig.add_subplot(grid[row, 4])
        draw_demo_matrix(ax_demo, palette["sequential"], palette["diverging"])

    fig.text(
        0.04,
        0.025,
        "Use: cell fill encodes rho = Spearman(D_true, R*); bottom strip encodes delta rho vs occupancy. This is the only active project palette.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    figure_style.save_figure(fig, output_dir / "Figure_palette_options", config)


if __name__ == "__main__":
    main()
