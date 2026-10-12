"""Frozen visual defaults and reference selection for Invariant-EvAM figures.

Every quantitative panel must name a primary reference ID from
configs/figure_reference_catalog.json in its figure manifest.  These defaults
implement the common visual grammar; the chosen reference controls layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping, Optional, Union

import matplotlib as mpl

MM_TO_INCH = 1 / 25.4
SINGLE_COLUMN_MM = 89
DOUBLE_COLUMN_MM = 183

PALETTES = {
    "design": {
        "complementary": "#219E98",
        "redundant": "#777777",
        "failure": "#C43C4B",
        "weak": "#E5BD55",
        "unknown": "#B5B5B5",
    },
    "method": {
        "Invariant-EvAM": "#3979AD",
        "shared-no-O": "#219E98",
        "independent-meta": "#E99A45",
        "pooled": "#777777",
    },
    "sign": {
        "positive": "#C43C4B",
        "negative": "#3979AD",
        "unknown": "#B5B5B5",
    },
    "identification": {
        "identified": "#219E98",
        "partial": "#E99A45",
        "weak": "#9380B5",
        "nonidentified": "#C43C4B",
        "not_evaluated": "#D9D9D9",
    },
    "cancer": {
        "LUAD": "#4C9AC7",
        "COAD": "#E99A45",
    },
}

REFERENCE_BY_PURPOSE = {
    "profile_or_sensitivity_curve": ("R06", "R19"),
    "performance_trend": ("R04", "R11"),
    "replicate_distribution": ("R27", "R07"),
    "identifiability_matrix": ("R20", "R24"),
    "before_after_environment": ("R25",),
    "status_composition": ("R28",),
    "cross_condition_signed_effect": ("R13", "R26"),
    "truth_estimate_positive": ("R02", "R18"),
    "theory_empirical_relation": ("R23",),
    "composition_audit": ("R05", "R17"),
    "small_group_scalar": ("R29", "R01", "R21"),
}

RCPARAMS = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_publication_style(extra: Optional[Mapping[str, object]] = None) -> None:
    """Apply the frozen common visual grammar before building a figure."""
    params = dict(RCPARAMS)
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)


def figure_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Convert a requested final size from millimetres to inches."""
    return width_mm * MM_TO_INCH, height_mm * MM_TO_INCH


def palette(name: str) -> dict[str, str]:
    """Return a copy so a plotting script cannot mutate global semantics."""
    if name not in PALETTES:
        raise KeyError(f"Unknown palette family: {name}")
    return dict(PALETTES[name])


def save_publication_figure(
    fig, output_stem: Union[str, Path], dpi: int = 600
) -> tuple[Path, Path]:
    """Save vector PDF and high-resolution PNG with a white background."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    return pdf, png


def validate_figure_manifest(
    manifest_path: Union[str, Path],
    catalog_path: Union[str, Path] = "configs/figure_reference_catalog.json",
    require_review_complete: bool = False,
) -> List[str]:
    """Return validation errors for a quantitative-panel figure manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    catalog = json.loads(Path(catalog_path).read_text())
    records = {row["id"]: row for row in catalog["records"]}
    errors: List[str] = []
    required = (
        "figure_id",
        "panel_id",
        "scientific_message",
        "data_source",
        "estimand",
        "unit_of_analysis",
        "n_definition",
        "primary_reference_id",
        "reference_selection_reason",
        "code_path",
        "plot_data_path",
        "output_paths",
    )
    for key in required:
        if manifest.get(key) in (None, "", []):
            errors.append(f"missing required field: {key}")
    reference_id = manifest.get("primary_reference_id")
    if reference_id not in records:
        errors.append(f"unknown primary_reference_id: {reference_id}")
    elif not records[reference_id]["selectable"]:
        errors.append(f"reference is not selectable: {reference_id}")
    secondary = manifest.get("secondary_reference_id")
    if secondary is not None:
        if secondary not in records:
            errors.append(f"unknown secondary_reference_id: {secondary}")
        elif not records[secondary]["selectable"]:
            errors.append(f"secondary reference is not selectable: {secondary}")
    if require_review_complete:
        checks = manifest.get("review_checks", {})
        for key, value in checks.items():
            if value is not True:
                errors.append(f"review check incomplete: {key}")
        if manifest.get("review_status") != "approved":
            errors.append("review_status must be approved")
    return errors
