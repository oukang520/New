"""Deprecated crop-based single-panel exporter.

The project now uses experiment scripts to render true standalone figures.
Keeping crop-derived panels as publication sources caused misleading truncated
or fragment-like images, so this exporter is retained only as historical code.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_ROOT = RESULTS_ROOT / "standardized_single_panels"
DPI = 600


EXCLUDE_PARTS = {
    "legacy_merged_sources",
    "figure_style_palette_options",
    "__pycache__",
}


TARGETS = {
    "large_square": {
        "px": (5600, 5600),
        "mm": (237.1, 237.1),
        "margin_px": 120,
        "min_final_width_mm": 160.0,
    },
    "large_wide": {
        "px": (7200, 4200),
        "mm": (304.8, 177.8),
        "margin_px": 120,
        "min_final_width_mm": 183.0,
    },
    "square_main": {
        "px": (2244, 2244),
        "mm": (95.0, 95.0),
        "margin_px": 70,
        "min_final_width_mm": 82.0,
    },
    "square_subpanel": {
        "px": (1536, 1536),
        "mm": (65.0, 65.0),
        "margin_px": 54,
        "min_final_width_mm": 52.0,
    },
    "wide_main": {
        "px": (4323, 2244),
        "mm": (183.0, 95.0),
        "margin_px": 76,
        "min_final_width_mm": 160.0,
    },
    "table_wide": {
        "px": (4323, 1536),
        "mm": (183.0, 65.0),
        "margin_px": 64,
        "min_final_width_mm": 170.0,
    },
}


@dataclass(frozen=True)
class ManualPanel:
    label: str
    fractions: tuple[float, float, float, float]
    target: str | None = None


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def expand(self, margin: int, width: int, height: int) -> "CropBox":
        return CropBox(
            max(0, self.left - margin),
            max(0, self.top - margin),
            min(width, self.right + margin),
            min(height, self.bottom + margin),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


def crop_from_fraction(width: int, height: int, fractions: tuple[float, float, float, float]) -> CropBox:
    left, top, right, bottom = fractions
    return CropBox(
        max(0, min(width - 1, int(round(left * width)))),
        max(0, min(height - 1, int(round(top * height)))),
        max(1, min(width, int(round(right * width)))),
        max(1, min(height, int(round(bottom * height)))),
    )


def trim_box_to_ink(image: Image.Image, box: CropBox, pad: int = 44) -> CropBox:
    crop = image.crop(box.as_tuple())
    local_mask = ink_mask(crop)
    local = bbox_from_mask(local_mask)
    if local is None:
        return box
    trimmed = CropBox(
        box.left + local.left,
        box.top + local.top,
        box.left + local.right,
        box.top + local.bottom,
    )
    return trimmed.expand(pad, image.width, image.height)


def safe_stem(path: Path) -> str:
    relative = path.relative_to(RESULTS_ROOT).with_suffix("")
    text = "__".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()) or "PANEL"


def source_figures() -> list[Path]:
    figures: list[Path] = []
    for path in RESULTS_ROOT.rglob("*.png"):
        if any(part in EXCLUDE_PARTS for part in path.relative_to(RESULTS_ROOT).parts):
            continue
        if "standardized_single_panels" in path.relative_to(RESULTS_ROOT).parts:
            continue
        if "experiment_07_topology_robustness" in path.relative_to(RESULTS_ROOT).parts:
            continue
        if "supplementary_four_cohort_screen" in path.relative_to(RESULTS_ROOT).parts:
            continue
        if not (path.name.startswith("Figure_") or path.name.startswith("Table_")):
            continue
        figures.append(path)
    return sorted(figures)


def grid_2x2(
    top: float = 0.10,
    mid_y: float = 0.52,
    left: float = 0.02,
    mid_x: float = 0.50,
    right: float = 0.985,
    bottom: float = 0.985,
) -> list[ManualPanel]:
    return [
        ManualPanel("A", (left, top, mid_x, mid_y), "square_main"),
        ManualPanel("B", (mid_x, top, right, mid_y), "square_main"),
        ManualPanel("C", (left, mid_y, mid_x, bottom), "square_main"),
        ManualPanel("D", (mid_x, mid_y, right, bottom), "square_main"),
    ]


def manual_panels_for_source(source: Path) -> list[ManualPanel] | None:
    rel = source.relative_to(RESULTS_ROOT).as_posix()
    name = source.name

    if name == "Table_E17_integrated_longitudinal_metrics.png":
        return [ManualPanel("TABLE", (0.00, 0.00, 1.00, 1.00), "table_wide")]

    if name == "Figure_E3_MHN_interface.png" and "combined_figures" not in rel:
        return [
            ManualPanel("A", (0.06, 0.04, 0.49, 0.47), "wide_main"),
            ManualPanel("B", (0.57, 0.04, 0.985, 0.47), "square_main"),
            ManualPanel("C", (0.00, 0.49, 0.51, 0.985), "wide_main"),
            ManualPanel("D", (0.58, 0.49, 0.985, 0.985), "wide_main"),
        ]

    if name == "Figure_E3_MHN_interface_three_cohorts.png":
        return [
            ManualPanel("LUAD_CV", (0.00, 0.105, 0.32, 0.365), "square_main"),
            ManualPanel("LUAD_ONE_STEP", (0.32, 0.115, 0.665, 0.385), "square_main"),
            ManualPanel("LUAD_EDGES", (0.675, 0.105, 1.00, 0.365), "square_main"),
            ManualPanel("COAD_CV", (0.00, 0.360, 0.32, 0.640), "square_main"),
            ManualPanel("COAD_ONE_STEP", (0.32, 0.420, 0.665, 0.690), "square_main"),
            ManualPanel("COAD_EDGES", (0.675, 0.360, 1.00, 0.640), "square_main"),
            ManualPanel("IDC_CV", (0.00, 0.635, 0.32, 1.00), "square_main"),
            ManualPanel("IDC_ONE_STEP", (0.32, 0.728, 0.665, 0.997), "square_main"),
            ManualPanel("IDC_EDGES", (0.675, 0.635, 1.00, 1.00), "square_main"),
        ]

    if name == "Figure_E4_relative_inflow.png" and "combined_figures" not in rel:
        return [
            ManualPanel("A", (0.04, 0.09, 0.51, 0.48), "square_main"),
            ManualPanel("B", (0.56, 0.09, 0.99, 0.48), "wide_main"),
            ManualPanel("C", (0.02, 0.52, 0.51, 0.98), "wide_main"),
            ManualPanel("D", (0.56, 0.52, 0.99, 0.98), "square_main"),
        ]

    if name == "Figure_E4_relative_inflow_three_cohorts.png":
        return [
            ManualPanel("A", (0.00, 0.08, 0.50, 0.52), "square_main"),
            ManualPanel("B", (0.50, 0.08, 1.00, 0.52), "square_main"),
            ManualPanel("C", (0.00, 0.55, 0.50, 1.00), "square_main"),
        ]

    if name == "Figure_E4_inflow_rule_sensitivity_three_cohorts.png":
        return [ManualPanel("TABLE", (0.00, 0.00, 1.00, 1.00), "large_square")]

    if name == "Figure_E4_dominant_inflow_edges_three_cohorts.png":
        return [ManualPanel("A", (0.00, 0.00, 1.00, 1.00), "large_square")]

    if name == "Figure_E5_core_results_three_cohorts.png":
        return [
            ManualPanel("A", (0.00, 0.10, 0.455, 0.51), "square_main"),
            ManualPanel("B", (0.47, 0.10, 0.88, 0.51), "square_main"),
            ManualPanel("C", (0.00, 0.55, 0.455, 0.99), "square_main"),
            ManualPanel("LEGEND", (0.84, 0.28, 1.00, 0.78), "square_subpanel"),
        ]

    if name == "Figure_E5_state_scores.png":
        return [
            ManualPanel("A", (0.00, 0.04, 0.51, 0.50), "wide_main"),
            ManualPanel("B", (0.52, 0.04, 1.00, 0.50), "square_main"),
            ManualPanel("C", (0.00, 0.53, 0.50, 1.00), "square_main"),
            ManualPanel("D", (0.52, 0.53, 1.00, 1.00), "square_main"),
        ]

    if name == "Figure_E6_enhanced_bottleneck_recovery.png":
        return [
            ManualPanel("A", (0.00, 0.035, 0.47, 0.51), "square_main"),
            ManualPanel("B", (0.47, 0.035, 1.00, 0.51), "wide_main"),
            ManualPanel("C", (0.00, 0.51, 0.47, 1.00), "square_main"),
            ManualPanel("D", (0.47, 0.51, 1.00, 1.00), "wide_main"),
        ]

    if name == "Figure_E6_continuous_dwell_gradient.png":
        return [
            ManualPanel("A", (0.00, 0.08, 0.50, 0.49), "square_main"),
            ManualPanel("B", (0.50, 0.08, 1.00, 0.49), "square_main"),
            ManualPanel("C", (0.00, 0.56, 0.50, 1.00), "square_main"),
            ManualPanel("D", (0.50, 0.56, 1.00, 1.00), "square_main"),
        ]

    if name == "Figure_E7_topology_robustness.png":
        return [ManualPanel("D", (0.00, 0.00, 1.00, 1.00), "wide_main")]

    if name == "Figure_E8_module_convergence_summary.png":
        return [
            ManualPanel("A", (0.00, 0.10, 1.00, 0.58), "wide_main"),
            ManualPanel("B", (0.10, 0.58, 0.48, 1.00), "square_main"),
            ManualPanel("C", (0.55, 0.58, 0.90, 1.00), "square_main"),
        ]

    if name == "Figure_E8_top_state_profiles.png":
        return [
            ManualPanel("A", (0.00, 0.10, 0.50, 0.52), "wide_main"),
            ManualPanel("B", (0.50, 0.10, 1.00, 0.52), "wide_main"),
            ManualPanel("C", (0.26, 0.52, 0.83, 1.00), "wide_main"),
        ]

    if name == "Figure_E8_top_state_legend.png":
        return [ManualPanel("LEGEND", (0.00, 0.00, 1.00, 1.00), "large_wide")]

    if name == "Figure_E9_observation_enrichment.png":
        return grid_2x2(top=0.09, mid_y=0.51, left=0.05, mid_x=0.50, right=0.98, bottom=0.985)

    if name in {
        "Figure_E10_real_cohort_main_results.png",
        "Figure_E11_information_gain_controls.png",
        "Figure_E14_ablation_backbone.png",
    }:
        return grid_2x2(top=0.08, mid_y=0.52, left=0.02, mid_x=0.50, right=0.985, bottom=0.985)

    if name == "Figure_E12_clinical_validation.png":
        return [
            ManualPanel("A", (0.00, 0.16, 1.00, 0.55), "wide_main"),
            ManualPanel("B", (0.00, 0.59, 1.00, 1.00), "wide_main"),
        ]

    if name == "Figure_E13_split_replication.png":
        return [
            ManualPanel("A", (0.00, 0.24, 0.51, 0.86), "square_main"),
            ManualPanel("B", (0.52, 0.10, 0.99, 0.43), "square_main"),
            ManualPanel("C", (0.52, 0.48, 0.99, 0.90), "square_main"),
        ]

    if name == "Figure_E15_innovation_falsification_controls.png":
        return [
            ManualPanel("A", (0.00, 0.08, 1.00, 0.52), "wide_main"),
            ManualPanel("B", (0.00, 0.52, 1.00, 0.96), "wide_main"),
        ]

    if name == "Figure_E16_real_relative_dwell_topology.png":
        return [
            ManualPanel("A", (0.00, 0.08, 0.51, 0.48), "wide_main"),
            ManualPanel("B", (0.49, 0.08, 1.00, 0.48), "wide_main"),
            ManualPanel("C", (0.00, 0.48, 0.68, 1.00), "wide_main"),
        ]

    if name == "Figure_E17_integrated_longitudinal_validation.png":
        return [
            ManualPanel("A", (0.00, 0.00, 0.45, 0.34), "square_main"),
            ManualPanel("B", (0.52, 0.00, 1.00, 0.34), "square_main"),
            ManualPanel("C_GLASS", (0.00, 0.37, 0.34, 0.70), "square_subpanel"),
            ManualPanel("C_CRC", (0.39, 0.37, 0.64, 0.70), "square_subpanel"),
            ManualPanel("C_MNM", (0.74, 0.37, 1.00, 0.70), "square_subpanel"),
            ManualPanel("D_GLASS", (0.00, 0.73, 0.34, 1.00), "square_subpanel"),
            ManualPanel("D_CRC", (0.39, 0.73, 0.64, 1.00), "square_subpanel"),
            ManualPanel("D_MNM", (0.74, 0.73, 1.00, 1.00), "square_subpanel"),
        ]

    if name == "Figure_E17_real_longitudinal_topology_routes.png":
        return [
            ManualPanel("A", (0.00, 0.00, 0.47, 0.50), "square_main"),
            ManualPanel("B", (0.50, 0.00, 1.00, 0.50), "square_main"),
            ManualPanel("C", (0.00, 0.50, 0.67, 1.00), "square_main"),
        ]

    if name == "Figure_E1_QC_overview_three_cohorts.png":
        return grid_2x2(top=0.07, mid_y=0.51, left=0.02, mid_x=0.50, right=0.985, bottom=0.985)

    if name == "Figure_E1_state_sparsity_three_cohorts.png":
        return grid_2x2(top=0.08, mid_y=0.52, left=0.02, mid_x=0.50, right=0.985, bottom=0.985)

    if name == "Figure_E1_oncoprint.png":
        return [ManualPanel("ONCOPRINT", (0.00, 0.00, 1.00, 1.00), "large_wide")]

    if name in {
        "Figure_E1_QC_overview.png",
        "Figure_E1_state_sparsity.png",
        "Figure_E2_stage_event_heatmaps.png",
        "Figure_E2_stage_sensitivity.png",
    }:
        return [ManualPanel("FULL", (0.00, 0.00, 1.00, 1.00), "large_wide")]

    return None


def ink_mask(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.int16)
    alpha = arr[..., 3] > 12
    rgb = arr[..., :3]
    # Treat near-white antialiasing as background, keep grey grid/text/lines.
    non_white = np.max(np.abs(rgb - 255), axis=2) > 14
    return alpha & non_white


def bbox_from_mask(mask: np.ndarray) -> CropBox | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return CropBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def runs_of_true(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def box_has_enough_ink(mask: np.ndarray, box: CropBox) -> bool:
    if box.width < 260 or box.height < 220:
        return False
    area = box.area
    if area <= 0:
        return False
    ink = int(mask[box.top : box.bottom, box.left : box.right].sum())
    return ink >= 700 and ink / area >= 0.0015


def best_gap(mask: np.ndarray, box: CropBox, axis: str) -> tuple[int, int, float] | None:
    region = mask[box.top : box.bottom, box.left : box.right]
    if axis == "x":
        projection = region.sum(axis=0)
        orthogonal = box.height
        length = box.width
    else:
        projection = region.sum(axis=1)
        orthogonal = box.width
        length = box.height

    low_threshold = max(2.0, orthogonal * 0.003)
    low = projection <= low_threshold
    min_gap = max(58, int(round(length * 0.035)))
    candidates: list[tuple[int, int, float]] = []
    for start, end in runs_of_true(low):
        gap_width = end - start
        if gap_width < min_gap:
            continue
        center = (start + end) / 2.0
        if center < length * 0.16 or center > length * 0.84:
            continue
        if axis == "x":
            left_box = CropBox(box.left, box.top, box.left + start, box.bottom)
            right_box = CropBox(box.left + end, box.top, box.right, box.bottom)
            absolute = (box.left + start, box.left + end)
        else:
            left_box = CropBox(box.left, box.top, box.right, box.top + start)
            right_box = CropBox(box.left, box.top + end, box.right, box.bottom)
            absolute = (box.top + start, box.top + end)
        if not (box_has_enough_ink(mask, left_box) and box_has_enough_ink(mask, right_box)):
            continue
        score = gap_width / length
        candidates.append((absolute[0], absolute[1], score))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2])


def split_box(mask: np.ndarray, box: CropBox, depth: int = 0) -> list[CropBox]:
    if depth >= 5 or box.width < 680 or box.height < 560:
        return [box]
    x_gap = best_gap(mask, box, "x")
    y_gap = best_gap(mask, box, "y")
    gaps = [("x", x_gap), ("y", y_gap)]
    gaps = [(axis, gap) for axis, gap in gaps if gap is not None]
    if not gaps:
        return [box]
    axis, gap = max(gaps, key=lambda item: item[1][2])
    start, end, score = gap
    if score < 0.035:
        return [box]
    if axis == "x":
        children = [
            CropBox(box.left, box.top, start, box.bottom),
            CropBox(end, box.top, box.right, box.bottom),
        ]
    else:
        children = [
            CropBox(box.left, box.top, box.right, start),
            CropBox(box.left, end, box.right, box.bottom),
        ]
    leaves: list[CropBox] = []
    for child in children:
        if box_has_enough_ink(mask, child):
            leaves.extend(split_box(mask, child, depth + 1))
    return leaves or [box]


def merge_tiny_label_boxes(boxes: list[CropBox], image_width: int, image_height: int) -> list[CropBox]:
    if len(boxes) <= 1:
        return boxes
    # Keep this conservative. The recursive splitter mostly returns panel-sized
    # boxes, but isolated panel letters can occasionally become tiny boxes.
    large = [box for box in boxes if box.width > 420 and box.height > 360]
    tiny = [box for box in boxes if box not in large]
    for tiny_box in tiny:
        if not large:
            large.append(tiny_box)
            continue
        distances = []
        for index, box in enumerate(large):
            dx = max(box.left - tiny_box.right, tiny_box.left - box.right, 0)
            dy = max(box.top - tiny_box.bottom, tiny_box.top - box.bottom, 0)
            distances.append((math.hypot(dx, dy), index))
        _, nearest = min(distances, key=lambda item: item[0])
        box = large[nearest]
        large[nearest] = CropBox(
            min(box.left, tiny_box.left),
            min(box.top, tiny_box.top),
            max(box.right, tiny_box.right),
            max(box.bottom, tiny_box.bottom),
        ).expand(12, image_width, image_height)
    return large


def sort_boxes(boxes: list[CropBox]) -> list[CropBox]:
    if not boxes:
        return []
    heights = np.array([box.height for box in boxes], dtype=float)
    row_tolerance = max(120.0, float(np.median(heights) * 0.45))
    sorted_by_top = sorted(boxes, key=lambda box: (box.top, box.left))
    rows: list[list[CropBox]] = []
    for box in sorted_by_top:
        placed = False
        for row in rows:
            if abs(np.mean([item.top for item in row]) - box.top) <= row_tolerance:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    rows.sort(key=lambda row: np.mean([box.top for box in row]))
    ordered: list[CropBox] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda box: box.left))
    return ordered


def panel_boxes(image: Image.Image) -> list[CropBox]:
    mask = ink_mask(image)
    full = bbox_from_mask(mask)
    if full is None:
        return [CropBox(0, 0, image.width, image.height)]
    # Use direct row/column ink projections rather than heavy morphology.
    # This finds true white gutters in composite figures and avoids shredding
    # dense plots/tables into local visual fragments.
    split_mask = mask
    full = bbox_from_mask(split_mask) or full
    full = full.expand(34, image.width, image.height)
    boxes = split_box(split_mask, full)
    boxes = merge_tiny_label_boxes(boxes, image.width, image.height)
    boxes = [box.expand(28, image.width, image.height) for box in boxes]
    boxes = [box for box in boxes if box_has_enough_ink(mask, box)]
    boxes = sort_boxes(boxes)
    # Dense tables and oncoprints should not be shredded into small visual
    # fragments. If the split appears too aggressive, keep the whole content.
    if len(boxes) > 12:
        return [full.expand(24, image.width, image.height)]
    return boxes or [full.expand(24, image.width, image.height)]


def reset_output_root() -> None:
    resolved_output = OUTPUT_ROOT.resolve()
    resolved_results = RESULTS_ROOT.resolve()
    if resolved_output.name != "standardized_single_panels":
        raise RuntimeError(f"Refusing to reset unexpected output directory: {resolved_output}")
    if resolved_results not in resolved_output.parents:
        raise RuntimeError(f"Refusing to reset directory outside results: {resolved_output}")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)


def target_for_crop(source_path: Path, box: CropBox) -> str:
    aspect = box.width / max(box.height, 1)
    if source_path.name.startswith("Table_"):
        return "table_wide"
    if aspect >= 1.55:
        return "wide_main"
    if aspect <= 0.64:
        return "square_main"
    if box.width < 1700 and box.height < 1700:
        return "square_subpanel"
    return "square_main"


def scale_for_target(crop: Image.Image, target_name: str) -> float:
    target = TARGETS[target_name]
    target_w, target_h = target["px"]
    margin = int(target["margin_px"])
    max_w = target_w - 2 * margin
    max_h = target_h - 2 * margin
    scale = min(max_w / crop.width, max_h / crop.height)
    # Allow mild upscaling for small-multiple panels, but keep raster text from
    # being inflated too far. Native panels remain available when needed.
    return min(scale, 1.35)


def choose_readable_target(crop: Image.Image, target_name: str) -> str:
    initial_scale = scale_for_target(crop, target_name)
    if initial_scale >= 0.72:
        return target_name
    aspect = crop.width / max(crop.height, 1)
    if target_name == "table_wide":
        candidates = ["table_wide", "large_wide" if aspect >= 1.25 else "large_square"]
    elif aspect >= 1.45:
        candidates = [target_name, "large_wide"]
    else:
        candidates = [target_name, "large_square"]
    best_name = target_name
    best_scale = initial_scale
    for candidate in candidates:
        scale = scale_for_target(crop, candidate)
        if scale > best_scale:
            best_name = candidate
            best_scale = scale
        if scale >= 0.72:
            return candidate
    return best_name


def place_on_canvas(crop: Image.Image, target_name: str) -> tuple[Image.Image, float]:
    scale = scale_for_target(crop, target_name)
    target = TARGETS[target_name]
    target_w, target_h = target["px"]
    margin = int(target["margin_px"])
    max_w = target_w - 2 * margin
    max_h = target_h - 2 * margin
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))
    resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS) if scale != 1 else crop
    canvas = Image.new("RGB", (target_w, target_h), "white")
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized.convert("RGB"), (x, y))
    return canvas, scale


def edge_ink_fraction(image: Image.Image, border_fraction: float = 0.012) -> float:
    mask = ink_mask(image)
    edge = max(1, int(round(min(image.width, image.height) * border_fraction)))
    fractions = [
        mask[:edge, :].mean(),
        mask[-edge:, :].mean(),
        mask[:, :edge].mean(),
        mask[:, -edge:].mean(),
    ]
    return float(max(fractions))


def export_panels() -> None:
    reset_output_root()
    native_dir = OUTPUT_ROOT / "native_panels"
    standard_dir = OUTPUT_ROOT / "standardized_panels"
    pdf_dir = OUTPUT_ROOT / "standardized_panel_pdfs"
    native_dir.mkdir(parents=True, exist_ok=True)
    standard_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    sources = source_figures()
    print(f"Exporting {len(sources)} source figures to {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}", flush=True)
    for source_index, source in enumerate(sources, start=1):
        print(f"[{source_index:02d}/{len(sources):02d}] {source.relative_to(PROJECT_ROOT)}", flush=True)
        with Image.open(source) as handle:
            image = handle.convert("RGB")
        manual_panels = manual_panels_for_source(source)
        if manual_panels:
            source_mask = ink_mask(image)
            panel_units = []
            for manual in manual_panels:
                broad_box = crop_from_fraction(image.width, image.height, manual.fractions)
                source_crop_edge = edge_ink_fraction(image.crop(broad_box.as_tuple()), border_fraction=0.006)
                box = trim_box_to_ink(image, broad_box)
                if box_has_enough_ink(source_mask, box):
                    panel_units.append((manual.label, box, manual.target, "manual", source_crop_edge))
        else:
            panel_units = [
                (
                    f"P{index:02d}",
                    box,
                    None,
                    "auto",
                    edge_ink_fraction(image.crop(box.as_tuple()), border_fraction=0.006),
                )
                for index, box in enumerate(panel_boxes(image), start=1)
            ]
        source_id = safe_stem(source)
        for index, (panel_label, box, target_override, split_method, source_crop_edge) in enumerate(panel_units, start=1):
            crop = image.crop(box.as_tuple())
            target_name = target_override or target_for_crop(source, box)
            target_name = choose_readable_target(crop, target_name)
            canvas, scale = place_on_canvas(crop, target_name)
            panel_id = f"{source_id}__{safe_label(panel_label)}"
            native_path = native_dir / f"{panel_id}.png"
            standard_path = standard_dir / f"{panel_id}.png"
            pdf_path = pdf_dir / f"{panel_id}.pdf"
            crop.save(native_path, dpi=(DPI, DPI))
            canvas.save(standard_path, dpi=(DPI, DPI))
            canvas.save(pdf_path, "PDF", resolution=DPI)

            target = TARGETS[target_name]
            content_fraction = float(ink_mask(canvas).mean())
            edge_ink = edge_ink_fraction(canvas)
            scale_status = "PASS" if 0.72 <= scale <= 1.35 else "WARN"
            edge_status = "PASS" if edge_ink <= 0.05 else "WARN"
            source_crop_edge_status = "PASS" if source_crop_edge <= 0.085 else "REVIEW"
            audit_status = "PASS" if scale_status == "PASS" and edge_status == "PASS" else "WARN"
            manifest_rows.append(
                {
                    "panel_id": panel_id,
                    "source_figure": str(source.relative_to(PROJECT_ROOT)),
                    "panel_index": index,
                    "panel_label": panel_label,
                    "split_method": split_method,
                    "source_width_px": image.width,
                    "source_height_px": image.height,
                    "crop_left": box.left,
                    "crop_top": box.top,
                    "crop_right": box.right,
                    "crop_bottom": box.bottom,
                    "native_width_px": crop.width,
                    "native_height_px": crop.height,
                    "standard_class": target_name,
                    "standard_width_px": target["px"][0],
                    "standard_height_px": target["px"][1],
                    "standard_width_mm": target["mm"][0],
                    "standard_height_mm": target["mm"][1],
                    "minimum_recommended_final_width_mm": target["min_final_width_mm"],
                    "scale_factor": round(scale, 3),
                    "native_png": str(native_path.relative_to(PROJECT_ROOT)),
                    "standard_png": str(standard_path.relative_to(PROJECT_ROOT)),
                    "standard_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
                }
            )
            audit_rows.append(
                {
                    "panel_id": panel_id,
                    "standard_class": target_name,
                    "standard_width_px": canvas.width,
                    "standard_height_px": canvas.height,
                    "dpi": DPI,
                    "content_fraction": round(content_fraction, 4),
                    "edge_ink_max": round(edge_ink, 4),
                    "source_crop_edge_ink_max": round(source_crop_edge, 4),
                    "scale_factor": round(scale, 3),
                    "scale_status": scale_status,
                    "edge_status": edge_status,
                    "source_crop_edge_status": source_crop_edge_status,
                    "status": audit_status,
                }
            )

    for filename, rows in [
        ("panel_manifest.tsv", manifest_rows),
        ("panel_audit.tsv", audit_rows),
    ]:
        path = OUTPUT_ROOT / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_readme(len(manifest_rows))


def write_readme(panel_count: int) -> None:
    text = f"""# Standardized Single Figure Panels

This directory contains standardized single-panel exports for later manuscript
figure assembly.

## Outputs

- `native_panels/`: direct crops from source figures, no resizing.
- `standardized_panels/`: 600-dpi PNG panels on fixed white canvases.
- `standardized_panel_pdfs/`: raster-PDF versions of the standardized panels.
- `panel_manifest.tsv`: source path, crop coordinates, size class and recommended
  physical width.
- `panel_audit.tsv`: DPI, scale and border-clipping audit.
- `source_crop_edge_ink_max` in `panel_audit.tsv`: manual source-window
  boundary audit for detecting over-tight cuts before the panel is trimmed and
  placed on a white canvas.
- `reassembly_index.tsv`, `main_text_panel_index.tsv`,
  `supplementary_panel_index.tsv`, `table_panel_index.tsv` and
  `panel_gallery.html` are generated by `src/build_panel_reassembly_index.py`
  after export.
- `panel_contact_sheet.png`, `source_window_review_sheet.png` and
  `panel_visual_qc.md` are generated by `src/audit_standardized_panels.py`
  after export.

## Style Contract

- Font family inherited from the source figures: Arial.
- Primary text color inherited from the shared figure style: `#263238`.
- Background: white.
- Output resolution: 600 dpi.
- Standard classes:
  - `large_square`: 237 mm x 237 mm, used only when a dense panel would become
    too small on the regular square canvas.
  - `large_wide`: 305 mm x 178 mm, used only for dense wide panels or oncoprints.
  - `square_main`: 95 mm x 95 mm.
  - `square_subpanel`: 65 mm x 65 mm.
  - `wide_main`: 183 mm x 95 mm.
  - `table_wide`: 183 mm x 65 mm.

## Assembly Guidance

Use the `standardized_panels` files for layout drafts. Use `native_panels` if a
panel needs custom scaling in Illustrator, PowerPoint or Inkscape.

To keep text readable after regrouping, do not reduce a panel below the
`minimum_recommended_final_width_mm` recorded in `panel_manifest.tsv`. This is
especially important for table panels and small-multiple calibration panels.

Generated panel count: {panel_count}.
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated: crop-based standardized panel export is disabled. "
        "Use direct single figures produced by each experiment script."
    )
