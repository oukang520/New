"""Shared top-journal figure style utilities for Rel-ObsTQ-MHN experiments."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


DEFAULT_STYLE_CONFIG = Path("configs/figure_style.yaml")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return _project_root() / DEFAULT_STYLE_CONFIG
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def load_style(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the shared figure style, allowing an experiment to override the path."""
    config = config or {}
    style_path = (
        config.get("plot_style_config")
        or config.get("plot", {}).get("style_config")
        or DEFAULT_STYLE_CONFIG
    )
    resolved = _resolve_path(style_path)
    if not resolved.exists():
        return {}
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def merged_plot_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return shared plot defaults with experiment-level plot values taking precedence."""
    config = config or {}
    style = load_style(config)
    plot = dict(style.get("plot", {}))
    plot.update(config.get("plot", {}))
    return plot


def configure_matplotlib(config: dict[str, Any] | None = None) -> None:
    """Apply shared matplotlib settings plus per-experiment plot overrides."""
    plot = merged_plot_config(config)
    rcparams = {
        "font.family": plot.get("font_family", "Arial"),
        "font.size": plot.get("base_font_size", 8),
    }
    rcparams.update(plot.get("rcparams", {}))
    sns.set_style("ticks")
    matplotlib.rcParams.update(rcparams)


def save_figure(
    fig: plt.Figure,
    base_path: str | Path,
    config: dict[str, Any] | None = None,
    dpi: int | None = None,
    pad_inches: float | None = None,
) -> None:
    """Save a figure using the shared output rules."""
    plot = merged_plot_config(config)
    output = Path(base_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    formats = plot.get("output_formats", ["pdf", "png"])
    save_dpi = int(dpi if dpi is not None else plot.get("dpi", 600))
    bbox_inches = plot.get("bbox_inches", "tight")
    pad = float(pad_inches if pad_inches is not None else plot.get("pad_inches", 0.04))
    for fmt in formats:
        suffix = f".{fmt.lstrip('.')}"
        if suffix == ".png":
            fig.savefig(output.with_suffix(suffix), dpi=save_dpi, bbox_inches=bbox_inches, pad_inches=pad)
        else:
            fig.savefig(output.with_suffix(suffix), bbox_inches=bbox_inches, pad_inches=pad)
    plt.close(fig)


def _safe_panel_slug(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\$[^$]*\$", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    return (text[:72].strip("_") or fallback).lower()


def _axis_area(ax: plt.Axes) -> float:
    box = ax.get_position()
    return max(float(box.width * box.height), 0.0)


def _axis_has_user_content(ax: plt.Axes) -> bool:
    if ax.images or ax.lines or ax.collections or ax.containers or ax.texts or ax.tables:
        return True
    visible_patches = [
        patch for patch in ax.patches if patch.get_visible() and patch is not ax.patch
    ]
    return bool(visible_patches)


def _is_colorbar_axis(ax: plt.Axes) -> bool:
    label = str(ax.get_label()).lower()
    if "colorbar" in label:
        return True
    ylabel = ax.get_ylabel().lower()
    xlabel = ax.get_xlabel().lower()
    return bool(label == "<colorbar>" or "colorbar" in ylabel or "colorbar" in xlabel)


def _axis_overlap(a: plt.Axes, b: plt.Axes) -> float:
    box_a = ax_box = a.get_position()
    box_b = b.get_position()
    x_overlap = max(0.0, min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0))
    y_overlap = max(0.0, min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0))
    denom = min(ax_box.width * ax_box.height, box_b.width * box_b.height)
    return (x_overlap * y_overlap / denom) if denom > 0 else 0.0


def _axis_shared_span(a: plt.Axes, b: plt.Axes) -> float:
    box_a = a.get_position()
    box_b = b.get_position()
    x_overlap = max(0.0, min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0))
    y_overlap = max(0.0, min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0))
    x_fraction = x_overlap / min(box_a.width, box_b.width) if min(box_a.width, box_b.width) > 0 else 0.0
    y_fraction = y_overlap / min(box_a.height, box_b.height) if min(box_a.height, box_b.height) > 0 else 0.0
    return max(float(x_fraction), float(y_fraction))


def _center_distance(a: plt.Axes, b: plt.Axes) -> float:
    box_a = a.get_position()
    box_b = b.get_position()
    ax_center = ((box_a.x0 + box_a.x1) / 2.0, (box_a.y0 + box_a.y1) / 2.0)
    bx_center = ((box_b.x0 + box_b.x1) / 2.0, (box_b.y0 + box_b.y1) / 2.0)
    return float(np.hypot(ax_center[0] - bx_center[0], ax_center[1] - bx_center[1]))


def _classify_panel_axes(fig: plt.Figure) -> tuple[list[plt.Axes], dict[plt.Axes, list[plt.Axes]]]:
    visible = [
        ax
        for ax in fig.axes
        if ax.get_visible() and (_is_colorbar_axis(ax) or _axis_has_user_content(ax))
    ]
    non_color = [ax for ax in visible if not _is_colorbar_axis(ax)]
    if not non_color:
        return [], {}

    areas = np.array([_axis_area(ax) for ax in non_color], dtype=float)
    median_area = float(np.median(areas[areas > 0])) if np.any(areas > 0) else 0.0
    large_axes = [
        ax
        for ax in non_color
        if median_area <= 0 or _axis_area(ax) >= median_area * 0.34
    ]
    if not large_axes:
        large_axes = non_color

    primary: list[plt.Axes] = []
    auxiliary: list[plt.Axes] = []
    for ax in non_color:
        area = _axis_area(ax)
        if ax in large_axes:
            primary.append(ax)
            continue
        if median_area > 0 and area < median_area * 0.18:
            auxiliary.append(ax)
            continue
        nearest_large = min(large_axes, key=lambda candidate: _center_distance(ax, candidate))
        near = _center_distance(ax, nearest_large) < 0.22
        aligned = _axis_overlap(ax, nearest_large) > 0.08
        shared_span = _axis_shared_span(ax, nearest_large) > 0.70 and _center_distance(ax, nearest_large) < 0.48
        if near or aligned or shared_span:
            auxiliary.append(ax)
        else:
            primary.append(ax)

    groups = {ax: [] for ax in primary}
    for ax in visible:
        if ax in primary:
            continue
        nearest = min(primary, key=lambda candidate: _center_distance(ax, candidate))
        groups[nearest].append(ax)
    return primary, groups


def _infer_panel_name(ax: plt.Axes, index: int) -> str:
    for loc in ["left", "center", "right"]:
        title = ax.get_title(loc=loc).strip()
        if title:
            return title
    panel_label_pattern = re.compile(r"^[A-Za-z]$")
    for text in ax.texts:
        value = text.get_text().strip()
        if not value or panel_label_pattern.match(value):
            continue
        first_line = value.splitlines()[0].strip()
        if first_line:
            return first_line
    label = ax.get_ylabel() or ax.get_xlabel()
    if label:
        return label
    return f"panel_{index:02d}"


@contextmanager
def _hide_publication_extras(fig: plt.Figure):
    changed: list[tuple[Any, bool]] = []
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_visible():
        changed.append((suptitle, True))
        suptitle.set_visible(False)
    for text in fig.texts:
        if text.get_visible():
            changed.append((text, True))
            text.set_visible(False)
    panel_label_pattern = re.compile(r"^[A-Za-z]$")
    for ax in fig.axes:
        for text in ax.texts:
            value = text.get_text().strip()
            outside_axes = (
                text.get_transform() == ax.transAxes
                and (
                    text.get_position()[0] < 0.0
                    or text.get_position()[0] > 1.0
                    or text.get_position()[1] < 0.0
                    or text.get_position()[1] > 1.0
                )
            )
            if panel_label_pattern.match(value) or outside_axes:
                changed.append((text, text.get_visible()))
                text.set_visible(False)
    try:
        yield
    finally:
        for artist, visible in changed:
            artist.set_visible(visible)


def _panel_bbox_inches(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    pad_inches: float,
) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = []
    for ax in axes:
        if not ax.get_visible():
            continue
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            bbox = ax.bbox
        bboxes.append(bbox)
    if not bboxes:
        return fig.bbox_inches
    return Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted()).padded(pad_inches)


def _cleanup_panel_outputs(base_path: Path) -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        candidate = base.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()
    for candidate in base.parent.glob(f"{base.name}__*"):
        if candidate.suffix.lower() in {".png", ".pdf"}:
            candidate.unlink()


def save_figure_panels(
    fig: plt.Figure,
    base_path: str | Path,
    config: dict[str, Any] | None = None,
    dpi: int | None = None,
    panel_names: Sequence[str] | None = None,
    pad_inches: float | None = None,
) -> list[Path]:
    """Save each plotted panel in a figure as an independent PNG/PDF pair.

    The drawing code can still build a multi-panel Matplotlib figure. This
    output helper exports the visible primary axes as manuscript assembly
    units, while keeping nearby colorbars/insets with the closest panel and
    hiding figure-level titles and A/B/C panel letters during export.
    """
    plot = merged_plot_config(config)
    output = Path(base_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    formats = plot.get("output_formats", ["pdf", "png"])
    save_dpi = int(dpi if dpi is not None else plot.get("dpi", 600))
    pad = max(float(pad_inches if pad_inches is not None else plot.get("pad_inches", 0.04)), 0.12)
    rendered = quality_gates(config).get("rendered_png", {})
    min_width_px = int(rendered.get("min_width_px", 1400))
    min_height_px = int(rendered.get("min_height_px", 1000))
    primary_axes, groups = _classify_panel_axes(fig)
    if not primary_axes:
        save_figure(fig, output, config=config, dpi=save_dpi, pad_inches=pad)
        return [output.with_suffix(".png")]

    written: list[Path] = []
    multiple = len(primary_axes) > 1
    _cleanup_panel_outputs(output)
    inferred_names: list[str] = []
    used: dict[str, int] = {}
    for index, ax in enumerate(primary_axes, start=1):
        if panel_names and index <= len(panel_names):
            raw_name = panel_names[index - 1]
        else:
            raw_name = _infer_panel_name(ax, index)
        slug = _safe_panel_slug(raw_name, f"panel_{index:02d}")
        used[slug] = used.get(slug, 0) + 1
        if used[slug] > 1:
            slug = f"{slug}_{used[slug]:02d}"
        inferred_names.append(slug)

    with _hide_publication_extras(fig):
        for ax, slug in zip(primary_axes, inferred_names):
            panel_output = output.with_name(f"{output.name}__{slug}") if multiple else output
            bbox = _panel_bbox_inches(fig, [ax, *groups.get(ax, [])], pad)
            panel_dpi = max(
                save_dpi,
                int(np.ceil(min_width_px / max(float(bbox.width), 1e-6))),
                int(np.ceil(min_height_px / max(float(bbox.height), 1e-6))),
            )
            for fmt in formats:
                suffix = f".{fmt.lstrip('.')}"
                target = panel_output.with_suffix(suffix)
                if suffix == ".png":
                    fig.savefig(target, dpi=panel_dpi, bbox_inches=bbox, pad_inches=0)
                else:
                    fig.savefig(target, bbox_inches=bbox, pad_inches=0)
                if suffix == ".png":
                    written.append(target)
    plt.close(fig)
    return written


def rendered_panel_paths(base_path: str | Path, suffix: str = ".png") -> list[Path]:
    """Return rendered panel paths for a figure base path.

    When a multi-panel figure has been exported with :func:`save_figure_panels`,
    the original ``base.png`` is replaced by ``base__panel_name.png`` files.
    Single-panel figures keep the original base name.
    """
    base = Path(base_path)
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    direct = base.with_suffix(normalized_suffix)
    panels = sorted(base.parent.glob(f"{base.name}__*{normalized_suffix}"))
    if panels:
        return panels
    return [direct] if direct.exists() else []


def audit_rendered_figure_outputs(
    base_path: str | Path,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Audit all rendered PNG outputs belonging to one figure base path."""
    return [audit_rendered_png(path, config) for path in rendered_panel_paths(base_path, ".png")]


def colors(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return load_style(config).get("colors", {})


def continuous_palette(name: str, config: dict[str, Any] | None = None, fallback: list[str] | None = None) -> list[str]:
    palette = colors(config).get("continuous", {}).get(name)
    if palette:
        return list(palette)
    return list(fallback or ["#FEEBB9", "#B8D2CC", "#B2E6FD", "#B5AED5"])


def categorical_palette(config: dict[str, Any] | None = None) -> dict[str, str]:
    return dict(colors(config).get("categorical", {}))


def design_sources_markdown(config: dict[str, Any] | None = None) -> str:
    sources = load_style(config).get("top_journal_sources", [])
    if not sources:
        return "- Shared top-journal source list not configured."
    lines = []
    for source in sources:
        title = source.get("title", "Untitled source")
        venue = source.get("venue", "")
        year = source.get("year", "")
        url = source.get("url", "")
        design_use = source.get("design_use", "")
        label = f"{title}, {venue} {year}".strip().strip(",")
        lines.append(f"- {label}:")
        lines.append(f"  {url}")
        if design_use:
            lines.append(f"  Design use: {design_use}.")
    return "\n".join(lines)


def design_rules_markdown(config: dict[str, Any] | None = None) -> str:
    rules = load_style(config).get("rules", [])
    if not rules:
        return "- Shared design rules not configured."
    return "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))


def design_patterns_markdown(config: dict[str, Any] | None = None) -> str:
    pattern_block = load_style(config).get("reference_design_patterns", {})
    patterns = pattern_block.get("patterns", [])
    if not patterns:
        return "- Shared reference design patterns not configured."
    lines = []
    color_policy = pattern_block.get("color_policy")
    if color_policy:
        lines.append(f"- Color policy: {color_policy}.")
    for index, pattern in enumerate(patterns, start=1):
        name = str(pattern.get("name", f"pattern_{index}")).replace("_", " ")
        use = pattern.get("use", "")
        suitable_for = pattern.get("suitable_for", "")
        line = f"{index}. {name}: {use}"
        if suitable_for:
            line += f" Suitable for: {suitable_for}."
        lines.append(line)
    return "\n".join(lines)


def external_skill_sources_markdown(config: dict[str, Any] | None = None) -> str:
    """Return the GitHub/Codex skill sources used to shape the shared figure policy."""
    sources = load_style(config).get("external_skill_sources", [])
    if not sources:
        return "- External figure-design skill sources not configured."
    lines = []
    for source in sources:
        name = source.get("name", "unnamed-skill")
        url = source.get("source", "")
        role = source.get("role", "")
        adopted = source.get("adopted", "")
        lines.append(f"- {name}: {url}")
        if role:
            lines.append(f"  Role: {str(role).rstrip('.')}.")
        if adopted:
            lines.append(f"  Adopted: {str(adopted).rstrip('.')}.")
    return "\n".join(lines)


def quality_gates(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return shared rendered-output quality gates."""
    return dict(load_style(config).get("quality_gates", {}))


def audit_rendered_png(path: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit a rendered PNG for size, blankness and likely border clipping.

    This is a lightweight screening check. A warning means the figure deserves
    visual inspection, not that the scientific result is invalid.
    """
    from PIL import Image

    image_path = Path(path)
    rendered = quality_gates(config).get("rendered_png", {})
    min_width = int(rendered.get("min_width_px", 1400))
    min_height = int(rendered.get("min_height_px", 1000))
    min_content = float(rendered.get("min_content_fraction", 0.012))
    max_edge_ink = float(rendered.get("max_edge_ink_fraction", 0.18))
    border_fraction = float(rendered.get("border_scan_fraction", 0.012))
    aspect_low = float(rendered.get("flag_aspect_ratio_below", 0.45))
    aspect_high = float(rendered.get("flag_aspect_ratio_above", 4.20))

    with Image.open(image_path) as img:
        rgba = img.convert("RGBA")
        width, height = rgba.size
        arr = np.asarray(rgba, dtype=np.float32)

    alpha = arr[..., 3:4] / 255.0
    rgb = arr[..., :3] * alpha + 255.0 * (1.0 - alpha)
    ink = np.max(np.abs(rgb - 255.0), axis=2) > 12.0
    content_fraction = float(np.mean(ink))
    edge = max(1, int(round(min(width, height) * border_fraction)))
    edge_fractions = {
        "top": float(np.mean(ink[:edge, :])),
        "bottom": float(np.mean(ink[-edge:, :])),
        "left": float(np.mean(ink[:, :edge])),
        "right": float(np.mean(ink[:, -edge:])),
    }
    max_edge = max(edge_fractions.values())
    aspect_ratio = width / height if height else float("inf")

    warnings: list[str] = []
    if width < min_width or height < min_height:
        warnings.append("small_png")
    if content_fraction < min_content:
        warnings.append("low_content")
    if max_edge > max_edge_ink:
        warnings.append("possible_border_clipping")
    if aspect_ratio < aspect_low or aspect_ratio > aspect_high:
        warnings.append("extreme_aspect_ratio")

    return {
        "path": str(image_path),
        "width_px": width,
        "height_px": height,
        "aspect_ratio": round(aspect_ratio, 3),
        "content_fraction": round(content_fraction, 4),
        "edge_ink_max": round(max_edge, 4),
        "edge_top": round(edge_fractions["top"], 4),
        "edge_bottom": round(edge_fractions["bottom"], 4),
        "edge_left": round(edge_fractions["left"], 4),
        "edge_right": round(edge_fractions["right"], 4),
        "status": "PASS" if not warnings else "WARN",
        "warnings": ";".join(warnings),
    }


def audit_figure_tree(root: str | Path, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Audit all PNG files below a result tree."""
    return [audit_rendered_png(path, config) for path in sorted(Path(root).rglob("*.png"))]
