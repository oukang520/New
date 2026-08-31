"""Audit rendered experiment figures against the shared Rel-ObsTQ-MHN style gates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import figure_style
import yaml
from PIL import Image, ImageDraw, ImageFont


def experiment_label(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = relative.parts
    if not parts:
        return "unknown"
    for part in parts:
        if part.startswith("experiment"):
            return part
        if part.startswith("experiments_"):
            return part
    return parts[0]


def audit_roots(roots: Iterable[Path], config: dict) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for row in figure_style.audit_figure_tree(root, config):
            row = dict(row)
            row["experiment"] = experiment_label(Path(row["path"]), root)
            rows.append(row)
    return sorted(rows, key=lambda item: (item["experiment"], item["path"]))


def load_main_figure_manifest(registry_path: Path) -> list[dict]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    figures = registry.get("main_figures", [])
    rows: list[dict] = []
    for figure in figures:
        base = Path(figure["base_path"])
        pngs = figure_style.rendered_panel_paths(base, ".png")
        pdfs = figure_style.rendered_panel_paths(base, ".pdf")
        row = {
            "figure_id": figure.get("id", base.name),
            "experiment": figure.get("experiment", experiment_label(base, Path("results"))),
            "base_path": str(base),
            "png_paths": [str(path) for path in pngs],
            "pdf_paths": [str(path) for path in pdfs],
            "role": figure.get("role", ""),
        }
        rows.append(row)
    return rows


def audit_main_figures(registry_path: Path, config: dict) -> list[dict]:
    rows: list[dict] = []
    for figure in load_main_figure_manifest(registry_path):
        png_paths = [Path(path) for path in figure["png_paths"]]
        pdf_paths = [Path(path) for path in figure["pdf_paths"]]
        if png_paths:
            pdf_lookup = {path.stem for path in pdf_paths}
            for png in png_paths:
                pdf_exists = png.stem in pdf_lookup
                row = figure_style.audit_rendered_png(png, config)
                row.update(
                    {
                        "figure_id": figure["figure_id"],
                        "experiment": figure["experiment"],
                        "role": figure["role"],
                        "pdf_exists": str(pdf_exists),
                    }
                )
                if not pdf_exists:
                    row["status"] = "WARN"
                    warnings = [item for item in str(row.get("warnings", "")).split(";") if item]
                    warnings.append("missing_pdf")
                    row["warnings"] = ";".join(warnings)
                rows.append(row)
        else:
            base = Path(figure["base_path"])
            row = {
                "figure_id": figure["figure_id"],
                "experiment": figure["experiment"],
                "role": figure["role"],
                "path": str(base.with_suffix(".png")),
                "status": "WARN",
                "warnings": "missing_png",
                "width_px": "",
                "height_px": "",
                "aspect_ratio": "",
                "content_fraction": "",
                "edge_ink_max": "",
                "edge_top": "",
                "edge_bottom": "",
                "edge_left": "",
                "edge_right": "",
                "pdf_exists": "False",
            }
            rows.append(row)
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_id",
        "experiment",
        "role",
        "path",
        "status",
        "warnings",
        "pdf_exists",
        "width_px",
        "height_px",
        "aspect_ratio",
        "content_fraction",
        "edge_ink_max",
        "edge_top",
        "edge_bottom",
        "edge_left",
        "edge_right",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], output: Path, config: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["status"] for row in rows)
    by_experiment: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_experiment[row["experiment"]][row["status"]] += 1

    lines = [
        "# Experiment Figure Style Audit",
        "",
        "This is a rendered-output screening report. `WARN` flags figures that deserve visual inspection; it is not a scientific validity failure.",
        "",
        "## Summary",
        "",
        f"- PNG figures audited: {len(rows)}",
        f"- PASS: {counts.get('PASS', 0)}",
        f"- WARN: {counts.get('WARN', 0)}",
        "",
        "## External Skill Sources Adopted",
        "",
        figure_style.external_skill_sources_markdown(config),
        "",
        "## Shared Figure Rules",
        "",
        figure_style.design_rules_markdown(config),
        "",
        "## Experiment-Level Status",
        "",
        "| experiment | PASS | WARN |",
        "| --- | ---: | ---: |",
    ]
    for experiment in sorted(by_experiment):
        counts_for_experiment = by_experiment[experiment]
        lines.append(
            f"| {experiment} | {counts_for_experiment.get('PASS', 0)} | {counts_for_experiment.get('WARN', 0)} |"
        )

    warn_rows = [row for row in rows if row["status"] == "WARN"]
    lines.extend(["", "## Warning Detail", ""])
    if not warn_rows:
        lines.append("- No warning-level PNG files detected by the automated screen.")
    else:
        lines.extend(
            [
                "| figure id | experiment | figure | warnings | pixels | aspect | edge ink |",
                "| --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in warn_rows:
            path = Path(row["path"])
            lines.append(
                "| "
                f"{row.get('figure_id', '')} | {row['experiment']} | {path.name} | {row['warnings']} | "
                f"{row['width_px']}x{row['height_px']} | {row['aspect_ratio']} | {row['edge_ink_max']} |"
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_contact_sheet(rows: list[dict], output: Path) -> None:
    """Write a thumbnail sheet for rapid visual comparison of audited figures."""
    output.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 520, 390
    label_h = 58
    pad = 18
    cols = 3
    rows_count = (len(rows) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows_count * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    for index, row in enumerate(rows):
        image_path = Path(row["path"])
        col, grid_row = index % cols, index // cols
        x = pad + col * (thumb_w + pad)
        y = pad + grid_row * (thumb_h + label_h + pad)
        draw.rectangle([x, y, x + thumb_w, y + thumb_h + label_h], outline=(220, 220, 220), width=1)
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_w - 20, thumb_h - 18), Image.Resampling.LANCZOS)
            ix = x + (thumb_w - image.width) // 2
            iy = y + 10 + (thumb_h - 18 - image.height) // 2
            canvas.paste(image, (ix, iy))
        draw.text((x + 10, y + thumb_h + 8), row.get("figure_id") or image_path.stem, fill=(38, 50, 56), font=font)
        draw.text((x + 10, y + thumb_h + 32), row.get("experiment", ""), fill=(78, 90, 94), font=small)

    canvas.save(output, dpi=(220, 220))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        action="append",
        default=["results"],
        help="Result root to scan. Can be provided multiple times.",
    )
    parser.add_argument(
        "--style-config",
        default="configs/figure_style.yaml",
        help="Shared figure-style YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figure_style_audit",
        help="Directory for audit CSV and Markdown reports.",
    )
    parser.add_argument(
        "--registry",
        default="configs/experiment_registry.yaml",
        help="Experiment registry containing the main_figures manifest.",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Audit only the final main figures listed in the experiment registry.",
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Write a PNG thumbnail sheet for quick visual review.",
    )
    args = parser.parse_args()

    config = {"plot_style_config": args.style_config}
    if args.main_only:
        rows = audit_main_figures(Path(args.registry), config)
    else:
        rows = audit_roots([Path(root) for root in args.results_root], config)
    output_dir = Path(args.output_dir)
    write_csv(rows, output_dir / "figure_style_audit.csv")
    write_markdown(rows, output_dir / "figure_style_audit.md", config)
    if args.contact_sheet:
        write_contact_sheet(rows, output_dir / "main_figure_contact_sheet.png")
    print(f"Audited {len(rows)} PNG figures.")
    print(f"Wrote {output_dir / 'figure_style_audit.csv'}")
    print(f"Wrote {output_dir / 'figure_style_audit.md'}")
    if args.contact_sheet:
        print(f"Wrote {output_dir / 'main_figure_contact_sheet.png'}")


if __name__ == "__main__":
    main()
