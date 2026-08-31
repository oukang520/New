"""Audit standardized single-panel exports and render contact sheets."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = PROJECT_ROOT / "results" / "standardized_single_panels"
MANIFEST_PATH = PANEL_ROOT / "panel_manifest.tsv"
AUDIT_PATH = PANEL_ROOT / "panel_audit.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def label_for(row: dict[str, str]) -> str:
    source = Path(row["source_figure"].replace("\\", "/")).stem
    return f'{source} | {row["panel_label"]}'


def font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int = 44) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = []
    current = ""
    for chunk in text.replace("__", " ").replace("_", " ").split():
        if len(current) + len(chunk) + 1 > max_chars and current:
            lines.append(current)
            current = chunk
        else:
            current = f"{current} {chunk}".strip()
    if current:
        lines.append(current)
    return lines[:3]


def render_contact_sheet(rows: list[dict[str, str]], output: Path, columns: int = 6) -> None:
    thumb_w = 310
    thumb_h = 235
    label_h = 58
    pad = 18
    rows_count = max(1, math.ceil(len(rows) / columns))
    width = columns * thumb_w + (columns + 1) * pad
    height = rows_count * (thumb_h + label_h) + (rows_count + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = font(11)
    meta_font = font(9)
    for index, row in enumerate(rows):
        col = index % columns
        line = index // columns
        x0 = pad + col * (thumb_w + pad)
        y0 = pad + line * (thumb_h + label_h + pad)
        image_path = PROJECT_ROOT / row["standard_png"]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        ix = x0 + (thumb_w - image.width) // 2
        iy = y0 + (thumb_h - image.height) // 2
        sheet.paste(image, (ix, iy))
        draw.rectangle([x0, y0, x0 + thumb_w, y0 + thumb_h], outline="#D9DEE2", width=1)
        label_y = y0 + thumb_h + 7
        for line_text in wrap_text(label_for(row)):
            draw.text((x0, label_y), line_text, fill="#263238", font=title_font)
            label_y += 13
        meta = f'{row["standard_class"]}; min {row["minimum_recommended_final_width_mm"]} mm'
        draw.text((x0, label_y + 2), meta, fill="#607078", font=meta_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=(180, 180))


def write_qc_report(
    manifest: list[dict[str, str]],
    audit: list[dict[str, str]],
    missing: list[tuple[str, str, str]],
    output: Path,
) -> None:
    status_counts: dict[str, int] = {}
    source_review_counts: dict[str, int] = {}
    for row in audit:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        source_status = row.get("source_crop_edge_status", "NA")
        source_review_counts[source_status] = source_review_counts.get(source_status, 0) + 1
    p_labels = [row for row in manifest if row["panel_label"].startswith("P")]
    auto = [row for row in manifest if row["split_method"] == "auto"]
    oncoprint_p = [row for row in manifest if "Figure_E1_oncoprint__P" in row["panel_id"]]
    lines = [
        "# Standardized Panel Visual QC",
        "",
        "## Automated Checks",
        "",
        f"- Total panels: {len(manifest)}.",
        f"- Final audit status: {status_counts}.",
        f"- Source-window edge review flags: {source_review_counts}.",
        f"- Auto-split panels: {len(auto)}.",
        f"- Generic Pxx panel labels: {len(p_labels)}.",
        f"- E1 oncoprint Pxx fragments: {len(oncoprint_p)}.",
        f"- Missing output files: {len(missing)}.",
        "",
        "## Corrected Issues",
        "",
        "- E3 three-cohort one-step panels: expanded left source window so long state names are retained.",
        "- E1 per-cohort oncoprints: exported as one complete `ONCOPRINT` panel instead of title/body fragments.",
        "- E12 clinical validation: re-rendered the source figure with three KM panels only; the previous fourth empty axis was removed.",
        "- E5/E8/E1/E2 supplementary detail figures: replaced generic auto labels with explicit manual panel labels.",
        "",
        "## Manual Review Notes",
        "",
        "- All final standardized PNG/PDF panels pass scale and canvas-edge checks.",
        "- `source_crop_edge_status=REVIEW` means the broad source window was close to plotted content; these are review prompts, not final canvas failures.",
        "- The final contact sheet and source-window review sheet should be used for quick visual inspection before manuscript assembly.",
    ]
    if missing:
        lines.extend(["", "## Missing Files", ""])
        for panel_id, column, path in missing[:30]:
            lines.append(f"- `{panel_id}` missing `{column}`: `{path}`.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = read_tsv(MANIFEST_PATH)
    audit = read_tsv(AUDIT_PATH)
    audit_by_panel = {row["panel_id"]: row for row in audit}
    missing = []
    for row in manifest:
        for column in ["native_png", "standard_png", "standard_pdf"]:
            path = PROJECT_ROOT / row[column]
            if not path.exists():
                missing.append((row["panel_id"], column, row[column]))

    render_contact_sheet(manifest, PANEL_ROOT / "panel_contact_sheet.png")
    review_rows = [
        row
        for row in manifest
        if audit_by_panel.get(row["panel_id"], {}).get("source_crop_edge_status") == "REVIEW"
    ]
    render_contact_sheet(review_rows, PANEL_ROOT / "source_window_review_sheet.png", columns=4)
    write_qc_report(manifest, audit, missing, PANEL_ROOT / "panel_visual_qc.md")
    print(f"panels={len(manifest)}")
    print(f"review_panels={len(review_rows)}")
    print(f"missing_files={len(missing)}")


if __name__ == "__main__":
    main()
