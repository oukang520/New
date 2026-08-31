"""Build manuscript reassembly indexes for standardized single-panel exports."""

from __future__ import annotations

import csv
import html
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = PROJECT_ROOT / "results" / "standardized_single_panels"
MANIFEST_PATH = PANEL_ROOT / "panel_manifest.tsv"


CORE_EXPERIMENTS = {
    "experiments_01_02": ("real_cross_sectional", 2),
    "experiment_04_relative_inflow": ("real_cross_sectional", 1),
    "experiment_06_bottleneck_recovery_enhanced": ("simulation", 1),
    "experiment_06_dwell_gradient": ("simulation", 1),
    "experiment_07_topology_robustness_balanced": ("simulation", 2),
    "experiment_08_biological_convergence": ("real_cross_sectional", 2),
    "experiment_09_observation_enrichment": ("simulation", 2),
    "experiment_14_ablation_backbone": ("method_control", 1),
    "experiment_15_uncertainty_negative_controls": ("method_control", 2),
    "experiment_16_real_topology": ("real_cross_sectional", 1),
    "experiment_17_longitudinal_public": ("real_longitudinal", 1),
}

SUPPORTING_EXPERIMENTS = {
    "experiment_03_mhn_interface": ("method_interface", 3),
    "experiment_05_state_scores": ("state_scoring", 3),
    "experiment_10_real_cohort_main": ("real_cross_sectional", 3),
    "experiment_11_information_gain": ("method_control", 3),
    "experiment_12_clinical_validation": ("clinical_association", 3),
    "experiment_13_cross_cohort_replication": ("replication", 3),
}


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def path_parts(path: str) -> list[str]:
    return Path(path.replace("\\", "/")).parts


def experiment_name(source_figure: str) -> str:
    parts = path_parts(source_figure)
    return parts[1] if len(parts) > 1 and parts[0] == "results" else parts[0]


def cohort_name(source_figure: str) -> str:
    for part in path_parts(source_figure):
        if part.startswith("AACR_"):
            return part.replace("AACR_", "")
    if "three_cohorts" in source_figure:
        return "LUAD_COAD_IDC"
    if "longitudinal" in source_figure:
        return "longitudinal_public"
    return "all"


def figure_name(source_figure: str) -> str:
    return Path(source_figure.replace("\\", "/")).stem


def output_role(row: dict[str, str]) -> str:
    exp = experiment_name(row["source_figure"])
    label = row["panel_label"]
    src = row["source_figure"]
    if label == "TABLE" or Path(src.replace("\\", "/")).name.startswith("Table_"):
        return "table_candidate"
    if "\\AACR_" in src or "/AACR_" in src:
        return "supplementary_detail"
    if exp in CORE_EXPERIMENTS:
        return "main_text_candidate"
    return "supplementary_candidate"


def story_block(exp: str) -> str:
    if exp in CORE_EXPERIMENTS:
        return CORE_EXPERIMENTS[exp][0]
    if exp in SUPPORTING_EXPERIMENTS:
        return SUPPORTING_EXPERIMENTS[exp][0]
    return "other"


def priority(exp: str) -> int:
    if exp in CORE_EXPERIMENTS:
        return CORE_EXPERIMENTS[exp][1]
    if exp in SUPPORTING_EXPERIMENTS:
        return SUPPORTING_EXPERIMENTS[exp][1]
    return 4


def assembly_slot(row: dict[str, str]) -> str:
    cls = row["standard_class"]
    if cls in {"large_wide", "large_square"}:
        return "full_width_or_large_panel"
    if cls in {"wide_main", "table_wide"}:
        return "full_width_or_two_column"
    if cls == "square_main":
        return "two_or_three_panel_row"
    if cls == "square_subpanel":
        return "small_multiple"
    return "custom"


def enrich(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        exp = experiment_name(row["source_figure"])
        item = {
            "role": output_role(row),
            "story_block": story_block(exp),
            "priority": str(priority(exp)),
            "experiment": exp,
            "figure": figure_name(row["source_figure"]),
            "cohort": cohort_name(row["source_figure"]),
            "panel_label": row["panel_label"],
            "standard_class": row["standard_class"],
            "assembly_slot": assembly_slot(row),
            "minimum_recommended_final_width_mm": row[
                "minimum_recommended_final_width_mm"
            ],
            "standard_width_mm": row["standard_width_mm"],
            "standard_height_mm": row["standard_height_mm"],
            "panel_id": row["panel_id"],
            "standard_png": row["standard_png"],
            "standard_pdf": row["standard_pdf"],
            "native_png": row["native_png"],
            "source_figure": row["source_figure"],
        }
        enriched.append(item)
    return sorted(
        enriched,
        key=lambda r: (
            int(r["priority"]),
            r["story_block"],
            r["experiment"],
            r["figure"],
            r["cohort"],
            r["panel_label"],
        ),
    )


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: str) -> str:
    return Path(path).as_posix()


def write_html(path: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        png = rel(row["standard_png"])
        title = f'{row["experiment"]} | {row["figure"]} | {row["panel_label"]}'
        meta = (
            f'{row["role"]}; {row["story_block"]}; '
            f'{row["standard_class"]}; min {row["minimum_recommended_final_width_mm"]} mm'
        )
        cards.append(
            "<article>"
            f'<a href="{html.escape(png)}"><img src="{html.escape(png)}" '
            f'alt="{html.escape(title)}"></a>'
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(meta)}</p>"
            "</article>"
        )
    body = "\n".join(cards)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Standardized Panel Gallery</title>
<style>
body {{
  margin: 28px;
  font-family: Arial, sans-serif;
  color: #263238;
  background: white;
}}
h1 {{
  font-size: 22px;
  margin: 0 0 8px;
}}
.note {{
  font-size: 13px;
  color: #607078;
  margin-bottom: 22px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 18px;
}}
article {{
  border-top: 1px solid #cfd8dc;
  padding-top: 10px;
}}
img {{
  width: 100%;
  height: 180px;
  object-fit: contain;
  background: #fff;
}}
h2 {{
  font-size: 11px;
  line-height: 1.3;
  margin: 8px 0 4px;
  word-break: break-word;
}}
p {{
  font-size: 10px;
  line-height: 1.35;
  color: #607078;
  margin: 0;
}}
</style>
</head>
<body>
<h1>Standardized Panel Gallery</h1>
<div class="note">Use the minimum final width in the index tables when regrouping panels.</div>
<main class="grid">
{body}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_blueprint(path: Path, rows: list[dict[str, str]]) -> None:
    role_counts: dict[str, int] = {}
    block_counts: dict[str, int] = {}
    for row in rows:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
        block_counts[row["story_block"]] = block_counts.get(row["story_block"], 0) + 1

    lines = [
        "# 后期组图索引说明",
        "",
        "本文件由 `src/build_panel_reassembly_index.py` 生成，用于从标准化单图中快速挑选主文和补充材料面板。",
        "",
        "## 面板分组",
        "",
    ]
    for role, count in sorted(role_counts.items()):
        lines.append(f"- `{role}`：{count} 张。")

    lines.extend(["", "## 证据链模块", ""])
    for block, count in sorted(block_counts.items()):
        lines.append(f"- `{block}`：{count} 张。")

    lines.extend(
        [
            "",
            "## 推荐主文链条",
            "",
            "- 真实横断面队列：优先选 E1/E4/E8/E16 中与 R*、相对停留时间、真实拓扑直接相关的面板。",
            "- 模拟队列：优先选 E6 增强版、E6 连续停留梯度、E7 鲁棒性、E9 观察富集和 E14 消融中直接验证创新点的面板。",
            "- 真实纵向队列：优先选 E17 的核心指标表、R* 与停留代理相关性、真实纵向路线面板。",
            "- 低优先级或重复展示的 per-cohort 细节图建议放入补充材料。",
            "",
            "## 使用方式",
            "",
            "- `reassembly_index.tsv`：全部 133 张面板的完整索引。",
            "- `main_text_panel_index.tsv`：主文候选面板。",
            "- `supplementary_panel_index.tsv`：补充候选面板。",
            "- `table_panel_index.tsv`：表格类面板。",
            "- `panel_gallery.html`：缩略图浏览页，可直接用浏览器打开。",
            "",
            "组图时请遵守 `minimum_recommended_final_width_mm`。如果需要把某张图压得更小，应回到原实验绘图脚本重新渲染更大的字体，而不是继续缩小裁图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = enrich(read_manifest())
    main_rows = [r for r in rows if r["role"] == "main_text_candidate"]
    supp_rows = [r for r in rows if r["role"].startswith("supplementary")]
    table_rows = [r for r in rows if r["role"] == "table_candidate"]

    write_tsv(PANEL_ROOT / "reassembly_index.tsv", rows)
    write_tsv(PANEL_ROOT / "main_text_panel_index.tsv", main_rows)
    write_tsv(PANEL_ROOT / "supplementary_panel_index.tsv", supp_rows)
    write_tsv(PANEL_ROOT / "table_panel_index.tsv", table_rows)
    write_html(PANEL_ROOT / "panel_gallery.html", rows)
    write_blueprint(PANEL_ROOT / "reassembly_blueprint.md", rows)

    print(f"All panels: {len(rows)}")
    print(f"Main-text candidates: {len(main_rows)}")
    print(f"Supplementary candidates: {len(supp_rows)}")
    print(f"Table candidates: {len(table_rows)}")


if __name__ == "__main__":
    main()
