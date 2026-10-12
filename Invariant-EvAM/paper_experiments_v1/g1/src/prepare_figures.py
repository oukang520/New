from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
G1 = ROOT / "paper_experiments_v1/g1"
RESULTS = G1 / "results"
FD = G1 / "figure_data"
PALETTE = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_panel(fid, data, message, estimand, unit, ndef, primary, secondary, reason, caption):
    d = FD / fid; d.mkdir(parents=True, exist_ok=True)
    path = d / "data.tsv"; data.to_csv(path, sep="\t", index=False)
    wrapper = f'''from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parents[4]\nsys.path.insert(0,str(ROOT/"paper_experiments_v1/g1/src"))\nfrom g1_plot import plot_panel\nplot_panel("{fid}", Path(__file__).resolve().parent)\n'''
    (d / "plot.py").write_text(wrapper)
    (d / "caption_notes.md").write_text(caption.strip() + "\n")
    manifest = {
        "figure_id": fid, "panel_id": fid, "scientific_message": message,
        "data_source": str(path.relative_to(ROOT)), "data_checksum": sha(path),
        "estimand": estimand, "unit_of_analysis": unit, "n_definition": ndef,
        "summary_statistic": "defined in data.tsv and caption_notes.md",
        "interval_definition": "Q1-Q3 or Wilson 95% interval where shown",
        "scale_and_transform": "linear except G1A log10 y-axis",
        "statistical_annotation": "descriptive; no P values",
        "primary_reference_id": primary, "secondary_reference_id": secondary,
        "reference_selection_reason": reason, "palette_family": "user_frozen_G1",
        "palette_mapping": {f"color_{i+1}": c for i,c in enumerate(PALETTE)},
        "shape_and_linetype_mapping": {"models": "marker shape where applicable"},
        "missing_failure_boundary_handling": "all eligible denominators retained; missing is not zero",
        "adaptations_from_reference": ["single standalone axes", "labels adapted to G1 estimands"],
        "code_path": str((d / "plot.py").relative_to(ROOT)),
        "plot_data_path": str(path.relative_to(ROOT)),
        "output_paths": [str((d / f"{fid}.pdf").relative_to(ROOT)), str((d / f"{fid}.svg").relative_to(ROOT))],
        "final_width_mm": 89, "final_height_mm": 66, "vector_text_preserved": True,
        "review_checks": {"statistical_semantics": False, "reference_match": False,
            "color_and_shape_redundancy": False, "final_size_legibility": False, "caption_complete": False},
        "review_status": "not_reviewed", "dpi_setting": 600,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    alias = pd.read_csv(RESULTS / "G1_alias_summary.tsv", sep="\t")
    a = alias[["model_id", "parameter_draw_id", "parameter_distance", "target_change",
               "max_abs_probability_error", "max_observed_error_K1"]].copy()
    write_panel("G1A", a,
        "完整参数化中，目标效应改变 0.4 仍可找到相同或近似相同源分布的兼容参数。",
        "固定目标坐标变化下的最大源概率误差", "独立确认参数 draw", "每个完整模型 10 个预设 draw",
        "R06", "R25", "R06 的计算点和误差尺度适合兼容路径诊断；R25 提供模型间配对比较语法。",
        "# G1A\n\n每点为一个独立确认参数 draw。横轴为参数向量距离；纵轴为固定目标 interaction 改变 0.4 后的最大源概率误差（log 尺度）。D3 为 10/10，D4 为 10/10 次优化完成；D4 中较大误差只算近似 alias。无显著性检验。")

    v = pd.read_csv(RESULTS / "G1A_population_geometry_v1_1.tsv", sep="\t")
    b = v[(v.operator_knowledge == "K1_STRUCTURED") &
          (v.budget_basis == "fixed_measurement_proxy") &
          (v.model_id.isin(["D3_SPARSE5", "D4_SPARSE10"])) &
          (((v.scenario == "single") & (v.environments == 1)) |
           ((v.scenario != "single") & (v.environments == 2)))]
    b = b.groupby(["model_id", "scenario"], as_index=False).agg(
        median_information=("primary_target_information_total", "median"),
        q1_information=("primary_target_information_total", lambda x: x.quantile(.25)),
        q3_information=("primary_target_information_total", lambda x: x.quantile(.75)),
        identified_fraction=("primary_target_identified", "mean"), n_draws=("parameter_draw_id", "size"))
    write_panel("G1B", b,
        "固定事件测量 proxy 预算下，异质环境的收益取决于模型结构；纯 cross-view failure 不提供目标信息。",
        "总目标有效信息（固定 1000×d 事件测量）", "模型×设计场景的 60 个确认 draw",
        "每个模型和场景 n=60；single E=1，其余 E=2", "R25", "R23",
        "R25 表达加入/更换环境后的设计差异，R23 的点和区间语法表达参数 draw 的离散性。",
        "# G1B\n\n点为 60 个 v1.1 独立确认参数 draw 的中位数，线为 Q1–Q3。预算固定为 1000×d 个患者-事件测量 proxy；single 使用 E=1，其余设计 E=2。颜色区分稀疏模型，形状冗余。")

    raw = pd.read_csv(RESULTS / "G1_population_geometry.tsv", sep="\t")
    c = raw[(raw.operator_knowledge == "K1_STRUCTURED") & raw.model_id.isin(["D3_SPARSE5", "D4_SPARSE10"])]
    c = c.groupby(["model_id", "design_family", "design_id"], as_index=False).agg(
        target_rescue_fraction=("primary_target_rescued", "mean"),
        rank_gain_fraction=("rank_gain", lambda x: np.mean(np.asarray(x) > 0)),
        target_identified_fraction=("primary_target_identified", "mean"), n_draws=("parameter_draw_id", "size"))
    write_panel("G1C", c,
        "非真值定制设计库的互补成功率高度异质，完整分母中同时存在成功与失败设计。",
        "每个预冻结设计跨 60 个确认 draw 的目标 rescue 比例", "预冻结设计", "2 模型×24 设计；每点 60 draws",
        "R27", "R20", "R27 的原始分布点保留所有设计，避免只画成功候选；R20 辅助类别分组。",
        "# G1C\n\n每点是一个预冻结设计在 60 个确认参数 draw 上的目标 rescue 比例；共 48 点，无删除。横线为类别中位数。这里不把 v1.0 的 nominal failure 当作纯 assay failure；其选择路径审计另表保存。")

    d = raw[(raw.model_id.isin(["D3_SPARSE5", "D4_SPARSE10"]))]
    d = d.groupby(["model_id", "operator_knowledge", "design_family"], as_index=False).agg(
        target_identified_fraction=("primary_target_identified", "mean"),
        target_rescue_fraction=("primary_target_rescued", "mean"),
        n_evaluations=("parameter_draw_id", "size"))
    write_panel("G1D", d,
        "assay 与选择变化在 K0/K1 下贡献不同；未知选择 nuisance 可显著改变可识别比例。",
        "目标 interaction 可识别比例", "模型×知识条件×设计机制类别", "每格 360 个设计-draw 评价",
        "R28", "R24", "R24 的气泡矩阵适合双编码机制类别与比例；R28 保留互斥设计类别分母。",
        "# G1D\n\n颜色和圆面积均表示目标可识别比例；每格 6 个预冻结设计×60 draws=360 次评价。列同时给出模型和 operator 知识条件。K0 已知权重，K1 为特征族已知但参数未知。")

    cap = pd.read_csv(RESULTS / "G1_source_information_cap.tsv", sep="\t")
    e = cap.groupby("model_id", as_index=False).agg(parameter_count=("parameter_count", "first"),
        source_rank=("source_rank", "median"), source_nullity=("source_nullity", "median"),
        primary_identified_fraction=("primary_target_identified_from_source", "mean"),
        n_draws=("parameter_draw_id", "size"))
    write_panel("G1E", e,
        "完整拓扑受源横断面概率维数上限约束；稀疏拓扑的识别结论依赖排除结构。",
        "源分布 Jacobian 的参数数、秩与零空间维数", "模型（跨 60 个确认 draw 汇总）", "4 模型；每模型 n=60 draws",
        "R20", "R23", "R20 的分块热图直接显示秩与零空间；R23 提供理论上限关系的表达语法。",
        "# G1E\n\n格内为 60 个确认参数 draw 的中位数（参数数为固定值）。完整 D3 为 9 参数/7 秩，完整 D4 为 16/15；其主目标从源横断面分布识别比例均为 0。稀疏模型为满秩且主目标比例为 1。")


if __name__ == "__main__":
    main()
