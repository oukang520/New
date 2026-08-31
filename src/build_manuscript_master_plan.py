from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def read_text(path: str, default: str = "") -> str:
    p = ROOT / path
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8", errors="replace")


def read_yaml(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def read_json(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def read_table(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    return pd.read_csv(p, sep=sep)


def fmt(x, digits: int = 3) -> str:
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass
    if isinstance(x, str):
        return x
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf) >= 1000:
        return f"{xf:,.0f}"
    if abs(xf) < 0.001 and xf != 0:
        return f"{xf:.2e}"
    return f"{xf:.{digits}f}"


def md_escape(value) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def md_table(rows: Iterable[dict], columns: list[str]) -> str:
    rows = list(rows)
    if not rows:
        return "_无可用记录。_"
    out = ["| " + " | ".join(columns) + " |"]
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        out.append("| " + " | ".join(md_escape(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def one_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def path_ref(path: str) -> str:
    return f"`{path}`" if exists(path) else f"`{path}`【待确认】"


def get_aacr_version() -> str:
    meta = read_text("Data/AACR/AACR/meta_study.txt")
    name = re.search(r"^name:\s*(.+)$", meta, re.M)
    desc = re.search(r"^description:\s*(.+)$", meta, re.M)
    if name or desc:
        return f"{name.group(1) if name else ''}; {desc.group(1) if desc else ''}".strip("; ")
    return "【待确认】"


def selected_dataset_rows() -> list[dict]:
    selected = read_yaml("configs/selected_experiment_datasets.yaml")
    e12 = read_table("results/experiments_01_02/experiments_01_02_summary.csv")
    rows = []
    for item in selected.get("included_datasets", []):
        name = item["dataset_name"]
        manifest = read_json(f"processed/experiment_ready/{name}/dataset_manifest.json")
        counts = manifest.get("counts", {})
        checks = manifest.get("checks", {})
        e12_row = e12[e12["dataset_name"].eq(name)].iloc[0].to_dict() if not e12.empty and name in set(e12["dataset_name"]) else {}
        rows.append(
            {
                "dataset": name,
                "role": item.get("role", ""),
                "analysis_units": fmt(counts.get("analysis_units"), 0),
                "patients": fmt(counts.get("patients"), 0),
                "mutation_rows": fmt(counts.get("mutation_rows"), 0),
                "events_retained": fmt(counts.get("events_retained"), 0),
                "valid_states_manifest": fmt(counts.get("valid_state_count"), 0),
                "valid_states_E1": fmt(e12_row.get("valid_states"), 0),
                "zero_event_fraction": fmt(e12_row.get("zero_event_fraction")),
                "manifest_checks": "all True" if checks and all(checks.values()) else "【待确认】",
                "warnings": "; ".join(manifest.get("warnings", [])) or "none",
            }
        )
    return rows


def excluded_dataset_rows() -> list[dict]:
    selected = read_yaml("configs/selected_experiment_datasets.yaml")
    return [
        {"dataset": x.get("dataset_name", ""), "reason": x.get("reason", "")}
        for x in selected.get("excluded_datasets", [])
    ]


def experiment_registry_rows() -> list[dict]:
    reg = read_yaml("configs/experiment_registry.yaml")
    experiments = reg.get("experiments", [])
    rows = []
    if isinstance(experiments, dict):
        iterator = [{"id": key, **(value or {})} for key, value in experiments.items()]
    else:
        iterator = experiments
    for e in iterator:
        if isinstance(e, str):
            e = {"id": e}
        rows.append(
            {
                "id": e.get("id", ""),
                "name": e.get("name", e.get("role", "")),
                "status": e.get("status", ""),
                "result_root": e.get("result_root", ""),
                "primary": "yes" if e.get("primary", True) else "no",
            }
        )
    if exists("results/experiment_17_longitudinal_public"):
        rows.append(
            {
                "id": "experiment_17",
                "name": "public longitudinal validation of R* persistence predictions",
                "status": "present in results; not listed in current registry",
                "result_root": "results/experiment_17_longitudinal_public",
                "primary": "yes",
            }
        )
    return rows


def figure_inventory_rows() -> list[dict]:
    rows = []
    for exp_dir in sorted((ROOT / "results").glob("experiment*")) + sorted((ROOT / "results").glob("experiments_*")):
        if not exp_dir.is_dir():
            continue
        pngs = list(exp_dir.rglob("*.png"))
        pdfs = list(exp_dir.rglob("*.pdf"))
        single = [p for p in pngs if "single_figures" in p.parts]
        figures = [p for p in pngs if "figures" in p.parts]
        rows.append(
            {
                "result_root": rel(exp_dir),
                "png_total": len(pngs),
                "pdf_total": len(pdfs),
                "single_png": len(single),
                "figure_png": len(figures),
            }
        )
    return rows


def core_result_rows() -> list[dict]:
    rows: list[dict] = []

    e3 = read_table("results/experiment_03_mhn_interface/experiment_03_summary.csv")
    for _, r in e3.iterrows():
        rows.append(
            {
                "experiment": "E3",
                "dataset": r["dataset_name"],
                "endpoint": "MHN fitting and one-step interface",
                "metric": "events / observed genotypes / state edges",
                "result": f"{fmt(r['events'],0)} / {fmt(r['observed_genotypes'],0)} / {fmt(r['state_transition_edges'],0)}",
                "comparison": f"fit_status={fmt(r['fit_status'],0)}, max probability-sum error={fmt(r['max_probability_sum_error'],2)}",
                "evidence": "results/experiment_03_mhn_interface/experiment_03_summary.csv",
                "interpretation": "MHN transition probabilities can be converted into same-stage one-step state inflow.",
            }
        )

    e4 = read_table("results/experiment_04_relative_inflow/experiment_04_summary.csv")
    for _, r in e4.iterrows():
        rows.append(
            {
                "experiment": "E4",
                "dataset": r["dataset_name"],
                "endpoint": "relative inflow feasibility",
                "metric": "positive inflow states / stable sample fraction / rho(L,F)",
                "result": f"{fmt(r['positive_inflow_states'],0)} / {fmt(r['stable_sample_fraction'])} / {fmt(r['spearman_L_vs_F'])}",
                "comparison": "same-stage one-step inflow rule",
                "evidence": "results/experiment_04_relative_inflow/experiment_04_summary.csv",
                "interpretation": "F_hat is defined for most observed states; L and F are related but not identical.",
            }
        )

    e5 = read_table("results/experiment_05_state_scores/experiment_05_summary.csv")
    for _, r in e5.iterrows():
        rows.append(
            {
                "experiment": "E5",
                "dataset": r["dataset_name"],
                "endpoint": "state-level R* and O* output",
                "metric": "eligible / high-confidence states / top high-confidence stability",
                "result": f"{fmt(r['states_eligible'],0)} / {fmt(r['states_high_confidence'],0)} / {fmt(r['top_high_confidence_stability'])}",
                "comparison": "bootstrap replicates=200",
                "evidence": "results/experiment_05_state_scores/experiment_05_summary.csv",
                "interpretation": "The real cross-sectional cohorts yield a sizable set of evaluable R* states.",
            }
        )

    e6 = read_table("results/experiment_06_bottleneck_recovery_enhanced/tables/performance_summary_table.tsv")
    for _, r in e6.iterrows():
        endpoint = str(r["endpoint"])
        rows.append(
            {
                "experiment": "E6",
                "dataset": "simulated positive control",
                "endpoint": endpoint,
                "metric": "R* vs occupancy",
                "result": f"R* {r.get('R_star_median_iqr', fmt(r.get('R_star_median')))}; occupancy {r.get('occupancy_median_iqr', fmt(r.get('occupancy_median')))}",
                "comparison": f"paired median delta={fmt(r.get('paired_delta_median'))}; p={fmt(r.get('paired_p_value'),2)}",
                "evidence": "results/experiment_06_bottleneck_recovery_enhanced/tables/performance_summary_table.tsv",
                "interpretation": "Positive-control bottleneck recovery strongly supports the R* estimator, but ceiling metrics should be reported with distribution.",
            }
        )

    e6g = read_table("results/experiment_06_dwell_gradient/tables/performance_summary.tsv")
    for _, r in e6g.iterrows():
        rows.append(
            {
                "experiment": "E6-gradient",
                "dataset": "continuous simulated dwell gradient",
                "endpoint": r["endpoint"],
                "metric": "R* vs occupancy",
                "result": f"R* {fmt(r['R_star_median'])} [{fmt(r['R_star_q1'])}, {fmt(r['R_star_q3'])}]; occupancy {fmt(r['occupancy_median'])}",
                "comparison": f"favorable delta={fmt(r['favorable_delta_median'])}; p={fmt(r['wilcoxon_p'],2)}",
                "evidence": "results/experiment_06_dwell_gradient/tables/performance_summary.tsv",
                "interpretation": "This directly addresses whether R* recovers a graded relative dwell signal, not merely an easy binary bottleneck.",
            }
        )

    e7 = one_row(read_table("results/experiment_07_topology_robustness_balanced/tables/experiment_07_global_summary.tsv"))
    if e7:
        rows.append(
            {
                "experiment": "E7",
                "dataset": "balanced topology simulation grid",
                "endpoint": "topology robustness",
                "metric": "global Spearman(D_true, score)",
                "result": f"R* {fmt(e7.get('global_spearman_R_star_median'))} [{fmt(e7.get('global_spearman_R_star_q1'))}, {fmt(e7.get('global_spearman_R_star_q3'))}]",
                "comparison": f"occupancy {fmt(e7.get('global_spearman_occupancy_median'))}; conditions={fmt(e7.get('conditions'),0)}, fits={fmt(e7.get('total_fits'),0)}",
                "evidence": "results/experiment_07_topology_robustness_balanced/tables/experiment_07_global_summary.tsv",
                "interpretation": "R* remains directionally better under topology/sparsity stress, but performance is condition-dependent.",
            }
        )

    e8 = read_table("results/experiment_08_biological_convergence/tables/cohort_summary.tsv")
    for _, r in e8.iterrows():
        rows.append(
            {
                "experiment": "E8",
                "dataset": r["dataset_name"],
                "endpoint": "biological convergence",
                "metric": "top expected-module fraction / CI>1 fraction / median top R*",
                "result": f"{fmt(r['top_expected_module_fraction'])} / {fmt(r['top_ci_above_one_fraction'])} / {fmt(r['median_top_R_star'])}",
                "comparison": f"module enrichment p={fmt(r['expected_module_p_value'])}; rho(R*,N)={fmt(r['spearman_R_star_vs_N'])}",
                "evidence": "results/experiment_08_biological_convergence/tables/cohort_summary.tsv",
                "interpretation": "Top R* states are biologically plausible, but module p-values are not strong enrichment evidence.",
            }
        )

    e9 = read_table("results/experiment_09_observation_enrichment/tables/experiment_09_summary.tsv")
    for _, r in e9.iterrows():
        rows.append(
            {
                "experiment": "E9",
                "dataset": f"simulated O* scenario: {r['scenario']}",
                "endpoint": "observation enrichment recovery",
                "metric": "Spearman(O*, omega) / high-omega AUC",
                "result": f"{fmt(r['spearman_O_star'])} / {fmt(r['high_omega_auc_O_star'])}",
                "comparison": f"occupancy Spearman={fmt(r['spearman_occupancy'])}, AUC={fmt(r['high_omega_auc_occupancy'])}",
                "evidence": "results/experiment_09_observation_enrichment/tables/experiment_09_summary.tsv",
                "interpretation": "O* is validated as auxiliary observation-enrichment residual, not as the main dwell estimator.",
            }
        )

    e10 = read_table("results/experiment_10_real_cohort_main/tables/cohort_main_summary.tsv")
    for _, r in e10.iterrows():
        rows.append(
            {
                "experiment": "E10",
                "dataset": r["dataset_name"],
                "endpoint": "real-cohort main synthesis",
                "metric": "top R* median / top O* median / states R*>1",
                "result": f"{fmt(r['top_R_star_median'])} / {fmt(r['top_O_star_median'])} / {fmt(r['states_R_gt_1'],0)}",
                "comparison": f"eligible={fmt(r['eligible_states'],0)}, high-confidence={fmt(r['high_confidence_states'],0)}",
                "evidence": "results/experiment_10_real_cohort_main/tables/cohort_main_summary.tsv",
                "interpretation": "Real cohorts contain high-R* states after progression-inflow normalization.",
            }
        )

    e11 = read_table("results/experiment_11_information_gain/tables/information_gain_summary.tsv")
    for _, r in e11.iterrows():
        rows.append(
            {
                "experiment": "E11",
                "dataset": r["dataset_name"],
                "endpoint": "information gain over MHN-only and occupancy-only",
                "metric": "top overlap R*-MHN / R*-occupancy; rho(L,F); rho(R*,F)",
                "result": f"{fmt(r['top_R_and_MHN_count'],0)}/{fmt(r['top_R_states'],0)}; {fmt(r['top_R_and_occupancy_count'],0)}/{fmt(r['top_R_states'],0)}; {fmt(r['spearman_occupancy_MHN'])}; {fmt(r['spearman_R_MHN'])}",
                "comparison": f"median rank gain vs MHN={fmt(r['median_rank_gain_vs_MHN'])}",
                "evidence": "results/experiment_11_information_gain/tables/information_gain_summary.tsv",
                "interpretation": "R* is not reducible to MHN inflow or raw state prevalence.",
            }
        )

    e12 = read_table("results/experiment_12_clinical_validation/tables/stage_subgroup_cox.tsv")
    for _, r in e12.iterrows():
        rows.append(
            {
                "experiment": "E12",
                "dataset": r["dataset_name"],
                "endpoint": f"clinical association: {r['subgroup']}",
                "metric": "Cox HR per SD log R*",
                "result": f"HR={fmt(r['hr_per_sd'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}], p={fmt(r['p_value'],2)}",
                "comparison": f"n={fmt(r['n'],0)}, events={fmt(r['events'],0)}",
                "evidence": "results/experiment_12_clinical_validation/tables/stage_subgroup_cox.tsv",
                "interpretation": "Clinical association is supportive and context-dependent; it is not a direct dwell-time truth label.",
            }
        )

    e13 = read_table("results/experiment_13_cross_cohort_replication/tables/experiment_13_summary.tsv")
    for _, r in e13.iterrows():
        rows.append(
            {
                "experiment": "E13",
                "dataset": r["dataset_name"],
                "endpoint": "split-cohort replication",
                "metric": "median split Spearman / top10 overlap enrichment",
                "result": f"{fmt(r['median_spearman_rho'])} / {fmt(r['median_top10_enrichment'])}x",
                "comparison": f"repeats={fmt(r['repeats'],0)}, fraction top10 above null={fmt(r['fraction_top10_above_null_p05'])}",
                "evidence": "results/experiment_13_cross_cohort_replication/tables/experiment_13_summary.tsv",
                "interpretation": "State rankings are internally stable across patient splits under the fixed backbone.",
            }
        )

    e14 = read_table("results/experiment_14_ablation_backbone/tables/backbone_top_state_retention.tsv")
    if not e14.empty:
        for _, r in e14.head(12).iterrows():
            rows.append(
                {
                    "experiment": "E14",
                    "dataset": r.get("dataset_name", ""),
                    "endpoint": f"denominator ablation: {r.get('variant_display', r.get('variant',''))}",
                    "metric": "top-state retention/Jaccard",
                    "result": f"retained={fmt(r.get('retained_top_k'),0)}/{fmt(r.get('top_k'),0)}, Jaccard={fmt(r.get('jaccard'))}",
                    "comparison": "compared with full R* top states",
                    "evidence": "results/experiment_14_ablation_backbone/tables/backbone_top_state_retention.tsv",
                    "interpretation": "Changing the denominator changes the discovered high-R* states, supporting denominator specificity.",
                }
            )
    else:
        e14s = read_table("results/experiment_14_ablation_backbone/tables/experiment_14_summary.tsv")
        for _, r in e14s.iterrows():
            rows.append(
                {
                    "experiment": "E14",
                    "dataset": r["dataset_name"],
                    "endpoint": f"denominator ablation: {r['variant_display']}",
                    "metric": "clinical C-index / top overlap",
                    "result": f"C-index={fmt(r['c_index'])}; median overlap={fmt(r['median_top_overlap'])}",
                    "comparison": f"delta C-index vs full={fmt(r['delta_c_index_vs_full'])}",
                    "evidence": "results/experiment_14_ablation_backbone/tables/experiment_14_summary.tsv",
                    "interpretation": "Ablations quantify how much conclusions depend on the MHN-derived denominator.",
                }
            )

    e15a = read_table("results/experiment_15_uncertainty_negative_controls/tables/matched_decoy_summary.tsv")
    for _, r in e15a.iterrows():
        rows.append(
            {
                "experiment": "E15A",
                "dataset": r["dataset_name"],
                "endpoint": "matched decoy contrast",
                "metric": "fraction above decoy q90 / log2 R advantage",
                "result": f"{fmt(r['fraction_above_decoy_q90'])} / {fmt(r['median_log2_R_advantage'])}",
                "comparison": f"top states tested={fmt(r['top_states_tested'],0)}",
                "evidence": "results/experiment_15_uncertainty_negative_controls/tables/matched_decoy_summary.tsv",
                "interpretation": "Top states remain high against matched decoys, reducing concern that results are only stage/event-count artifacts.",
            }
        )
    e15b = read_table("results/experiment_15_uncertainty_negative_controls/tables/inflow_pairing_falsification_summary.tsv")
    for _, r in e15b.iterrows():
        rows.append(
            {
                "experiment": "E15B",
                "dataset": r["dataset_name"],
                "endpoint": "inflow pairing shuffle",
                "metric": "median shuffled overlap / overlap loss",
                "result": f"{fmt(r['median_shuffled_overlap'])} / {fmt(r['median_overlap_loss'])}",
                "comparison": f"repeats={fmt(r['repeats'],0)}, exact recovery={fmt(r['exact_recovery_fraction'])}",
                "evidence": "results/experiment_15_uncertainty_negative_controls/tables/inflow_pairing_falsification_summary.tsv",
                "interpretation": "R* depends on correct state-specific L-F pairing rather than marginals alone.",
            }
        )

    e16 = read_table("results/experiment_16_real_topology/tables/real_topology_audit.tsv")
    for _, r in e16.iterrows():
        rows.append(
            {
                "experiment": "E16",
                "dataset": r["dataset_name"],
                "endpoint": "real topology route display",
                "metric": "display paths / unique nodes / median target R*",
                "result": f"{fmt(r['display_paths'],0)} / {fmt(r['unique_nodes'],0)} / {fmt(r['median_target_R_star'])}",
                "comparison": f"top-R* paths={fmt(r['top_rstar_paths'],0)}, long-event paths={fmt(r['long_event_rstar_paths'],0)}",
                "evidence": "results/experiment_16_real_topology/tables/real_topology_audit.tsv",
                "interpretation": "Directly visualizes real-data progression routes annotated by relative dwell signal.",
            }
        )

    e17 = read_table("results/experiment_17_longitudinal_public/tables/integrated_longitudinal_metrics_table.tsv")
    if e17.empty:
        e17 = read_table("results/experiment_17_longitudinal_public/tables/core_metric_table.tsv")
        for _, r in e17.iterrows():
            rows.append(
                {
                    "experiment": "E17",
                    "dataset": r["cohort"],
                    "endpoint": "external longitudinal validation",
                    "metric": "AUC / AP lift / top-bottom persistence / rho minimum dwell proxy",
                    "result": f"{r['AUC_95CI']}; AP lift={fmt(r['AP_lift'])}; Δ={r['Delta_persist_95CI']}; rho={r['rho_minimum_dwell_95CI']}",
                    "comparison": f"n(P/C)={r['n_P_C']}",
                    "evidence": "results/experiment_17_longitudinal_public/tables/core_metric_table.tsv",
                    "interpretation": "External longitudinal evidence is directionally supportive, with limited pair counts.",
                }
            )
    else:
        for _, r in e17.iterrows():
            rows.append(
                {
                    "experiment": "E17",
                    "dataset": r["cohort"],
                    "endpoint": f"external longitudinal validation ({r['evidence_role']})",
                    "metric": "AUC / AP lift / tertile Δ persistence / rho minimum dwell proxy",
                    "result": f"{r['auc_95ci']}; AP lift={fmt(r['ap_lift'])}; Δ={r['tertile_delta_persistence_95ci']}; rho={r['tertile_rho_minimum_dwell_95ci']}",
                    "comparison": f"n(P/C)={r['n_P_C']}",
                    "evidence": "results/experiment_17_longitudinal_public/tables/integrated_longitudinal_metrics_table.tsv",
                    "interpretation": "External longitudinal evidence directly tests whether high R* states persist longer, but remains sample-size limited.",
                }
            )

    return rows


def claim_rows() -> list[dict]:
    return [
        {
            "claim": "Rel-ObsTQ-MHN defines a state-level relative dwell-time index from cross-sectional genomic data.",
            "evidence": "Formula and implementation in core scoring/transitions/pipeline; E3-E5 real cohort execution.",
            "strength": "Strong for computational definition and implementation.",
            "safe_wording": "estimates a relative dwell/stasis proxy; does not measure absolute calendar time.",
            "caveat": "Requires a progression transition model and sufficient state support.",
        },
        {
            "claim": "R* is not equivalent to raw occupancy or MHN inflow alone.",
            "evidence": "E11 top-overlap, rank-gain and correlation results; E14 denominator ablation; E15 falsification.",
            "strength": "Strong.",
            "safe_wording": "R* adds state-specific information beyond prevalence and model inflow.",
            "caveat": "Information gain is shown on selected three AACR cohorts.",
        },
        {
            "claim": "R* can recover known relative dwell structure in controlled simulations.",
            "evidence": "E6 enhanced bottleneck recovery and E6 continuous dwell-gradient.",
            "strength": "Strong.",
            "safe_wording": "R* outperforms occupancy in controlled settings with known dwell truth.",
            "caveat": "Simulations still simplify real tumor evolution and use known/inferred topology assumptions.",
        },
        {
            "claim": "R* remains robust under changes in topology, sparsity and bottleneck placement.",
            "evidence": "E7 balanced topology grid: R* global Spearman 0.482 vs occupancy 0.255.",
            "strength": "Moderate to strong.",
            "safe_wording": "R* retains directional advantage across stress conditions.",
            "caveat": "The gain is condition-dependent and not a universal perfect recovery guarantee.",
        },
        {
            "claim": "Real AACR high-R* states are biologically plausible in LUAD, COAD and IDC.",
            "evidence": "E8 and E10 top states/modules; E16 route overlays.",
            "strength": "Moderate to strong as biological plausibility evidence.",
            "safe_wording": "high-R* states are enriched for recognizable tumor-type-relevant driver contexts.",
            "caveat": "Module p-values in E8 are not strong enough to claim formal pathway enrichment.",
        },
        {
            "claim": "R* has clinical relevance.",
            "evidence": "E12 subgroup Cox and survival profiles.",
            "strength": "Supportive but secondary.",
            "safe_wording": "R* is associated with clinical outcome in several cohort/subgroup settings.",
            "caveat": "IDC shows mixed direction; survival is not a direct dwell-time label.",
        },
        {
            "claim": "R* generalizes to held-out/internal splits.",
            "evidence": "E13 patient split replication: median rho around 0.87-0.90, top10 enrichment >5x.",
            "strength": "Strong for internal stability.",
            "safe_wording": "R* ranking is stable under patient resampling/splitting.",
            "caveat": "Backbone was fixed from full E5 outputs according to current review; independent split refit would be stronger.",
        },
        {
            "claim": "R* predictions are directionally consistent with real longitudinal persistence.",
            "evidence": "E17 GLASS, CRC-triplets, MNM-WashU integrated metrics.",
            "strength": "Moderate and direct, with sample-size caveat.",
            "safe_wording": "external longitudinal cohorts provide directional support for high-R* states being more persistent.",
            "caveat": "Small evaluable pairs, particularly CRC/MNM changed pairs; fallback backbone in E17 should be disclosed.",
        },
        {
            "claim": "O* identifies cross-sectional enrichment not expected from progression-only models.",
            "evidence": "E9 O* simulation.",
            "strength": "Strong for auxiliary O* positive control.",
            "safe_wording": "O* is a companion residual for observation enrichment, not the main R* dwell claim.",
            "caveat": "O* should not be over-positioned as the primary novelty.",
        },
    ]


def method_mapping_rows() -> list[dict]:
    return [
        {
            "concept": "State/genotype representation",
            "definition": "stage::gene1+gene2 state plus canonical binary event vector.",
            "code": "src/relobstq_mhn/core/states.py:122 build_state_occupancy",
            "experiment_use": "E1/E2 data tables; E4-E5 state scoring; E16 route display; E17 longitudinal states.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "MHN one-step event addition probability",
            "definition": "Softmax over absent event log-rates derived from theta diagonal and context terms.",
            "code": "src/relobstq_mhn/core/transitions.py:21 softmax_addition_probabilities",
            "experiment_use": "E3 interface; E4 inflow; E5/E10/E16 scoring and display.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Same-stage one-step predecessor graph",
            "definition": "Edges from u to v when v adds exactly one event and stage is unchanged.",
            "code": "src/relobstq_mhn/core/transitions.py:49 same_stage_one_step_edges",
            "experiment_use": "E4 inflow rule, E16 real topology paths.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Expected inflow F_hat",
            "definition": "Sum over predecessor occupancy times edge-addition probability.",
            "code": "src/relobstq_mhn/core/transitions.py:106 aggregate_inflow",
            "experiment_use": "E4, E5, E10, E11, E14, E15, E16.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Relative dwell index R*",
            "definition": "R_raw=L/(F_hat+epsilon), median-normalized among eligible states.",
            "code": "src/relobstq_mhn/core/scoring.py:21 compute_relative_dwell",
            "experiment_use": "All core method validation experiments E5-E8, E10-E17.",
            "audit": "Primary novelty; confirmed from code.",
        },
        {
            "concept": "Observation enrichment O*",
            "definition": "Observed occupancy over progression-only expected occupancy.",
            "code": "src/relobstq_mhn/core/scoring.py:65 compute_observation_enrichment",
            "experiment_use": "E5/E9/E10 as auxiliary analysis.",
            "audit": "Confirmed; secondary metric.",
        },
        {
            "concept": "Bootstrap uncertainty",
            "definition": "Multinomial resampling of state counts and recomputation of F_hat/R*.",
            "code": "src/relobstq_mhn/core/bootstrap.py:24 bootstrap_relative_dwell",
            "experiment_use": "E5, E6 variants, E17 intervals.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Integrated scoring pipeline",
            "definition": "State occupancy + theta + event set -> edges, F_hat, R*, optional bootstrap.",
            "code": "src/relobstq_mhn/core/pipeline.py:20 score_states_from_mhn",
            "experiment_use": "Reusable method package for experiments.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Dominant predecessor topology route",
            "definition": "Backtrack through dominant predecessor to display a representative inflow-supported path.",
            "code": "src/relobstq_mhn/core/topology.py:22 build_dominant_predecessor_path",
            "experiment_use": "E16 real topology and E17 route figures.",
            "audit": "Confirmed; descriptive route, not full phylogenetic tree.",
        },
        {
            "concept": "Experiment-ready data construction",
            "definition": "Metadata, event matrix, state table and MHN matrix construction.",
            "code": "src/relobstq_mhn/data/processing.py:220 build_experiment_ready_tables",
            "experiment_use": "Data processing package; raw-to-analysis interface.",
            "audit": "Confirmed from code.",
        },
        {
            "concept": "Simulation generator",
            "definition": "cMHN-like trajectory simulation with state dwell multipliers and audit outputs.",
            "code": "src/relobstq_mhn/simulation/generator.py:183 simulate_cohort_with_audit",
            "experiment_use": "E6, E6-gradient, E7, E9.",
            "audit": "Confirmed; should be described as controlled positive-control simulator.",
        },
    ]


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text.rstrip() + "\n", encoding="utf-8")


def build_results_master() -> str:
    rows = core_result_rows()
    return "\n\n".join(
        [
            "# RESULTS_MASTER_TABLE",
            "",
            "本表按“实验-数据集-指标-证据文件”整理当前仓库中可确认的核心结果。数值直接来自现有 CSV/TSV 表；没有读到的结果不在此表中扩写。",
            "",
            md_table(
                rows,
                [
                    "experiment",
                    "dataset",
                    "endpoint",
                    "metric",
                    "result",
                    "comparison",
                    "evidence",
                    "interpretation",
                ],
            ),
        ]
    )


def build_claim_matrix() -> str:
    return "\n\n".join(
        [
            "# CLAIM_EVIDENCE_MATRIX",
            "",
            "目标是把论文中的每一句主要 claim 约束在已有证据边界内，避免把“相对停留时间代理”写成“真实绝对时间”。",
            "",
            md_table(claim_rows(), ["claim", "evidence", "strength", "safe_wording", "caveat"]),
        ]
    )


def build_method_mapping() -> str:
    return "\n\n".join(
        [
            "# METHOD_CODE_MAPPING",
            "",
            "本表给出方法概念、数学含义、实现位置和实验使用位置。代码位置以当前仓库审计为准。",
            "",
            md_table(method_mapping_rows(), ["concept", "definition", "code", "experiment_use", "audit"]),
            "",
            "## Minimal Computational Contract",
            "",
            "- 输入：样本级二值事件矩阵、stage/metastasis 分组、状态表、MHN theta 或可替代的转移概率 backbone。",
            "- 输出：`state`, `N_v`, `L_v`, `F_hat`, `R_raw`, `R_star`, `log2_R_star`, eligibility/high-confidence flags, predecessor/edge tables, bootstrap intervals where requested.",
            "- 必须披露：事件筛选、最小状态数、F_hat eligibility、是否使用真实 MHN backend、是否使用 fallback backbone。",
        ]
    )


def build_figure_plan() -> str:
    inv = figure_inventory_rows()
    main_figures = [
        {
            "figure": "Figure 1",
            "role": "方法框架",
            "recommended_panels": "input state table -> MHN one-step graph -> F_hat denominator -> R* relative dwell map",
            "source": "core method code plus schematic, not a result-only plot",
            "message": "把创新点定义清楚：R* = observed occupancy / progression-expected inflow, normalized.",
        },
        {
            "figure": "Figure 2",
            "role": "真实横断面主队列输入与主输出",
            "recommended_panels": "E1/E2 data readiness, E4 inflow feasibility table/plot, E5/E10 high-R* summary",
            "source": "results/experiments_01_02; results/experiment_04_relative_inflow; results/experiment_05_state_scores; results/experiment_10_real_cohort_main",
            "message": "三癌种真实队列可被统一处理并产生稳定的 state-level R* 结果。",
        },
        {
            "figure": "Figure 3",
            "role": "模拟真值验证",
            "recommended_panels": "E6 enhanced bottleneck table, E6 dwell-gradient compact statistical plot, E7 robustness panel",
            "source": "results/experiment_06_bottleneck_recovery_enhanced; results/experiment_06_dwell_gradient; results/experiment_07_topology_robustness_balanced",
            "message": "在已知真值下 R* 能恢复相对停留时间强弱，且不是只在过易二分类中有效。",
        },
        {
            "figure": "Figure 4",
            "role": "真实生物学可解释性与非冗余性",
            "recommended_panels": "E8 biological convergence, E11 information gain, E14 denominator ablation, E15 falsification",
            "source": "results/experiment_08_biological_convergence; results/experiment_11_information_gain; results/experiment_14_ablation_backbone; results/experiment_15_uncertainty_negative_controls",
            "message": "高 R* 状态既有生物学可解释性，也不是 raw occupancy/MHN-only 的简单重命名。",
        },
        {
            "figure": "Figure 5",
            "role": "真实数据拓扑直观展示",
            "recommended_panels": "E16 three-cohort real topology routes with R* overlay",
            "source": "results/experiment_16_real_topology",
            "message": "从真实输入到方法处理再到带 R* 的进化路线，直接展示创新点。",
        },
        {
            "figure": "Figure 6",
            "role": "真实纵向外部验证",
            "recommended_panels": "E17 core metrics table + top-bottom persistence + minimum dwell proxy correlation",
            "source": "results/experiment_17_longitudinal_public",
            "message": "高 R* 状态在纵向配对中更倾向保持，形成真实纵向方向性支持。",
        },
    ]
    supplements = [
        {
            "item": "Supplementary Figure S1-S2",
            "content": "E1/E2 detailed QC, oncoprint, sparsity and stage sensitivity.",
            "reason": "数据准备必要但不应抢占主文篇幅。",
        },
        {
            "item": "Supplementary Figure S3",
            "content": "E3 MHN interface details and one-step edge examples.",
            "reason": "说明接口正确性。",
        },
        {
            "item": "Supplementary Table S1-SN",
            "content": "All per-state scores, edges, split repeats, bootstrap intervals, longitudinal pair-level outputs.",
            "reason": "保证可审查和可复现。",
        },
    ]
    return "\n\n".join(
        [
            "# FIGURE_PLAN",
            "",
            "## Main Figure Strategy",
            "",
            "主文需要保留“真实横断面队列 -> 模拟真值 -> 真实纵向队列”的全链条。建议不要再把核心主图过度压缩成表格；表格服务于核心指标，图服务于机制与直观证据。",
            "",
            md_table(main_figures, ["figure", "role", "recommended_panels", "source", "message"]),
            "",
            "## Supplementary Strategy",
            "",
            md_table(supplements, ["item", "content", "reason"]),
            "",
            "## Current Figure File Inventory",
            "",
            md_table(inv, ["result_root", "png_total", "pdf_total", "single_png", "figure_png"]),
            "",
            "## Style Rules",
            "",
            "- 使用当前公共配色模板：`#B5AED5`, `#B2E6FD`, `#B8D2CC`, `#E8B2A7`, `#FEEBB9`；辅助灰黑用中性灰，不新增竞争性主色。",
            "- 主文单图优先接近正方形或紧凑矩形；组图后正文可读字号不低于期刊常见 6-8 pt 的最低线。",
            "- 不在单图中保留 A/B/C/D panel title；panel 字母由后期组图统一添加。",
            "- 表格用于精确指标和 CI；图用于机制、趋势、拓扑、分布。",
            "- 所有图导出时保留足够边距，尤其网络节点、CI 端帽、右侧图例和顶部热力色带。",
        ]
    )


def build_todo() -> str:
    p0 = [
        {
            "priority": "P0",
            "task": "统一数据版本叙述：当前本地 AACR 文件显示 GENIE Cohort v18.0-public；README_RUN 仍提到 PACA-CA，需更新或在论文材料中避免使用旧说法。",
            "why": "版本和队列不一致会被审稿人立即抓住。",
            "owner": "methods/data",
        },
        {
            "priority": "P0",
            "task": "明确 Python/MHN 环境：当前运行 Python 3.13.5，但 `mhn==1.2.3` 要求 python_version < 3.13；需要提供 Python 3.11/3.12 复现环境或明确 fallback backend 的范围。",
            "why": "这是可复现性和方法真实性的核心风险。",
            "owner": "reproducibility",
        },
        {
            "priority": "P0",
            "task": "在正文中严格限定 R*：相对停留/滞留 proxy，不是绝对时间、不等于临床随访时间。",
            "why": "避免概念夸大。",
            "owner": "writing",
        },
        {
            "priority": "P0",
            "task": "披露 E17 的 fallback/frequency-cooccurrence backbone 状态，不能把所有纵向验证都写成 full cMHN-trained validation。",
            "why": "外部纵向验证是最容易被追问的部分。",
            "owner": "E17",
        },
        {
            "priority": "P0",
            "task": "决定最终主文是否纳入 MNM-WashU：指标方向好，但 n=10 且 changed=1，更适合 supplementary/supportive cohort。",
            "why": "小样本可能抬高 AUC，被质疑。",
            "owner": "results",
        },
    ]
    p1 = [
        {
            "priority": "P1",
            "task": "若时间允许，为 E13 增加 independent split refit 版本或在局限性中说明 fixed backbone。",
            "why": "可加强泛化性。",
            "owner": "E13",
        },
        {
            "priority": "P1",
            "task": "临床 E12 做多重检验/方向一致性说明，IDC mixed result 需要客观呈现。",
            "why": "临床相关性不是主证据，不能选择性报告。",
            "owner": "E12",
        },
        {
            "priority": "P1",
            "task": "整理 all single figures 与 final figure assembly 清单，避免中间版混入投稿材料。",
            "why": "图件版本混乱会影响投稿质量。",
            "owner": "figures",
        },
        {
            "priority": "P1",
            "task": "补充 runtime/hardware 和 random seed manifest。",
            "why": "提升工程可复现性。",
            "owner": "reproducibility",
        },
    ]
    p2 = [
        {
            "priority": "P2",
            "task": "将所有 per-state/edge/pair-level 大表组织为 supplementary tables，并在正文只引用关键表。",
            "why": "提升论文可读性。",
            "owner": "submission package",
        },
        {
            "priority": "P2",
            "task": "准备审稿答辩模板：为什么 R* 不是 occupancy、为什么不是 MHN-only、为什么 E17 样本小仍有价值。",
            "why": "提前消化 Reviewer #2 风险。",
            "owner": "response planning",
        },
    ]
    return "\n\n".join(
        [
            "# MANUSCRIPT_TODO",
            "",
            "## P0 Must Fix Before Submission",
            "",
            md_table(p0, ["priority", "task", "why", "owner"]),
            "",
            "## P1 Should Fix If Time Allows",
            "",
            md_table(p1, ["priority", "task", "why", "owner"]),
            "",
            "## P2 Polish / Supplementary Work",
            "",
            md_table(p2, ["priority", "task", "why", "owner"]),
        ]
    )


def build_master_plan() -> str:
    datasets = selected_dataset_rows()
    excluded = excluded_dataset_rows()
    registry = experiment_registry_rows()
    results = core_result_rows()
    claims = claim_rows()
    mappings = method_mapping_rows()
    aacr_version = get_aacr_version()
    req = read_text("requirements.txt")
    py_version = sys.version.split()[0]

    key_result_excerpt = [
        r
        for r in results
        if r["experiment"] in {"E6", "E6-gradient", "E7", "E11", "E12", "E13", "E15A", "E15B", "E16", "E17"}
    ][:50]

    reviewer_risks = [
        {
            "risk": "R* 是否只是 occupancy 的变形？",
            "answer": "E11 显示 top overlap 较低，rho(R*,F) 为负/弱相关而 rho(L,F) 较强；E14/E15 进一步显示 denominator 和 pairing 具有特异性。",
            "remaining": "正文需避免只展示漂亮 top states；必须给全队列统计。",
        },
        {
            "risk": "E6 AUC=1.000 是否太容易？",
            "answer": "保留 E6 enhanced 作为 positive control，同时用 E6-gradient 连续停留梯度证明排序/校准也成立。",
            "remaining": "报告 AUC=1 的 repeat fraction，不只报单点中位数。",
        },
        {
            "risk": "真实纵向验证是否足够强？",
            "answer": "E17 是直接证据：GLASS AUC 0.67，CRC 0.65，MNM 0.89，方向一致。",
            "remaining": "样本数小，尤其 MNM changed=1；主文措辞应是 directional support。",
        },
        {
            "risk": "是否使用了真实 MHN？",
            "answer": "E3 在三 AACR 队列完成 MHN interface；但 E17 纵向 cohort 使用 fallback/frequency-cooccurrence backbone 的证据需要披露。",
            "remaining": "需要环境 pinning 来保证 mhn==1.2.3 可复现。",
        },
        {
            "risk": "临床结果是否被过度解释？",
            "answer": "E12 只能作为 secondary association；HR 方向在 IDC 有混合结果。",
            "remaining": "不要用临床生存证明 dwell-time truth。",
        },
        {
            "risk": "跨癌种是否混合训练导致偏差？",
            "answer": "配置显示三个 AACR 癌种独立训练、独立评分，只在结果层比较。",
            "remaining": "需要在 Methods 明写。",
        },
    ]

    maturity = [
        {"dimension": "创新点清晰度", "score": "8.5/10", "basis": "R*=L/F_hat median-normalized 的主线清楚，且和 MHN-only/occupancy-only 区分明确。"},
        {"dimension": "真实横断面证据", "score": "8/10", "basis": "LUAD/COAD/IDC 样本量充足，处理链完整；PACA 删除合理。"},
        {"dimension": "模拟真值证据", "score": "8.5/10", "basis": "E6 positive control + continuous gradient + robustness grid，能够覆盖核心创新点。"},
        {"dimension": "真实纵向证据", "score": "6.5/10", "basis": "方向支持但 pair count 有限，MNM 仅补充。"},
        {"dimension": "临床与生物学解释", "score": "7/10", "basis": "多处支持但应作为辅助，避免夸大。"},
        {"dimension": "工程可复现性", "score": "7/10", "basis": "方法包结构较完整；Python/MHN 版本兼容性需 P0 解决。"},
        {"dimension": "投稿成熟度", "score": "7.5/10", "basis": "可形成完整论文，但需要修正文档漂移、环境声明和 E17 局限措辞。"},
    ]

    sections = [
        "# MANUSCRIPT_MASTER_PLAN",
        "",
        "生成日期：2026-08-31",
        "",
        "## 0. 审计规则",
        "",
        "本文件基于当前 VSCode 工作区的代码、配置、结果表、审查报告和数据 manifest 生成。未能从文件确认的内容标为 `【待确认】`；需要补充但当前缺失的内容标为 `【建议补充】`；基于代码结构和结果链条得出的解释标为 `【解释性推断】`。",
        "",
        "## 1. Editor-Level Verdict",
        "",
        "从顶级期刊审稿/编辑视角看，这个项目已经形成一条相对完整的证据链：真实横断面队列提出并运行 R*，模拟真值证明 R* 能恢复相对停留时间，真实纵向队列提供方向一致的外部验证，消融和反事实证明 R* 不是 occupancy 或 MHN-only 的简单替代。",
        "",
        "最稳妥的主结论应写为：Rel-ObsTQ-MHN estimates a state-level relative dwell-time proxy by contrasting observed cross-sectional state occupancy with MHN-derived progression-expected inflow. 该方法不能声称直接测量绝对演化时间，但可以比较状态间“相对更滞留/更短暂”的倾向。",
        "",
        "当前最强证据是 E6/E6-gradient/E7 的模拟真值验证、E11/E14/E15 的非冗余性和 falsification、E13 的内部稳定性，以及 E16 的真实拓扑可视化。最薄弱但也最直接的是 E17 真实纵向验证：方向上支持创新点，但 pair count 有限，需要把它作为 direct but sample-size-limited validation，而不是唯一决定性证据。",
        "",
        "## 2. Confirmed Project Identity",
        "",
        f"- 当前本地 Python：`{py_version}`。",
        f"- Python/MHN 依赖风险：`requirements.txt` 包含 `{next((line for line in req.splitlines() if 'mhn==' in line), '【待确认】')}`；因此当前 Python 3.13.5 与官方 mhn 依赖条件不完全匹配，需要在复现环境中使用 Python <3.13 或清楚说明 fallback/backend 使用范围。",
        f"- AACR/GENIE 数据版本：`{aacr_version}`，证据文件 `{rel(ROOT / 'Data/AACR/AACR/meta_study.txt')}`。",
        "- 当前主实验横断面队列：AACR_LUAD、AACR_COAD、AACR_IDC。",
        "- PACA-CA 已从当前主配置排除；如果 README 或旧报告仍出现 PACA，应视为文档漂移。",
        "",
        "## 3. Data Cohorts",
        "",
        "### 3.1 Included Real Cross-Sectional Cohorts",
        "",
        md_table(
            datasets,
            [
                "dataset",
                "role",
                "analysis_units",
                "patients",
                "mutation_rows",
                "events_retained",
                "valid_states_manifest",
                "valid_states_E1",
                "zero_event_fraction",
                "manifest_checks",
                "warnings",
            ],
        ),
        "",
        "### 3.2 Excluded Datasets",
        "",
        md_table(excluded, ["dataset", "reason"]),
        "",
        "## 4. Core Scientific Question",
        "",
        "横断面肿瘤测序数据只能观察到某个患者/样本落在哪个突变状态，通常无法直接看到该状态停留了多长时间。项目的核心问题是：能否利用 MHN 给出的 progression transition tendency，构造一个状态层面的相对停留时间指标，使“观察到很多但按进展模型不该这么常见”的状态被识别为相对滞留/稳定状态？",
        "",
        "## 5. Main Hypothesis",
        "",
        "如果某个状态 v 的横断面占有率 L_v 高于其从前驱状态按 MHN 进展流入所预期的 F_hat_v，那么该状态更可能具有较长相对停留/滞留时间。反之，如果 L_v 低于 F_hat_v，则可能是相对短暂或快速过渡状态。",
        "",
        "## 6. Mathematical Definition",
        "",
        "令 `v` 为一个 stage-specific genotype state，`N_v` 为该状态样本数，`N` 为 cohort 总样本数：",
        "",
        "- Observed occupancy: `L_v = N_v / N`。",
        "- 对每个 one-step predecessor `u -> v`，MHN theta 给出增加缺失事件 j 的相对概率：`P(u -> v | theta)`。",
        "- Expected progression inflow: `F_hat_v = sum_u L_u * P(u -> v | theta)`。",
        "- Raw relative dwell: `R_raw_v = L_v / (F_hat_v + epsilon)`。",
        "- Median-normalized relative dwell: `R*_v = R_raw_v / median(R_raw among eligible states)`。",
        "- 常用展示量：`log2 R*_v`，其中 `R*>1` 表示相对更滞留，`R*<1` 表示相对更短暂。",
        "- Auxiliary observation enrichment: `O*_v = L_v / (Lhat_progression_v + epsilon)`，主要用于识别 progression-only 预期之外的观察富集，不是主创新点。",
        "",
        "## 7. Code Evidence Chain",
        "",
        md_table(mappings, ["concept", "definition", "code", "experiment_use", "audit"]),
        "",
        "## 8. Experiment Registry",
        "",
        "以下来自当前 `configs/experiment_registry.yaml`。若某些旧结果目录仍在 results 下但不在 registry 主线，应归为 legacy/intermediate。",
        "",
        md_table(registry, ["id", "name", "status", "result_root", "primary"]),
        "",
        "## 9. Experiment Chain Interpretation",
        "",
        "### E1-E2: Data Preparation and Stage-Sensitivity QC",
        "",
        "作用：证明三套 AACR/GENIE 癌种队列能被统一处理成 mutation event matrix、state table、metadata 和 MHN training matrix。它是所有后续实验的地基。注意：E1/2 validation markdown 显示 per-cohort checks OK；experiment-ready manifest 也显示结构检查 all True；若旧 summary 中有综合 flag 异常，应以 validation/manifest 逐项解释，不要笼统省略。",
        "",
        "### E3-E4: MHN Interface and Relative Inflow",
        "",
        "作用：把 MHN 学到的事件依赖结构转成 one-step transition probabilities，再按同 stage predecessor 汇总为 `F_hat`。这一步是从普通 MHN 到 Rel-ObsTQ-MHN 的关键接口。",
        "",
        "### E5/E10: Real-Cohort State Scores",
        "",
        "作用：在真实 LUAD/COAD/IDC 中给每个 eligible state 计算 R* 和 O*，得到真实数据下的 high-R* states。E10 更像主结果汇总，E5 更像 state score 方法展开。",
        "",
        "### E6 and E6-Gradient: Simulated Truth Validation",
        "",
        "作用：在已知真实 dwell multipliers 的模拟环境中评价 R* 是否能恢复相对停留时间。E6 enhanced 证明极端 bottleneck 可以被识别；E6-gradient 进一步证明在连续停留梯度上也能排序和校准，解决 “AUC=1 是否太容易” 的质疑。",
        "",
        "### E7: Topology Robustness",
        "",
        "作用：在不同拓扑、稀疏性、bottleneck 位置下测试 R* 对相对停留时间排序的稳健性。它不是为了证明每个条件都完美，而是证明方向性优势在压力条件下仍存在。",
        "",
        "### E8: Biological Convergence",
        "",
        "作用：检查 high-R* states 是否落在 LUAD/COAD/IDC 合理的 driver/module 背景中。它支持生物学可解释性，但不能过度写成强 pathway enrichment，因为当前 module p 值并不强。",
        "",
        "### E9: O* Simulation",
        "",
        "作用：验证 O* 可以识别 progression-only 预期之外的观察富集。它服务于辅助指标，不能喧宾夺主。",
        "",
        "### E11: Information Gain",
        "",
        "作用：证明 R* 不是 raw occupancy，也不是 MHN-only inflow score。这个实验直接保护创新点，非常重要。",
        "",
        "### E12: Clinical Validation",
        "",
        "作用：检验 R* 与临床生存/风险之间是否有关联。它是 secondary validation。结果中 COAD 较稳，LUAD 分层有差异，IDC 有混合方向，因此应客观写。",
        "",
        "### E13: Split-Cohort Replication",
        "",
        "作用：证明 R* ranking 在患者拆分中稳定。当前证据强，但根据已有审查，它更接近 fixed-backbone held-out occupancy stability；如果要声称 fully independent generalization，需要补充 independent refit 或谨慎措辞。",
        "",
        "### E14-E15: Ablation and Falsification",
        "",
        "作用：回答“是不是换个 denominator 也一样”“是不是 matched decoy 也会高”“是不是打乱 inflow pairing 也能恢复”。当前结果支持 R* 对 MHN-derived denominator 和 state-specific pairing 的依赖，属于创新点防御实验。",
        "",
        "### E16: Real Topology with R* Overlay",
        "",
        "作用：这是你前面指出缺失的“从真实数据输入到方法处理再到带相对停留时间的基因进化拓扑”的直观实验。它应成为主文中的核心展示之一。注意它展示的是 dominant predecessor route，不是完整单细胞谱系树。",
        "",
        "### E17: Public Longitudinal Validation",
        "",
        "作用：用真实纵向配对数据检验 high-R* state 是否更容易在后续样本中保持，以及 R* 是否和最小观察停留时间 proxy 方向一致。它是最直接但样本最有限的外部验证。",
        "",
        "## 10. Core Results Excerpt",
        "",
        md_table(
            key_result_excerpt,
            ["experiment", "dataset", "endpoint", "metric", "result", "comparison", "evidence", "interpretation"],
        ),
        "",
        "完整结果表见 `RESULTS_MASTER_TABLE.md`。",
        "",
        "## 11. Claim Boundaries",
        "",
        md_table(claims, ["claim", "evidence", "strength", "safe_wording", "caveat"]),
        "",
        "## 12. Recommended Manuscript Story",
        "",
        "建议主文叙事顺序：",
        "",
        "1. 传统横断面测序能看到状态频率，但不能直接知道状态相对停留长短。",
        "2. MHN 提供 progression inflow expectation，但 MHN 本身不输出相对停留时间。",
        "3. Rel-ObsTQ-MHN 的创新是把 observed occupancy 与 MHN-derived expected inflow 相除，形成 state-level `R*`。",
        "4. 在三套真实 AACR/GENIE 癌种队列中，R* 能稳定产生并揭示高滞留候选状态。",
        "5. 在已知真值模拟中，R* 能恢复离散 bottleneck 和连续 dwell gradient。",
        "6. R* 与 occupancy/MHN-only 不等价，消融和 falsification 排除了主要平凡解释。",
        "7. 真实拓扑图展示了从基因事件到状态滞留的可解释进化路径。",
        "8. 公开纵向队列显示高 R* 状态更倾向保持，为核心创新提供直接但样本有限的外部支持。",
        "",
        "## 13. Reviewer #2 Simulation",
        "",
        md_table(reviewer_risks, ["risk", "answer", "remaining"]),
        "",
        "## 14. Statistical Audit",
        "",
        "- 优先报告 median/IQR 或 bootstrap CI，而不是只报告单点最好值。",
        "- E6 AUC=1.000 要同时报告 mean/sd、min-max 和 perfect repeat fraction，避免过易质疑。",
        "- E17 应报告 `n(P/C)`，因为 changed pair 数量对 AUC/CI 影响巨大。",
        "- E12 临床结果必须报告 HR、CI、p、n/events 和 subgroup；IDC mixed direction 不能隐藏。",
        "- E8 模块分析的 p 值不强，建议写 biological coherence/plausibility，不写显著富集。",
        "- E13 如果 backbone 固定，应该写 split stability under fixed fitted backbone，而不是完全独立复现实验。",
        "",
        "## 15. Reproducibility Workflow",
        "",
        "建议论文补充材料给出如下复现顺序：",
        "",
        "1. 轻量测试记录：`D:\\ai\\python.exe -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q` 当前 6 passed；`D:\\ai\\python.exe -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q` 当前 5 passed。",
        "2. 注意：不要把两个同名 `test_relobstq_core.py` 放在同一条 pytest collection 命令里，否则会触发 pytest import mismatch；这属于测试命名/collection 问题，不是方法函数失败。",
        "3. 运行数据处理脚本生成 `processed/experiment_ready/AACR_LUAD`, `AACR_COAD`, `AACR_IDC`。",
        "4. 运行 E1-E5 建立真实横断面主链。",
        "5. 运行 E6/E6-gradient/E7/E9 建立模拟真值链。",
        "6. 运行 E8/E10/E11/E12/E13/E14/E15/E16 建立真实数据解释、消融、临床和拓扑链。",
        "7. 运行 E17 建立公开纵向验证链。",
        "8. 运行当前脚本：`D:\\ai\\python.exe src/build_manuscript_master_plan.py` 重新生成论文总控文档。",
        "",
        "## 16. Maturity Score",
        "",
        md_table(maturity, ["dimension", "score", "basis"]),
        "",
        "## 17. Bottom Line",
        "",
        "客观评价：当前实验链条已经足以支持一个清晰、谨慎、有创新点的 SCI 方法学论文。最成功的结论是：R* 是一个能从横断面队列中估计 state-level relative dwell/stasis 的新指标，并且在模拟真值、真实横断面、消融、拓扑和有限纵向真实数据中得到一致支持。最不能过度声称的是：R* 直接等于真实绝对停留时间，或者 E17 已经给出大规模纵向决定性验证。",
    ]
    return "\n\n".join(sections)


def main() -> None:
    outputs = {
        "MANUSCRIPT_MASTER_PLAN.md": build_master_plan(),
        "RESULTS_MASTER_TABLE.md": build_results_master(),
        "CLAIM_EVIDENCE_MATRIX.md": build_claim_matrix(),
        "METHOD_CODE_MAPPING.md": build_method_mapping(),
        "FIGURE_PLAN.md": build_figure_plan(),
        "MANUSCRIPT_TODO.md": build_todo(),
    }
    for path, text in outputs.items():
        write(path, text)
        print(f"Wrote {path} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
