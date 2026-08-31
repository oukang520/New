"""Run Experiment 17 extension on four newly collected longitudinal cohorts.

The validation question is deliberately narrow: do Rel-ObsTQ-derived R* state
scores, learned from cross-sectional state occupancy plus an MHN/fallback
backbone, agree with observed longitudinal dwell/persistence in independent
public cohorts?
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

import figure_style
import run_experiment_17_longitudinal_public as e17


CONFIG_PATH = Path("src/relobstq_mhn/configs/experiment_17_new_four_cohorts.yaml")

STUDY_PRIORITY_EVENTS = {
    "aml_target_2018_pub": [
        "NRAS",
        "FLT3",
        "KIT",
        "KRAS",
        "PTPN11",
        "TET2",
        "WT1",
        "IDH2",
        "IDH1",
        "NPM1",
        "RUNX1",
        "ASXL1",
    ],
    "lipo_msk_2026": [
        "FRS2",
        "ATRX",
        "TP53",
        "RB1",
        "NF1",
        "CDK4",
        "MDM2",
        "DDIT3",
        "ZFHX3",
        "NOTCH3",
    ],
    "breast_msk_2018": [
        "PIK3CA",
        "TP53",
        "CDH1",
        "GATA3",
        "ESR1",
        "MAP3K1",
        "PTEN",
        "AKT1",
        "ERBB2",
        "NF1",
        "ERBB2_AMP",
        "CCND1_AMP",
        "MYC_AMP",
        "FGFR1_AMP",
        "PTEN_DEL",
        "RB1_DEL",
    ],
    "mnm_washu_2016": [
        "TP53",
        "ASXL1",
        "SRSF2",
        "IDH2",
        "DNMT3A",
        "SF3B1",
        "RUNX1",
        "TET2",
        "IDH1",
        "NRAS",
        "U2AF1",
        "NPM1",
    ],
}

STUDY_CNA_RULES = {
    "breast_msk_2018": {
        "ERBB2": [("ERBB2_AMP", 2)],
        "CCND1": [("CCND1_AMP", 2)],
        "MYC": [("MYC_AMP", 2)],
        "FGFR1": [("FGFR1_AMP", 2)],
        "PTEN": [("PTEN_DEL", -2)],
        "RB1": [("RB1_DEL", -2)],
    },
    "aml_target_2018_pub": {
        "FLT3": [("FLT3_AMP", 2)],
        "KIT": [("KIT_AMP", 2)],
        "WT1": [("WT1_DEL", -2)],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 17 four-cohort extension.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_for_study(config: dict, study_config: dict) -> dict:
    local = copy.deepcopy(config)
    for section in ["analysis", "mhn"]:
        overrides = study_config.get(f"{section}_overrides", {})
        if overrides:
            local.setdefault(section, {}).update(overrides)
    return local


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def infer_four_cohort_metadata(study_id: str, study_dir: Path, sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()
    if "SAMPLE_ID" not in sample_df or "PATIENT_ID" not in sample_df:
        raise ValueError(f"{study_id}: missing SAMPLE_ID/PATIENT_ID in clinical sample table")

    work = sample_df.copy()
    work["sample_id"] = work["SAMPLE_ID"].astype(str)
    work["patient_id"] = work["PATIENT_ID"].astype(str)
    work["stage"] = "unknown"
    work["sample_role"] = "unknown"
    work["time_rank"] = np.nan
    work["order_evaluable"] = False
    work["temporal_evidence"] = "not_evaluable"

    if study_id == "aml_target_2018_pub":
        suffix = work["sample_id"].str.extract(r"-(\d+)$", expand=False).fillna("")
        role_map = {
            "03": "diagnostic_peripheral_blood",
            "09": "diagnostic_bone_marrow",
            "04": "recurrent_bone_marrow",
            "40": "cell_line_or_derived",
            "41": "cell_line_or_derived",
            "42": "cell_line_or_derived",
        }
        work["sample_role"] = suffix.map(role_map).fillna("unknown_target_sample_code")
        work["stage"] = np.where(suffix.isin(["03", "09"]), "baseline", "unknown")
        work["stage"] = np.where(suffix.eq("04"), "progressed", work["stage"])
        work["time_rank"] = np.where(suffix.eq("04"), 1.0, np.where(suffix.isin(["03", "09"]), 0.0, np.nan))
        # The mutation table lacks usable recurrent AML samples, so these
        # paired diagnostic code differences are audited but not treated as
        # longitudinal dwell observations.
        work["order_evaluable"] = False
        work["temporal_evidence"] = "TARGET sample code audited; no reliable recurrent mutation pairs"

    elif study_id == "lipo_msk_2026":
        sample_type = _text_series(work, "SAMPLE_TYPE").str.lower()
        sample_class = _text_series(work, "SAMPLE_CLASS").str.lower()
        histology = _text_series(work, "TUMOR_SAMPLE_HISTOLOGY").str.lower()
        years = _numeric_series(work, "YEARS_POST_DIAGNOSIS")
        tumor = sample_class.eq("tumor") | sample_class.eq("")
        recurrence = sample_type.str.contains("recurrence|metastatic", regex=True, na=False)
        primary = sample_type.str.contains("primary", na=False) & ~recurrence
        dediff = histology.str.contains("dedifferentiated", na=False)
        work["stage"] = np.where(recurrence | dediff, "progressed", np.where(primary, "baseline", "unknown"))
        work["sample_role"] = np.where(recurrence, "local_recurrence_or_metastasis", np.where(primary, "primary", "histology_or_unknown"))
        work["time_rank"] = years
        work["order_evaluable"] = tumor & years.notna() & work["stage"].ne("unknown")
        work["temporal_evidence"] = np.where(work["order_evaluable"], "years_post_diagnosis", "not_year_evaluable")

    elif study_id == "breast_msk_2018":
        sample_type = _text_series(work, "SAMPLE_TYPE").str.lower()
        sample_site = _text_series(work, "SAMPLE_SITE").str.lower()
        time = _numeric_series(work, "NGS_SAMPLE_COLLECTION_TIME")
        progressed = sample_type.str.contains("metast", na=False)
        baseline = sample_type.str.contains("primary", na=False)
        post_treatment_primary = sample_site.str.contains("post-treatment", na=False)
        work["stage"] = np.where(progressed | post_treatment_primary, "progressed", np.where(baseline, "baseline", "unknown"))
        role = np.where(progressed, "metastasis", np.where(post_treatment_primary, "post_treatment_primary", "primary"))
        work["sample_role"] = np.where(work["stage"].eq("unknown"), "unknown", role)
        work["time_rank"] = time
        work["order_evaluable"] = time.notna() & work["stage"].ne("unknown")
        work["temporal_evidence"] = np.where(work["order_evaluable"], "ngs_collection_time", "not_time_evaluable")

    elif study_id == "mnm_washu_2016":
        suffix = pd.to_numeric(work["sample_id"].str.extract(r"-(\d+)$", expand=False), errors="coerce")
        work["time_rank"] = suffix
        work["stage"] = np.where(suffix.eq(1), "baseline", np.where(suffix.gt(1), "progressed", "unknown"))
        work["sample_role"] = np.where(suffix.eq(1), "paired_early_sample", np.where(suffix.gt(1), "paired_late_sample", "unknown"))
        work["order_evaluable"] = suffix.notna()
        work["temporal_evidence"] = np.where(work["order_evaluable"], "paired_sample_suffix", "not_suffix_evaluable")

    else:
        raise ValueError(f"{study_id}: no four-cohort temporal rule configured")

    columns = [
        "patient_id",
        "sample_id",
        "stage",
        "sample_role",
        "time_rank",
        "order_evaluable",
        "temporal_evidence",
    ]
    for optional in [
        "SAMPLE_TYPE",
        "SAMPLE_SITE",
        "YEARS_POST_DIAGNOSIS",
        "NGS_SAMPLE_COLLECTION_TIME",
        "ONCOTREE_CODE",
        "CANCER_TYPE_DETAILED",
    ]:
        if optional in work:
            columns.append(optional)
    return work[columns].copy()


def load_selected_cna_events(study_id: str, study_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    rules = STUDY_CNA_RULES.get(study_id, {})
    path = study_dir / "data_cna.txt"
    if not rules or not path.exists():
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])

    header = e17.first_data_header(path)
    if not header:
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])
    sample_ids = set(metadata["sample_id"].astype(str))
    usecols = [col for col in header if col in {"Hugo_Symbol", "Entrez_Gene_Id"} or col in sample_ids]
    if "Hugo_Symbol" not in usecols:
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])

    cna = pd.read_csv(path, sep="\t", comment="#", dtype=str, usecols=usecols, low_memory=False)
    cna["Hugo_Symbol"] = cna["Hugo_Symbol"].map(e17.clean_gene)
    cna = cna[cna["Hugo_Symbol"].isin(rules)].copy()
    if cna.empty:
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])

    sample_columns = [col for col in cna.columns if col not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    rows: list[dict[str, str]] = []
    for record in cna.to_dict(orient="records"):
        gene = str(record["Hugo_Symbol"])
        for event_name, threshold in rules.get(gene, []):
            for sample_id in sample_columns:
                value = pd.to_numeric(pd.Series([record.get(sample_id)]), errors="coerce").iloc[0]
                if not np.isfinite(value):
                    continue
                hit = value >= threshold if threshold > 0 else value <= threshold
                if hit:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "gene": event_name,
                            "alteration_type": "deep_cna" if threshold < 0 else "high_level_cna",
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])
    return pd.DataFrame(rows).drop_duplicates(["sample_id", "gene"]).reset_index(drop=True)


def load_driver_alterations(study_id: str, study_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    e17.STUDY_PRIORITY_GENES.update(STUDY_PRIORITY_EVENTS)
    for genes in STUDY_PRIORITY_EVENTS.values():
        e17.DRIVER_CANDIDATES.update(event for event in genes if not event.endswith(("_AMP", "_DEL")))

    mutations = e17.load_driver_mutations(study_id, study_dir, metadata)
    if mutations.empty:
        mutations = pd.DataFrame(columns=["sample_id", "gene"])
    mutations = mutations.assign(alteration_type="mutation")
    cna = load_selected_cna_events(study_id, study_dir, metadata)
    combined = pd.concat([mutations, cna], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["sample_id", "gene", "alteration_type"])
    return combined.drop_duplicates(["sample_id", "gene"]).reset_index(drop=True)


def process_study(study_id: str, study_config: dict, config: dict, result_root: Path) -> dict:
    local_config = config_for_study(config, study_config)
    study_dir = Path(config["data_root"]) / study_id
    tables_dir = result_root / study_id / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    sample_df = e17.read_cbio_table(study_dir / "data_clinical_sample.txt")
    metadata = infer_four_cohort_metadata(study_id, study_dir, sample_df)
    alterations = load_driver_alterations(study_id, study_dir, metadata)
    events, event_support = e17.select_events(study_id, metadata, alterations, local_config)
    selected_alterations = alterations[alterations["gene"].isin(events)].copy()
    matrix = e17.build_event_matrix(metadata, selected_alterations, events)
    occupancy = e17.build_occupancy(metadata, matrix, events)
    study_seed = int(hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    theta, fit_meta = e17.fit_or_build_theta(matrix, events, local_config, int(config["random_seed"]) + study_seed)
    scores, edges, normalizer = e17.score_external_states(occupancy, theta, events, local_config)
    timepoints = e17.aggregate_timepoint_events(metadata, matrix, events)
    dwell_predictions, dwell_persistence = e17.evaluate_rstar_dwell_persistence(
        study_id,
        metadata,
        matrix,
        timepoints,
        theta,
        events,
        local_config,
    )
    dwell, paired_dwell = e17.dwell_contrast(study_id, metadata, matrix, scores, timepoints, events)

    metadata.to_csv(tables_dir / "sample_metadata.tsv", sep="\t", index=False)
    alterations.to_csv(tables_dir / "driver_alterations_long.tsv", sep="\t", index=False)
    event_support.to_csv(tables_dir / "event_support.tsv", sep="\t", index=False)
    matrix.to_csv(tables_dir / "event_matrix.tsv", sep="\t", index=False)
    occupancy.to_csv(tables_dir / "state_occupancy.tsv", sep="\t", index=False)
    pd.DataFrame(theta, index=events, columns=events).rename_axis("target_event").to_csv(tables_dir / "theta.tsv", sep="\t")
    scores.to_csv(tables_dir / "state_scores.tsv", sep="\t", index=False)
    edges.to_csv(tables_dir / "state_edges.tsv", sep="\t", index=False)
    timepoints.to_csv(tables_dir / "ordered_timepoint_states.tsv", sep="\t", index=False)
    dwell_predictions.to_csv(tables_dir / "dwell_persistence_predictions.tsv", sep="\t", index=False)
    dwell_persistence.to_csv(tables_dir / "dwell_persistence_summary.tsv", sep="\t", index=False)
    dwell.to_csv(tables_dir / "dwell_stage_contrast.tsv", sep="\t", index=False)
    paired_dwell.to_csv(tables_dir / "paired_dwell_delta.tsv", sep="\t", index=False)
    (tables_dir / "fit_metadata.json").write_text(json.dumps(fit_meta, indent=2), encoding="utf-8")

    multi_sample = metadata.groupby("patient_id")["sample_id"].nunique()
    ordered_sample = metadata[metadata["order_evaluable"].astype(bool)].copy()
    eligible_scores = scores[scores["eligible_relobstq"].astype(bool)].copy()
    raw_pair_count = int(dwell_persistence["total_ordered_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    retained_pair_count = int(dwell_persistence["pair_qc_retained_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    ordered_pair_count = int(dwell_persistence["evaluable_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    excluded_loss_count = (
        int(dwell_persistence["pair_qc_excluded_event_loss_pairs"].iloc[0]) if not dwell_persistence.empty else 0
    )
    qc = {
        "study_id": study_id,
        "short_name": study_config["short_name"],
        "display_name": study_config["display_name"],
        "temporal_rule": study_config.get("temporal_rule", ""),
        "samples": int(metadata["sample_id"].nunique()),
        "patients": int(metadata["patient_id"].nunique()),
        "multi_sample_patients": int((multi_sample > 1).sum()),
        "order_evaluable_samples": int(ordered_sample["sample_id"].nunique()),
        "order_evaluable_patients": int(ordered_sample["patient_id"].nunique()) if not ordered_sample.empty else 0,
        "raw_ordered_pair_count": raw_pair_count,
        "pair_qc_retained_count": retained_pair_count,
        "pair_qc_excluded_loss_count": excluded_loss_count,
        "ordered_pair_count": ordered_pair_count,
        "selected_events": int(len(events)),
        "selected_event_names": ";".join(events),
        "observed_states": int(len(occupancy)),
        "eligible_states": int(scores["eligible_relobstq"].astype(bool).sum()) if "eligible_relobstq" in scores else 0,
        "top_R_star": float(eligible_scores["R_star"].replace([np.inf, -np.inf], np.nan).max()) if not eligible_scores.empty else np.nan,
        "median_R_star": float(eligible_scores["R_star"].median()) if not eligible_scores.empty else np.nan,
        "rstar_normalizer": float(normalizer),
        "backend": fit_meta["backend"],
        "fit_status": fit_meta["fit_status"],
    }
    return {
        "qc": qc,
        "scores": scores.assign(study_id=study_id, short_name=study_config["short_name"]),
        "dwell": dwell,
        "paired_dwell": paired_dwell,
        "dwell_predictions": dwell_predictions,
        "dwell_persistence": dwell_persistence,
    }


def _finite(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _fmt(value: object, digits: int = 2, missing: str = "NE") -> str:
    number = _finite(value)
    if not np.isfinite(number):
        return missing
    return f"{number:.{digits}f}"


def make_core_metric_table(dwell_persistence: pd.DataFrame, cohort_qc: pd.DataFrame) -> pd.DataFrame:
    if dwell_persistence.empty:
        return pd.DataFrame()
    rows = []
    qc_lookup = cohort_qc.set_index("study_id")
    for row in dwell_persistence.to_dict(orient="records"):
        study_id = row["study_id"]
        short_name = qc_lookup.loc[study_id, "short_name"] if study_id in qc_lookup.index else study_id
        auc_text = _fmt(row.get("auc"), 2)
        if np.isfinite(_finite(row.get("auc_ci_low"))) and np.isfinite(_finite(row.get("auc_ci_high"))):
            auc_text = f"{auc_text} [{_fmt(row.get('auc_ci_low'), 2)}, {_fmt(row.get('auc_ci_high'), 2)}]"
        delta_text = _fmt(row.get("delta_persistence_rate_high_minus_low"), 2)
        if np.isfinite(_finite(row.get("delta_persistence_ci_low"))) and np.isfinite(_finite(row.get("delta_persistence_ci_high"))):
            delta_text = f"{delta_text} [{_fmt(row.get('delta_persistence_ci_low'), 2)}, {_fmt(row.get('delta_persistence_ci_high'), 2)}]"
        rho_text = _fmt(row.get("spearman_r_minimum_dwell_interval"), 2)
        if np.isfinite(_finite(row.get("spearman_r_minimum_dwell_ci_low"))) and np.isfinite(_finite(row.get("spearman_r_minimum_dwell_ci_high"))):
            rho_text = f"{rho_text} [{_fmt(row.get('spearman_r_minimum_dwell_ci_low'), 2)}, {_fmt(row.get('spearman_r_minimum_dwell_ci_high'), 2)}]"
        rows.append(
            {
                "cohort": short_name,
                "n_P_C": f"{int(row.get('evaluable_pairs', 0))} ({int(row.get('persistent_pairs', 0))}/{int(row.get('changed_pairs', 0))})",
                "AUC_95CI": auc_text,
                "AP_lift": _fmt(row.get("average_precision_lift"), 2),
                "Delta_persist_95CI": delta_text,
                "rho_minimum_dwell_95CI": rho_text,
                "exact_state_fraction": _fmt(row.get("exact_state_score_fraction"), 2),
            }
        )
    return pd.DataFrame(rows)


def make_metric_audit(dwell_persistence: pd.DataFrame) -> pd.DataFrame:
    if dwell_persistence.empty:
        return pd.DataFrame()
    rows = []
    for row in dwell_persistence.to_dict(orient="records"):
        auc = _finite(row.get("auc"))
        delta = _finite(row.get("delta_persistence_rate_high_minus_low"))
        rho = _finite(row.get("spearman_r_minimum_dwell_interval"))
        ap_lift = _finite(row.get("average_precision_lift"))
        evaluable = int(row.get("evaluable_pairs", 0))
        persistent = int(row.get("persistent_pairs", 0))
        changed = int(row.get("changed_pairs", 0))
        positive_signals = sum(
            [
                bool(np.isfinite(auc) and auc > 0.50),
                bool(np.isfinite(ap_lift) and ap_lift > 1.00),
                bool(np.isfinite(delta) and delta > 0.00),
                bool(np.isfinite(rho) and rho > 0.00),
            ]
        )
        negative_signals = sum(
            [
                bool(np.isfinite(auc) and auc < 0.50),
                bool(np.isfinite(delta) and delta < 0.00),
                bool(np.isfinite(rho) and rho < 0.00),
            ]
        )
        if evaluable == 0:
            support = "not_evaluable_no_longitudinal_pairs"
        elif persistent == 0 or changed == 0:
            support = "underpowered_single_outcome_class"
        elif evaluable < 10:
            support = "underpowered_too_few_pairs"
        elif evaluable < 20 and positive_signals >= 3:
            support = "supportive_small_cohort"
        elif evaluable >= 20 and positive_signals >= 3 and negative_signals == 0:
            support = "supportive"
        elif evaluable >= 20 and positive_signals >= 2 and negative_signals > 0:
            support = "mixed_weak_support"
        else:
            support = "not_supportive"
        rows.append(
            {
                "study_id": row["study_id"],
                "evaluable_pairs": evaluable,
                "auc": auc,
                "average_precision_lift": ap_lift,
                "delta_persistence_rate_high_minus_low": delta,
                "spearman_r_minimum_dwell_interval": rho,
                "support_call": support,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return e17.dataframe_to_markdown(df)


def draw_table(ax: plt.Axes, table: pd.DataFrame, palette: dict[str, str]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if table.empty:
        ax.text(0.5, 0.5, "No evaluable metrics", ha="center", va="center", fontsize=8)
        return
    columns = table.columns.tolist()
    header_labels = {
        "cohort": "cohort",
        "n_P_C": "n (P/C)",
        "AUC_95CI": "AUC\n(95% CI)",
        "AP_lift": "AP\nlift",
        "Delta_persist_95CI": "Delta\npersist",
        "rho_minimum_dwell_95CI": "rho\nminimum dwell",
        "exact_state_fraction": "exact\nstate",
    }
    widths = np.array([0.14, 0.12, 0.18, 0.09, 0.20, 0.19, 0.08], dtype=float)
    widths = widths / widths.sum()
    x_edges = np.concatenate([[0.0], np.cumsum(widths)])
    top, bottom = 0.92, 0.08
    n_rows = len(table) + 1
    row_h = (top - bottom) / n_rows
    ax.add_patch(
        plt.Rectangle((0, top - row_h), 1, row_h, facecolor=palette.get("pale_yellow", "#FEEBB9"), alpha=0.55, lw=0)
    )
    for i in range(n_rows + 1):
        y = top - i * row_h
        color = "#333333" if i in {0, 1, n_rows} else "#E6E6E6"
        lw = 0.75 if i in {0, 1, n_rows} else 0.45
        ax.hlines(y, 0, 1, color=color, lw=lw)
    for x in x_edges[1:-1]:
        ax.vlines(x, bottom, top, color="#E6E6E6", lw=0.45)
    for j, col in enumerate(columns):
        x = (x_edges[j] + x_edges[j + 1]) / 2
        ha = "center"
        if j == 0:
            x = x_edges[j] + 0.012
            ha = "left"
        ax.text(
            x,
            top - row_h / 2,
            header_labels.get(col, col),
            ha=ha,
            va="center",
            fontsize=6.35,
            fontweight="bold",
            linespacing=1.05,
        )
    for i, row in enumerate(table.to_dict(orient="records"), start=1):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, top - (i + 1) * row_h), 1, row_h, facecolor="#FAFAFA", lw=0))
        for j, col in enumerate(columns):
            x = (x_edges[j] + x_edges[j + 1]) / 2
            ha = "center"
            fs = 6.2
            if j == 0:
                x = x_edges[j] + 0.012
                ha = "left"
                fs = 6.45
            value = str(row[col]).replace(" [", "\n[")
            ax.text(x, top - (i + 0.5) * row_h, value, ha=ha, va="center", fontsize=fs, linespacing=1.08)


def make_summary_figure(
    result_root: Path,
    config: dict,
    cohort_qc: pd.DataFrame,
    dwell_persistence: pd.DataFrame,
    metric_table: pd.DataFrame,
) -> None:
    figure_style.configure_matplotlib(config)
    palette = figure_style.categorical_palette(config)
    cohort_order = cohort_qc["study_id"].tolist()
    short = cohort_qc.set_index("study_id")["short_name"].to_dict()
    colors = {
        study: color
        for study, color in zip(
            cohort_order,
            [
                palette.get("lavender", "#B5AED5"),
                palette.get("sky_blue", "#B2E6FD"),
                palette.get("sage", "#B8D2CC"),
                palette.get("coral", "#E8B2A7"),
            ],
        )
    }
    persistence = dwell_persistence.set_index("study_id") if not dwell_persistence.empty else pd.DataFrame()
    fig = plt.figure(figsize=tuple(config["plot"].get("summary_figure_size", [8.4, 6.2])))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.05],
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.94,
        wspace=0.30,
        hspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    y = np.arange(len(cohort_order))
    for yi, study in zip(y, cohort_order):
        color = colors[study]
        if study not in persistence.index:
            ax_a.text(0.03, yi, "NE", ha="left", va="center", fontsize=6.5, color="#4E5A5E")
            continue
        row = persistence.loc[study]
        low = _finite(row.get("low_rstar_persistence_rate"))
        high = _finite(row.get("high_rstar_persistence_rate"))
        if np.isfinite(low) and np.isfinite(high):
            ax_a.hlines(yi, low, high, color="#4E5A5E", lw=1.0, zorder=1)
            ax_a.scatter(low, yi, s=24, facecolor="white", edgecolor=color, linewidth=0.9, zorder=2)
            ax_a.scatter(high, yi, s=30, facecolor=color, edgecolor="#263238", linewidth=0.55, zorder=3)
            delta = _finite(row.get("delta_persistence_rate_high_minus_low"))
            ax_a.text(1.02, yi, f"d={delta:+.2f}" if np.isfinite(delta) else "d=NE", ha="left", va="center", fontsize=6.1)
        else:
            ax_a.text(0.03, yi, "NE", ha="left", va="center", fontsize=6.5, color="#4E5A5E")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([short[s] for s in cohort_order])
    ax_a.set_xlim(-0.03, 1.20)
    ax_a.set_ylim(-0.55, len(cohort_order) - 0.45)
    ax_a.set_xticks([0, 0.5, 1.0])
    ax_a.set_xlabel("genotype-persistence rate")
    ax_a.set_title(r"top $R^*$ states persist more often", loc="left", pad=5)
    ax_a.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#4E5A5E", markersize=4.1, label="bottom third"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get("sky_blue", "#B2E6FD"), markeredgecolor="#263238", markersize=4.1, label="top third"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=6.2,
        handletextpad=0.25,
    )
    ax_a.set_box_aspect(1)
    e17.clean_axis(ax_a)
    e17.panel_label(ax_a, "a")

    for yi, study in zip(y, cohort_order):
        color = colors[study]
        if study not in persistence.index:
            ax_b.text(-0.18, yi, "NE", ha="left", va="center", fontsize=6.3, color="#4E5A5E")
            continue
        row = persistence.loc[study]
        rho = _finite(row.get("spearman_r_minimum_dwell_interval"))
        low = _finite(row.get("spearman_r_minimum_dwell_ci_low"))
        high = _finite(row.get("spearman_r_minimum_dwell_ci_high"))
        n = int(row.get("evaluable_pairs", 0))
        if np.isfinite(rho):
            if np.isfinite(low) and np.isfinite(high):
                ax_b.hlines(yi, low, high, color="#4E5A5E", lw=0.9, zorder=1)
                ax_b.vlines([low, high], yi - 0.07, yi + 0.07, color="#4E5A5E", lw=0.7, zorder=1)
            ax_b.scatter(rho, yi, s=24 + math.sqrt(max(n, 1)) * 2.2, color=color, edgecolor="#263238", linewidth=0.55, zorder=2)
            ax_b.text(0.68, yi + 0.12, f"rho={rho:.2f}, n={n}", fontsize=5.9, ha="right", va="center")
        else:
            ax_b.text(-0.18, yi, "NE", ha="left", va="center", fontsize=6.3, color="#4E5A5E")
    ax_b.axvline(0, color="#333333", lw=0.7, ls=":")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([short[s] for s in cohort_order])
    ax_b.set_xlim(-0.22, 0.72)
    ax_b.set_ylim(-0.55, len(cohort_order) - 0.45)
    ax_b.set_xlabel(r"Spearman $\rho$: $R^*$ vs minimum dwell")
    ax_b.set_title(r"$R^*$ tracks observed dwell proxy", loc="left", pad=5)
    ax_b.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax_b.set_box_aspect(1)
    e17.clean_axis(ax_b)
    e17.panel_label(ax_b, "b")

    draw_table(ax_c, metric_table, palette)
    ax_c.set_title("core longitudinal validation metrics", loc="left", pad=5)
    e17.panel_label(ax_c, "c")

    figure_style.save_figure_panels(fig, result_root / "figures" / "Figure_E17_four_cohort_longitudinal_validation", config)


def write_reviews(
    result_root: Path,
    config: dict,
    cohort_qc: pd.DataFrame,
    dwell_persistence: pd.DataFrame,
    metric_table: pd.DataFrame,
    metric_audit: pd.DataFrame,
) -> None:
    sources = figure_style.design_sources_markdown(config)
    rules = figure_style.design_rules_markdown(config)
    patterns = figure_style.design_patterns_markdown(config)
    audit_text = dataframe_to_markdown(metric_audit)
    qc_text = dataframe_to_markdown(
        cohort_qc[
            [
                "study_id",
                "short_name",
                "samples",
                "patients",
                "multi_sample_patients",
                "order_evaluable_samples",
                "order_evaluable_patients",
                "raw_ordered_pair_count",
                "pair_qc_retained_count",
                "ordered_pair_count",
                "selected_events",
                "backend",
            ]
        ]
    )
    table_text = dataframe_to_markdown(metric_table)

    (result_root / "experiment_17_four_cohort_protocol_audit.md").write_text(
        "\n".join(
            [
                "# Experiment 17 four-cohort extension protocol audit",
                "",
                "## Focused Question",
                "",
                "This extension tests whether Rel-ObsTQ-MHN R* scores predict observed relative dwell/persistence in four newly collected public longitudinal or quasi-longitudinal cohorts.",
                "",
                "## Cohort Rules",
                "",
                "- AML-TARGET: sample-code audit retained for feasibility; not treated as longitudinal validation because mutation records are almost entirely diagnostic -03/-09 samples.",
                "- LIPO-MSK: tumor samples ordered by YEARS_POST_DIAGNOSIS; cell lines and non-year-evaluable samples are excluded from ordered-pair validation.",
                "- BRCA-MSK: samples ordered by NGS_SAMPLE_COLLECTION_TIME; primary/metastatic and post-treatment primary categories define baseline/progressed labels.",
                "- MNM-WashU: paired sample suffix -1/-2 defines early/late order; this supports pair persistence but has limited real-time interval resolution.",
                "",
                "## Figure Design Sources",
                "",
                sources,
                "",
                "## Shared Design Rules Applied",
                "",
                rules,
                "",
                "## Reused Design Patterns",
                "",
                patterns,
                "",
                "## Cohort QC",
                "",
                qc_text,
            ]
        ),
        encoding="utf-8",
    )

    (result_root / "experiment_17_four_cohort_scientific_review.md").write_text(
        "\n".join(
            [
                "# Experiment 17 four-cohort extension scientific review",
                "",
                "## Core Metric Table",
                "",
                table_text,
                "",
                "## Objective Metric Audit",
                "",
                audit_text,
                "",
                "## Interpretation",
                "",
                "- A clearly supportive cohort should have both persistent and changed outcomes, at least 10 retained evaluable pairs, and directionally positive evidence in most core metrics.",
                "- AUC > 0.50, AP lift > 1.00, top-bottom persistence delta > 0, and positive R*-minimum dwell correlation are treated as separate directional signals, not as interchangeable proof.",
                "- AML-TARGET is an audited negative-feasibility result for this specific longitudinal validation, not evidence against the model.",
                "- MNM-WashU is useful for pairwise persistence but its suffix-based time axis limits continuous dwell-gradient interpretation.",
                "- BRCA-MSK is the largest external check but should be interpreted as mixed/weak support if discrimination and top-bottom persistence do not agree.",
                "- LIPO-MSK has clinically meaningful timing metadata but too few scoreable pairs after monotone QC to support a formal conclusion.",
            ]
        ),
        encoding="utf-8",
    )

    (result_root / "experiment_17_four_cohort_chinese_summary.md").write_text(
        "\n".join(
            [
                "# 实验17四队列扩展：中文总结",
                "",
                "## 实验目的",
                "",
                "这组实验不是再证明MHN能重建拓扑，而是直接检验我们的创新点：模型给出的 R* 是否真的对应“状态更容易停留/保持更久”。",
                "",
                "## 核心结果表",
                "",
                table_text,
                "",
                "## 客观评价",
                "",
                "AML-TARGET 缺少可靠复发突变配对，因此只作为数据可行性审查，不作为成功或失败的模型验证。",
                "",
                "LIPO-MSK 有诊断后年份信息，但严格QC后只有很少可评分配对，只能说明数据潜力，不能作为正式阳性证据。",
                "",
                "BRCA-MSK 样本量最大，但结果是混合的：AP lift 和 dwell rho 略为正向，AUC 与 top-bottom 差值没有支持。因此它只能提供弱的、方向不完全一致的真实队列证据。",
                "",
                "MNM-WashU 的结果最正向：AUC、top-bottom差值和 dwell rho 都支持 R* 与状态保持/停留更久有关，但样本规模较小，适合作为外部补充证据而不是单独决定性证据。",
                "",
                "这类真实纵向队列通常样本少、取样混杂、治疗干预强，因此不能要求每个指标都显著；更重要的是方向是否一致、是否跨队列复现。",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = read_yaml(Path(args.config))
    result_root = Path(config["result_root"])
    (result_root / "figures").mkdir(parents=True, exist_ok=True)
    (result_root / "tables").mkdir(parents=True, exist_ok=True)
    figure_style.configure_matplotlib(config)

    e17.STUDY_PRIORITY_GENES.update(STUDY_PRIORITY_EVENTS)
    outputs = {}
    for study_id, study_config in config["studies"].items():
        print(f"Processing {study_id} ...", flush=True)
        outputs[study_id] = process_study(study_id, study_config, config, result_root)

    cohort_qc = pd.DataFrame([value["qc"] for value in outputs.values()])
    cohort_qc.to_csv(result_root / "tables" / "cohort_qc.tsv", sep="\t", index=False)
    scores_all = pd.concat([value["scores"] for value in outputs.values()], ignore_index=True)
    scores_all.to_csv(result_root / "tables" / "state_scores_all.tsv", sep="\t", index=False)
    dwell_predictions = pd.concat(
        [value["dwell_predictions"] for value in outputs.values() if not value["dwell_predictions"].empty],
        ignore_index=True,
    )
    dwell_predictions.to_csv(result_root / "tables" / "dwell_persistence_predictions_all.tsv", sep="\t", index=False)
    dwell_persistence = pd.concat([value["dwell_persistence"] for value in outputs.values()], ignore_index=True)
    dwell_persistence.to_csv(result_root / "tables" / "dwell_persistence_summary_all.tsv", sep="\t", index=False)
    paired_dwell_frames = [value["paired_dwell"] for value in outputs.values() if not value["paired_dwell"].empty]
    paired_dwell = pd.concat(paired_dwell_frames, ignore_index=True) if paired_dwell_frames else pd.DataFrame()
    paired_dwell.to_csv(result_root / "tables" / "paired_dwell_delta_all.tsv", sep="\t", index=False)

    metric_table = make_core_metric_table(dwell_persistence, cohort_qc)
    metric_table.to_csv(result_root / "tables" / "core_metric_table.tsv", sep="\t", index=False)
    metric_audit = make_metric_audit(dwell_persistence)
    metric_audit.to_csv(result_root / "tables" / "metric_audit.tsv", sep="\t", index=False)
    make_summary_figure(result_root, config, cohort_qc, dwell_persistence, metric_table)

    figure_audits = [figure_style.audit_rendered_png(path, config) for path in sorted((result_root / "figures").glob("*.png"))]
    pd.DataFrame(figure_audits).to_csv(result_root / "tables" / "figure_render_audit.tsv", sep="\t", index=False)
    write_reviews(result_root, config, cohort_qc, dwell_persistence, metric_table, metric_audit)
    print(f"Done. Results written to {result_root}", flush=True)


if __name__ == "__main__":
    main()
