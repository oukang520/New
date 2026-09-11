"""Run experiments 1 and 2 for the selected Rel-ObsTQ-MHN cohorts.

Experiment 1:
    Data preparation, cancer-specific driver panel construction, event matrices,
    state tables and QC metrics.

Experiment 2:
    Stage/progression definition sensitivity before MHN training. This builds
    clinical, metastasis, mutation-burden, and pathway-burden schemes and
    quantifies their distributions, agreement, and state-space sparsity.

MHN-derived progression scores and R* ranking comparisons are intentionally
left as pending outputs because they require later MHN training.
"""

from __future__ import annotations
from .parameters import config_path as bundled_config_path
import argparse
import json
import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency

STAGE_ORDER = ["early", "local_advanced", "primary", "metastatic", "unknown"]
PASSENGER_EXACT = {
    "TTN",
    "MUC16",
    "MUC4",
    "MUC2",
    "MUC19",
    "OBSCN",
    "SYNE1",
    "HMCN1",
    "XIRP2",
    "CSMD1",
    "CSMD2",
    "CSMD3",
    "RYR1",
    "RYR2",
    "RYR3",
}
PASSENGER_PREFIXES = ("DNAH", "MUC", "CSMD", "RYR", "ANKRD")


def setup_logging(root: Path) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "experiments_01_02.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def cohort_short_label(dataset_name: str) -> str:
    if dataset_name.startswith("AACR_"):
        return dataset_name.replace("AACR_", "", 1)
    return dataset_name


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        input_dir / "analysis_metadata.csv", dtype=str, low_memory=False
    )
    mutations = pd.read_csv(
        input_dir / "mutations_long.csv", dtype=str, low_memory=False
    )
    metadata["analysis_id"] = metadata["analysis_id"].astype(str)
    mutations["analysis_id"] = mutations["analysis_id"].astype(str)
    mutations["gene"] = mutations["gene"].fillna("").astype(str).str.strip().str.upper()
    mutations = mutations[
        (mutations["gene"] != "")
        & mutations["analysis_id"].isin(set(metadata["analysis_id"]))
    ]
    mutations = mutations.drop_duplicates(["analysis_id", "gene"])
    return (metadata, mutations)


def is_likely_passenger(gene: str) -> bool:
    if gene in PASSENGER_EXACT:
        return True
    return any((gene.startswith(prefix) for prefix in PASSENGER_PREFIXES))


def pathway_lookup(pathways: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for pathway, genes in pathways.items():
        for gene in genes:
            lookup.setdefault(str(gene).upper(), pathway)
    return lookup


def construct_event_panels(
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    dataset_config: dict,
    panel_sizes: list[int],
    minimum_event_samples: int,
) -> tuple[dict[int, list[str]], pd.DataFrame]:
    n = metadata["analysis_id"].nunique()
    support_floor = max(minimum_event_samples, int(math.ceil(0.01 * n)))
    support = (
        mutations.groupby("gene")["analysis_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("sample_count")
    )
    frequency = support / max(n, 1)
    priority = [str(gene).upper() for gene in dataset_config.get("driver_priority", [])]
    priority_rank = {gene: idx + 1 for idx, gene in enumerate(priority)}
    pathways = pathway_lookup(dataset_config.get("pathways", {}))
    selected_rank: list[str] = []
    for gene in priority:
        if gene in support.index and int(support[gene]) >= support_floor:
            selected_rank.append(gene)
    for gene in support.index:
        if gene in selected_rank or int(support[gene]) < support_floor:
            continue
        if is_likely_passenger(gene):
            continue
        selected_rank.append(gene)
    for gene in support.index:
        if gene not in selected_rank and int(support[gene]) >= support_floor:
            selected_rank.append(gene)
    max_panel = max(panel_sizes)
    selected_rank = selected_rank[:max_panel]
    panels = {
        size: selected_rank[: min(size, len(selected_rank))] for size in panel_sizes
    }
    rows = []
    for gene in support.index:
        row = {
            "event": gene,
            "sample_count": int(support[gene]),
            "frequency": round(float(frequency[gene]), 6),
            "is_curated_driver": gene in priority_rank,
            "driver_priority_rank": priority_rank.get(gene, ""),
            "pathway": pathways.get(gene, "Other"),
            "likely_passenger_or_size_related": is_likely_passenger(gene),
            "minimum_support_required": support_floor,
            "panel_rank": (
                selected_rank.index(gene) + 1 if gene in selected_rank else ""
            ),
            "selection_reason": (
                "curated_cancer_driver"
                if gene in priority_rank
                else "frequency_fill" if gene in selected_rank else "not_selected"
            ),
        }
        for size in panel_sizes:
            row[f"selected_p{size}"] = gene in panels[size]
        rows.append(row)
    return (panels, pd.DataFrame(rows))


def build_matrix(
    metadata: pd.DataFrame, mutations: pd.DataFrame, events: list[str]
) -> pd.DataFrame:
    ids = metadata["analysis_id"].drop_duplicates().astype(str)
    work = mutations[mutations["gene"].isin(events)][
        ["analysis_id", "gene"]
    ].drop_duplicates()
    work["_value"] = 1
    if work.empty:
        matrix = pd.DataFrame(0, index=ids, columns=events)
    else:
        matrix = work.pivot_table(
            index="analysis_id",
            columns="gene",
            values="_value",
            aggfunc="max",
            fill_value=0,
        )
        matrix = matrix.reindex(index=ids, columns=events, fill_value=0).astype(int)
    matrix.index.name = "analysis_id"
    return matrix.reset_index()


def normalize_clinical_stage(series: pd.Series) -> pd.Series:
    values = series.fillna("unknown").astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[values.eq("early")] = "early"
    out[values.eq("local_advanced")] = "local_advanced"
    out[values.eq("primary")] = "primary"
    out[values.eq("metastatic")] = "metastatic"
    return out


def normalize_metastasis(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[values.str.contains("metast|distant", regex=True)] = "metastatic"
    out[values.str.contains("primary tumour|primary tumor|primary", regex=True)] = (
        "primary"
    )
    return out


def burden_groups(series: pd.Series, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    try:
        binned = pd.qcut(values, q=3, labels=False, duplicates="drop")
        if binned.nunique(dropna=True) >= 3:
            mapping = {0: f"{prefix}_low", 1: f"{prefix}_mid", 2: f"{prefix}_high"}
            return binned.map(mapping).fillna(f"{prefix}_unknown")
    except ValueError:
        pass
    positive = values[values > 0]
    median_positive = float(positive.median()) if not positive.empty else 0
    out = pd.Series(f"{prefix}_low", index=values.index, dtype=object)
    out[(values > 0) & (values <= median_positive)] = f"{prefix}_mid"
    out[values > median_positive] = f"{prefix}_high"
    return out


def build_state_table(
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    stage_assignment: pd.Series,
    events: list[str],
    min_state_count: int,
    stage_column_name: str = "stage_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = metadata.merge(matrix, on="analysis_id", how="left")
    work[stage_column_name] = stage_assignment.reindex(metadata.index).values
    for event in events:
        work[event] = pd.to_numeric(work[event], errors="coerce").fillna(0).astype(int)
    work["event_count"] = work[events].sum(axis=1).astype(int)
    active = work[events].apply(
        lambda row: "+".join(sorted(row.index[row.astype(int).eq(1)].tolist())) or "WT",
        axis=1,
    )
    work["genotype_signature"] = active
    work["state_id"] = work[stage_column_name].astype(str) + "::" + active
    counts = work["state_id"].value_counts()
    work["state_count"] = work["state_id"].map(counts).astype(int)
    work["state_count_flag"] = np.where(
        work["state_count"] >= min_state_count, "valid_state", "rare_state"
    )
    work["usable_for_mhn"] = True
    work["usable_for_relobstq"] = work[stage_column_name].ne("unknown") & work[
        "state_count_flag"
    ].eq("valid_state")
    occupancy = (
        work.groupby(
            [stage_column_name, "genotype_signature", "state_id", "state_count_flag"],
            dropna=False,
        )
        .size()
        .reset_index(name="state_count")
        .sort_values(["state_count", "state_id"], ascending=[False, True])
    )
    occupancy["occupancy_fraction"] = occupancy["state_count"] / len(work)
    columns = [
        "analysis_id",
        "patient_id",
        "sample_id",
        "dataset_name",
        "cancer_code",
        "cancer_type",
        "cancer_type_detailed",
        "stage_raw",
        stage_column_name,
        "metastasis_status",
        "genotype_signature",
        "event_count",
        "state_id",
        "state_count",
        "state_count_flag",
        "usable_for_mhn",
        "usable_for_relobstq",
        "survival_time",
        "survival_event",
        "age",
        "sex",
    ]
    return (work[columns], occupancy)


def stage_event_frequency(
    matrix: pd.DataFrame, assignments: pd.Series, events: list[str]
) -> pd.DataFrame:
    work = matrix.copy()
    work["stage"] = assignments.values
    result = work.groupby("stage")[events].mean()
    order = [stage for stage in STAGE_ORDER if stage in result.index]
    remaining = [stage for stage in result.index if stage not in order]
    return result.reindex(order + sorted(remaining))


def cramer_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.empty or min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def event_pathway_burden(
    mutations: pd.DataFrame, metadata: pd.DataFrame, pathways: dict[str, list[str]]
) -> pd.Series:
    gene_to_pathways: dict[str, set[str]] = {}
    for pathway, genes in pathways.items():
        for gene in genes:
            gene_to_pathways.setdefault(str(gene).upper(), set()).add(pathway)
    selected = mutations[mutations["gene"].isin(gene_to_pathways)].copy()
    if selected.empty:
        return pd.Series(0, index=metadata.index)
    selected["pathway"] = selected["gene"].map(
        lambda gene: sorted(gene_to_pathways.get(gene, {"Other"}))[0]
    )
    counts = (
        selected.drop_duplicates(["analysis_id", "pathway"])
        .groupby("analysis_id")["pathway"]
        .nunique()
    )
    return metadata["analysis_id"].map(counts).fillna(0).astype(int)


def inclusion_count(inclusion: pd.DataFrame, keywords: list[str]) -> int:
    """Return the first inclusion count whose step contains all keywords."""
    lowered = inclusion["step"].astype(str).str.lower()
    mask = pd.Series(True, index=inclusion.index)
    for keyword in keywords:
        mask &= lowered.str.contains(keyword.lower(), regex=False)
    if not mask.any():
        raise ValueError(f"Missing inclusion step containing {keywords}")
    return int(inclusion.loc[mask, "n"].iloc[0])


def run_dataset(
    dataset_name: str,
    input_dir: Path,
    dataset_config: dict,
    global_config: dict,
    result_root: Path,
) -> dict:
    logging.info("Starting Experiments 1-2 for %s", dataset_name)
    display_name = dataset_config["display_name"]
    metadata, mutations = read_inputs(input_dir)
    metadata = metadata.sort_values("analysis_id").reset_index(drop=True)
    mutations = mutations[mutations["analysis_id"].isin(set(metadata["analysis_id"]))]
    exp1_root = result_root / dataset_name / "experiment_01_data_preparation"
    exp2_root = result_root / dataset_name / "experiment_02_stage_sensitivity"
    exp1_tables = exp1_root / "tables"
    exp2_tables = exp2_root / "tables"
    for path in [exp1_tables, exp2_tables]:
        path.mkdir(parents=True, exist_ok=True)
    panel_sizes = [int(x) for x in global_config["sensitivity_panel_sizes"]]
    panels, panel_table = construct_event_panels(
        metadata,
        mutations,
        dataset_config,
        panel_sizes,
        int(global_config["minimum_event_samples"]),
    )
    panel_table.to_csv(exp1_tables / "event_panel_candidates.csv", index=False)
    clinical_stage = normalize_clinical_stage(metadata["stage_group"])
    panel_metrics = []
    matrices: dict[int, pd.DataFrame] = {}
    occupancies: dict[int, pd.DataFrame] = {}
    state_tables: dict[int, pd.DataFrame] = {}
    for size in panel_sizes:
        events = panels[size]
        matrix = build_matrix(metadata, mutations, events)
        state_table, occupancy = build_state_table(
            metadata,
            matrix,
            clinical_stage,
            events,
            int(global_config["minimum_state_count"]),
        )
        matrices[size] = matrix
        state_tables[size] = state_table
        occupancies[size] = occupancy
        matrix.to_csv(exp1_tables / f"event_matrix_p{size}.csv", index=False)
        matrix[events].to_csv(
            exp1_tables / f"mhn_training_matrix_p{size}.csv", index=False
        )
        pd.DataFrame({"panel_rank": range(1, len(events) + 1), "event": events}).to_csv(
            exp1_tables / f"event_panel_p{size}.csv", index=False
        )
        state_table.to_csv(exp1_tables / f"state_table_p{size}.csv", index=False)
        occupancy.to_csv(exp1_tables / f"state_occupancy_p{size}.csv", index=False)
        panel_metrics.append(
            {
                "dataset_name": dataset_name,
                "panel_size": size,
                "events_retained": len(events),
                "unique_states": int(len(occupancy)),
                "valid_states": int(
                    occupancy["state_count_flag"].eq("valid_state").sum()
                ),
                "rare_states": int(
                    occupancy["state_count_flag"].eq("rare_state").sum()
                ),
                "samples_in_valid_states": int(
                    state_table["state_count_flag"].eq("valid_state").sum()
                ),
                "valid_unit_fraction": float(
                    state_table["state_count_flag"].eq("valid_state").mean()
                ),
                "zero_event_fraction": float(state_table["event_count"].eq(0).mean()),
            }
        )
    panel_metrics_df = pd.DataFrame(panel_metrics)
    panel_metrics_df.to_csv(exp1_tables / "panel_sensitivity_metrics.csv", index=False)
    primary_size = int(global_config["primary_panel_size"])
    primary_events = panels[primary_size]
    primary_matrix = matrices[primary_size]
    primary_state = state_tables[primary_size]
    primary_occupancy = occupancies[primary_size]
    clinical_clean = metadata.copy()
    clinical_clean["stage_group_experiment1"] = clinical_stage
    clinical_clean.to_csv(exp1_tables / "clinical_clean.csv", index=False)
    row_map = primary_state[
        [
            "analysis_id",
            "patient_id",
            "sample_id",
            "stage_group",
            "genotype_signature",
            "event_count",
            "state_id",
            "state_count_flag",
            "usable_for_mhn",
            "usable_for_relobstq",
        ]
    ].copy()
    row_map.insert(0, "row_index", range(len(row_map)))
    row_map.to_csv(exp1_tables / "mhn_row_index_map_p15.csv", index=False)
    stage_freq = stage_event_frequency(primary_matrix, clinical_stage, primary_events)
    stage_freq.to_csv(exp1_tables / "stage_event_frequency_p15.csv")
    sample_event_counts = primary_state[
        ["analysis_id", "patient_id", "sample_id", "stage_group", "event_count"]
    ]
    sample_event_counts.to_csv(exp1_tables / "sample_event_counts_p15.csv", index=False)
    inclusion = pd.DataFrame(
        [
            {"step": "All tumor analysis units", "n": len(metadata)},
            {
                "step": "With ≥1 functional mutation",
                "n": mutations["analysis_id"].nunique(),
            },
            {
                "step": "Known clinical/progression state",
                "n": int(clinical_stage.ne("unknown").sum()),
            },
            {
                "step": "With ≥1 p15 event",
                "n": int(primary_state["event_count"].gt(0).sum()),
            },
            {
                "step": "In valid p15 states",
                "n": int(primary_state["state_count_flag"].eq("valid_state").sum()),
            },
        ]
    )
    inclusion.to_csv(exp1_tables / "sample_inclusion_summary.csv", index=False)
    selected_panel = panel_table[panel_table["selected_p15"]].sort_values("panel_rank")
    selected_panel.to_csv(exp1_tables / "main_event_panel_p15.csv", index=False)
    exp1_metrics = {
        "dataset_name": dataset_name,
        "analysis_units": int(len(metadata)),
        "unique_patients": int(metadata["patient_id"].nunique()),
        "mutated_units": int(mutations["analysis_id"].nunique()),
        "functional_mutation_rows": int(len(mutations)),
        "main_panel_events": int(len(primary_events)),
        "known_stage_fraction": float(clinical_stage.ne("unknown").mean()),
        "zero_event_fraction": float(primary_state["event_count"].eq(0).mean()),
        "unique_states": int(len(primary_occupancy)),
        "valid_states": int(
            primary_occupancy["state_count_flag"].eq("valid_state").sum()
        ),
        "samples_in_valid_states": int(
            primary_state["state_count_flag"].eq("valid_state").sum()
        ),
        "mhn_format_binary": bool(
            primary_matrix[primary_events].isin([0, 1]).all().all()
        ),
        "mhn_rows_match_metadata": bool(len(primary_matrix) == len(metadata)),
    }
    pd.DataFrame([exp1_metrics]).to_csv(
        exp1_tables / "experiment_01_metrics.csv", index=False
    )
    mutation_burden = primary_state["event_count"].astype(int)
    pathway_count = event_pathway_burden(
        mutations, metadata, dataset_config.get("pathways", {})
    )
    assignments = pd.DataFrame(
        {
            "analysis_id": metadata["analysis_id"],
            "clinical_stage": clinical_stage,
            "metastasis_status": normalize_metastasis(metadata["metastasis_status"]),
            "mutation_burden": burden_groups(mutation_burden, "burden"),
            "pathway_burden": burden_groups(pathway_count, "pathway"),
            "mhn_progression_score": "pending_mhn_training",
            "p15_event_count": mutation_burden,
            "pathway_count": pathway_count,
        }
    )
    assignments.to_csv(exp2_tables / "stage_scheme_assignments.csv", index=False)
    schemes = [
        "clinical_stage",
        "metastasis_status",
        "mutation_burden",
        "pathway_burden",
    ]
    summary_rows = []
    state_metric_rows = []
    event_heatmaps: dict[str, pd.DataFrame] = {}
    scheme_state_tables: dict[str, pd.DataFrame] = {}
    for scheme in schemes:
        counts = assignments[scheme].value_counts(dropna=False)
        for group, count in counts.items():
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme": scheme,
                    "group": group,
                    "n": int(count),
                    "fraction": float(count / len(assignments)),
                }
            )
        state_table, occupancy = build_state_table(
            metadata,
            primary_matrix,
            assignments[scheme],
            primary_events,
            int(global_config["minimum_state_count"]),
            stage_column_name=scheme,
        )
        scheme_state_tables[scheme] = state_table
        state_table.to_csv(exp2_tables / f"state_table_{scheme}.csv", index=False)
        occupancy.to_csv(exp2_tables / f"state_occupancy_{scheme}.csv", index=False)
        state_metric_rows.append(
            {
                "dataset_name": dataset_name,
                "scheme": scheme,
                "unique_groups": int(assignments[scheme].nunique()),
                "unknown_fraction": float(assignments[scheme].eq("unknown").mean()),
                "unique_states": int(len(occupancy)),
                "valid_states": int(
                    occupancy["state_count_flag"].eq("valid_state").sum()
                ),
                "rare_states": int(
                    occupancy["state_count_flag"].eq("rare_state").sum()
                ),
                "valid_unit_fraction": float(
                    state_table["state_count_flag"].eq("valid_state").mean()
                ),
            }
        )
        event_heatmaps[scheme] = stage_event_frequency(
            primary_matrix, assignments[scheme], primary_events
        )
        event_heatmaps[scheme].to_csv(exp2_tables / f"event_frequency_{scheme}.csv")
    scheme_summary = pd.DataFrame(summary_rows)
    state_metrics = pd.DataFrame(state_metric_rows)
    scheme_summary.to_csv(exp2_tables / "stage_scheme_summary.csv", index=False)
    state_metrics.to_csv(exp2_tables / "stage_scheme_state_metrics.csv", index=False)
    agreement_rows = []
    for i, scheme_a in enumerate(schemes):
        for scheme_b in schemes[i + 1 :]:
            valid = assignments[scheme_a].ne("unknown") & assignments[scheme_b].ne(
                "unknown"
            )
            agreement_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme_a": scheme_a,
                    "scheme_b": scheme_b,
                    "n_compared": int(valid.sum()),
                    "cramers_v": cramer_v(
                        assignments.loc[valid, scheme_a],
                        assignments.loc[valid, scheme_b],
                    ),
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(exp2_tables / "stage_scheme_pairwise_agreement.csv", index=False)
    pending_rows = []
    for i, scheme_a in enumerate(schemes + ["mhn_progression_score"]):
        for scheme_b in (schemes + ["mhn_progression_score"])[i + 1 :]:
            pending_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme_a": scheme_a,
                    "scheme_b": scheme_b,
                    "rstar_spearman": "",
                    "top10_overlap": "",
                    "status": "pending_mhn_training_and_rstar",
                }
            )
    pd.DataFrame(pending_rows).to_csv(
        exp2_tables / "rstar_stage_sensitivity_pending.csv", index=False
    )
    checks = {
        "main_panel_has_15_events": len(primary_events) == 15,
        "nested_panels": all(
            (
                set(panels[a]).issubset(set(panels[b]))
                for a, b in zip(panel_sizes[:-1], panel_sizes[1:])
            )
        ),
        "binary_mhn_matrix": bool(
            primary_matrix[primary_events].isin([0, 1]).all().all()
        ),
        "row_alignment": len(primary_matrix) == len(metadata) == len(primary_state),
        "no_duplicate_analysis_ids": metadata["analysis_id"].is_unique,
        "stage_assignments_complete": len(assignments) == len(metadata),
    }
    return {
        **exp1_metrics,
        "clinical_groups": int(assignments["clinical_stage"].nunique()),
        "metastasis_groups": int(assignments["metastasis_status"].nunique()),
        "mutation_burden_groups": int(assignments["mutation_burden"].nunique()),
        "pathway_burden_groups": int(assignments["pathway_burden"].nunique()),
        "all_validation_checks_passed": bool(all(checks.values())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Rel-ObsTQ-MHN Experiments 1 and 2."
    )
    parser.add_argument(
        "--experiment-config",
        default=str(bundled_config_path("experiments_01_02.yaml")),
    )
    parser.add_argument(
        "--dataset-config",
        default=str(bundled_config_path("selected_experiment_datasets.yaml")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(".").resolve()
    setup_logging(project_root)
    with Path(args.experiment_config).open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    with Path(args.dataset_config).open("r", encoding="utf-8") as handle:
        selection_config = yaml.safe_load(handle)
    result_root = Path(experiment_config["experiment_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    results = []
    dataset_names = [
        entry["dataset_name"] for entry in selection_config["included_datasets"]
    ]
    for entry in selection_config["included_datasets"]:
        dataset_name = entry["dataset_name"]
        input_dir = Path(entry["input_dir"]).resolve()
        dataset_config = experiment_config["datasets"][dataset_name]
        results.append(
            run_dataset(
                dataset_name, input_dir, dataset_config, experiment_config, result_root
            )
        )
        print(f"Completed Experiments 1-2: {dataset_name}")
    pd.DataFrame(results).to_csv(
        result_root / "experiments_01_02_summary.csv", index=False
    )
    print(f"Results written to {result_root}")


if __name__ == "__main__":
    main()
