"""Run Experiment 4: estimate relative state inflow F_hat.

The primary analysis uses same-stage, one-event genotype predecessors. Stage
bridges, two-event predecessors, and occupancy smoothing are sensitivity rules
and are never mixed into the primary result.
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
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 4.")
    parser.add_argument(
        "--config", default=str(bundled_config_path("experiment_04.yaml"))
    )
    parser.add_argument(
        "--dataset-config",
        default=str(bundled_config_path("selected_experiment_datasets.yaml")),
    )
    return parser.parse_args()


def setup_logging(root: Path) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "logs" / "experiment_04.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def genotype_signature(vector: np.ndarray, events: list[str]) -> str:
    present = [events[i] for i, value in enumerate(vector) if int(value) == 1]
    return "+".join(present) if present else "WT"


def genotype_vector(signature: str, events: list[str]) -> np.ndarray:
    present = set() if signature == "WT" else set(signature.split("+"))
    return np.array([int(event in present) for event in events], dtype=np.int32)


def compact_state(state: str, max_events: int = 3) -> str:
    stage, genotype = state.split("::", 1)
    if genotype == "WT":
        return f"{stage}: WT"
    events = genotype.split("+")
    if len(events) > max_events:
        genotype = "+".join(events[:max_events]) + "+..."
    return f"{stage}: {genotype}"


def clean_stage(
    dataset: str, state_table: pd.DataFrame, dataset_config: dict
) -> tuple[pd.Series, pd.Series]:
    stage = state_table["stage_group"].fillna("unknown").astype(str).str.lower()
    status = state_table["metastasis_status"].fillna("").astype(str).str.lower()
    raw = state_table["stage_raw"].fillna("").astype(str).str.upper()
    reason = pd.Series("included", index=state_table.index, dtype=object)
    if dataset.startswith("AACR_"):
        clean = stage.where(stage.isin(["primary", "metastatic"]), "excluded")
        reason[clean.eq("excluded")] = "unknown_or_nonordered_stage"
        return (clean, reason)
    excluded_keywords = [
        str(x).lower() for x in dataset_config.get("exclude_specimen_keywords", [])
    ]
    model_system = pd.Series(False, index=state_table.index)
    for keyword in excluded_keywords:
        model_system |= status.str.contains(keyword, regex=False)
    metastatic = (
        status.str.contains("metastatic", regex=False)
        | raw.str.match("^IV[A-C]?$", na=False)
        | raw.str.contains("M1(?:\\D|$)", regex=True)
    )
    primary = status.str.contains("primary tumour", regex=False)
    clean = pd.Series("excluded", index=state_table.index, dtype=object)
    clean[primary] = "primary"
    clean[metastatic] = "metastatic"
    clean[model_system] = "excluded"
    reason[model_system] = "non_patient_model_or_recurrent_specimen"
    reason[clean.eq("excluded") & ~model_system] = "unresolved_disease_compartment"
    return (clean, reason)


def prepare_states(
    dataset: str, config: dict
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    source = (
        Path(config["experiments_01_02_root"])
        / dataset
        / "experiment_01_data_preparation"
        / "tables"
    )
    panel_size = int(config["panel_size"])
    matrix = pd.read_csv(source / f"mhn_training_matrix_p{panel_size}.csv")
    panel = pd.read_csv(source / f"event_panel_p{panel_size}.csv")
    state_table = pd.read_csv(source / f"state_table_p{panel_size}.csv")
    events = panel["event"].astype(str).tolist()
    if matrix.columns.astype(str).tolist() != events:
        raise ValueError(f"{dataset}: p15 event order mismatch")
    if len(matrix) != len(state_table):
        raise ValueError(f"{dataset}: matrix/state row mismatch")
    stage, reason = clean_stage(dataset, state_table, config["datasets"][dataset])
    work = state_table[
        ["analysis_id", "patient_id", "sample_id", "stage_raw", "metastasis_status"]
    ].copy()
    work["stage"] = stage
    work["inclusion_reason"] = reason
    work["genotype"] = [
        genotype_signature(row, events) for row in matrix.to_numpy(dtype=np.int32)
    ]
    work["event_count"] = matrix.sum(axis=1).astype(int).values
    work["included"] = work["stage"].ne("excluded")
    included = work[work["included"]].copy()
    included["state"] = included["stage"] + "::" + included["genotype"]
    occupancy = (
        included.groupby(["state", "stage", "genotype", "event_count"], dropna=False)
        .size()
        .rename("N_v")
        .reset_index()
    )
    n = len(included)
    occupancy["L_v"] = occupancy["N_v"] / max(n, 1)
    occupancy = occupancy.sort_values(
        ["N_v", "state"], ascending=[False, True]
    ).reset_index(drop=True)
    qc = (
        work.groupby(["included", "stage", "inclusion_reason"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    return (occupancy, events, qc)


def transition_lookup(transitions: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.source_genotype), str(row.event_added)): float(row.probability)
        for row in transitions.itertuples(index=False)
    }


def theta_next_probabilities(
    signature: str, events: list[str], log_theta: np.ndarray
) -> dict[str, float]:
    state = genotype_vector(signature, events)
    absent = np.flatnonzero(state == 0)
    if len(absent) == 0:
        return {}
    log_hazards = np.array(
        [
            log_theta[idx, idx] + log_theta[idx, state.astype(bool)].sum()
            for idx in absent
        ]
    )
    scaled = np.exp(log_hazards - log_hazards.max())
    probabilities = scaled / scaled.sum()
    return {events[idx]: float(probabilities[pos]) for pos, idx in enumerate(absent)}


def build_one_step_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    lookup: dict[tuple[str, str], float],
    rule: str,
    source_l: dict[str, float],
    probability_scale: float = 1.0,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    rows = []
    for target in occupancy.itertuples(index=False):
        vector = genotype_vector(target.genotype, events)
        for event_idx in np.flatnonzero(vector == 1):
            source_vector = vector.copy()
            source_vector[event_idx] = 0
            source_genotype = genotype_signature(source_vector, events)
            source_state = f"{target.stage}::{source_genotype}"
            if source_state not in observed_states:
                continue
            probability = lookup.get((source_genotype, events[event_idx]), 0.0)
            if probability <= 0:
                continue
            edge_probability = probability_scale * probability
            rows.append(
                {
                    "rule": rule,
                    "source_state": source_state,
                    "target_state": target.state,
                    "predecessor_type": "same_stage_one_event",
                    "event_added": events[event_idx],
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[source_state],
                    "inflow_contribution": source_l[source_state] * edge_probability,
                }
            )
    return pd.DataFrame(rows)


def build_stage_bridge_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    lookup: dict[tuple[str, str], float],
    source_l: dict[str, float],
    stage_order: list[str],
    stage_mass: float,
    rule: str,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    previous = {
        stage_order[idx]: stage_order[idx - 1] for idx in range(1, len(stage_order))
    }
    rows = []
    for target in occupancy.itertuples(index=False):
        if target.stage not in previous:
            continue
        prev_stage = previous[target.stage]
        same_genotype_source = f"{prev_stage}::{target.genotype}"
        if same_genotype_source in observed_states:
            edge_probability = stage_mass / 2
            rows.append(
                {
                    "rule": rule,
                    "source_state": same_genotype_source,
                    "target_state": target.state,
                    "predecessor_type": "previous_stage_same_genotype",
                    "event_added": "STAGE_ADVANCE",
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[same_genotype_source],
                    "inflow_contribution": source_l[same_genotype_source]
                    * edge_probability,
                }
            )
        vector = genotype_vector(target.genotype, events)
        for event_idx in np.flatnonzero(vector == 1):
            source_vector = vector.copy()
            source_vector[event_idx] = 0
            source_genotype = genotype_signature(source_vector, events)
            source_state = f"{prev_stage}::{source_genotype}"
            if source_state not in observed_states:
                continue
            probability = lookup.get((source_genotype, events[event_idx]), 0.0)
            if probability <= 0:
                continue
            edge_probability = stage_mass / 2 * probability
            rows.append(
                {
                    "rule": rule,
                    "source_state": source_state,
                    "target_state": target.state,
                    "predecessor_type": "previous_stage_plus_event",
                    "event_added": events[event_idx],
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[source_state],
                    "inflow_contribution": source_l[source_state] * edge_probability,
                }
            )
    return pd.DataFrame(rows)


def build_two_step_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    log_theta: np.ndarray,
    source_l: dict[str, float],
    two_step_mass: float,
    rule: str,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    rows = []
    probability_cache: dict[str, dict[str, float]] = {}

    def probabilities(signature: str) -> dict[str, float]:
        if signature not in probability_cache:
            probability_cache[signature] = theta_next_probabilities(
                signature, events, log_theta
            )
        return probability_cache[signature]

    for target in occupancy.itertuples(index=False):
        target_vector = genotype_vector(target.genotype, events)
        present = np.flatnonzero(target_vector == 1)
        for left_pos in range(len(present)):
            for right_pos in range(left_pos + 1, len(present)):
                first_idx = present[left_pos]
                second_idx = present[right_pos]
                source_vector = target_vector.copy()
                source_vector[[first_idx, second_idx]] = 0
                source_genotype = genotype_signature(source_vector, events)
                source_state = f"{target.stage}::{source_genotype}"
                if source_state not in observed_states:
                    continue
                first_event, second_event = (events[first_idx], events[second_idx])
                p0 = probabilities(source_genotype)
                intermediate_first = source_vector.copy()
                intermediate_first[first_idx] = 1
                signature_first = genotype_signature(intermediate_first, events)
                intermediate_second = source_vector.copy()
                intermediate_second[second_idx] = 1
                signature_second = genotype_signature(intermediate_second, events)
                path_probability = p0.get(first_event, 0) * probabilities(
                    signature_first
                ).get(second_event, 0) + p0.get(second_event, 0) * probabilities(
                    signature_second
                ).get(
                    first_event, 0
                )
                if path_probability <= 0:
                    continue
                edge_probability = two_step_mass * path_probability
                rows.append(
                    {
                        "rule": rule,
                        "source_state": source_state,
                        "target_state": target.state,
                        "predecessor_type": "same_stage_two_event",
                        "event_added": f"{first_event}+{second_event}",
                        "step_distance": 2,
                        "edge_probability": edge_probability,
                        "source_L": source_l[source_state],
                        "inflow_contribution": source_l[source_state]
                        * edge_probability,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_inflow(
    occupancy: pd.DataFrame,
    edges: pd.DataFrame,
    rule: str,
    min_count: int,
    min_inflow: float,
) -> pd.DataFrame:
    result = occupancy.copy()
    if edges.empty:
        result["F_hat"] = 0.0
        result["n_predecessors"] = 0
        result["dominant_predecessor"] = ""
        result["dominant_edge_probability"] = 0.0
        result["dominant_contribution"] = 0.0
        result["genotype_inflow"] = 0.0
        result["stage_inflow"] = 0.0
    else:
        totals = (
            edges.groupby("target_state")["inflow_contribution"].sum().rename("F_hat")
        )
        counts = edges.groupby("target_state").size().rename("n_predecessors")
        dominant = (
            edges.sort_values("inflow_contribution", ascending=False)
            .drop_duplicates("target_state")
            .set_index("target_state")
        )
        genotype = (
            edges[edges["predecessor_type"].str.startswith("same_stage")]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("genotype_inflow")
        )
        stage = (
            edges[edges["predecessor_type"].str.startswith("previous_stage")]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("stage_inflow")
        )
        result = result.join(totals, on="state").join(counts, on="state")
        result = result.join(genotype, on="state").join(stage, on="state")
        result["dominant_predecessor"] = result["state"].map(dominant["source_state"])
        result["dominant_edge_probability"] = result["state"].map(
            dominant["edge_probability"]
        )
        result["dominant_contribution"] = result["state"].map(
            dominant["inflow_contribution"]
        )
        for column in [
            "F_hat",
            "n_predecessors",
            "genotype_inflow",
            "stage_inflow",
            "dominant_edge_probability",
            "dominant_contribution",
        ]:
            result[column] = result[column].fillna(0)
        result["dominant_predecessor"] = result["dominant_predecessor"].fillna("")
    result["rule"] = rule
    result["count_eligible"] = result["N_v"] >= min_count
    result["inflow_eligible"] = result["F_hat"] >= min_inflow
    result["stable_for_experiment5"] = (
        result["count_eligible"] & result["inflow_eligible"]
    )
    result["flags"] = np.select(
        [
            ~result["count_eligible"],
            result["count_eligible"] & ~result["inflow_eligible"],
        ],
        ["rare_state", "low_or_zero_inflow"],
        default="stable",
    )
    return result.sort_values(["F_hat", "N_v"], ascending=[False, False])


def make_rule_outputs(
    dataset: str,
    config: dict,
    occupancy: pd.DataFrame,
    events: list[str],
    transitions: pd.DataFrame,
    theta: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    min_count = int(config["thresholds"]["minimum_state_count"])
    min_inflow = float(config["thresholds"]["minimum_inflow"])
    lookup = transition_lookup(transitions)
    source_l_main = occupancy.set_index("state")["L_v"].to_dict()
    main_name = config["main_rule"]["name"]
    main_edges = build_one_step_edges(
        occupancy, events, lookup, main_name, source_l_main
    )
    main = aggregate_inflow(occupancy, main_edges, main_name, min_count, min_inflow)
    stage_cfg = config["sensitivity_rules"]["stage_bridge"]
    stage_mass = float(stage_cfg["stage_mass"])
    stage_name = stage_cfg["name"]
    stage_edges = build_one_step_edges(
        occupancy,
        events,
        lookup,
        stage_name,
        source_l_main,
        probability_scale=1 - stage_mass,
    )
    bridge = build_stage_bridge_edges(
        occupancy,
        events,
        lookup,
        source_l_main,
        config["datasets"][dataset]["stage_order"],
        stage_mass,
        stage_name,
    )
    stage_edges = pd.concat([stage_edges, bridge], ignore_index=True)
    stage_result = aggregate_inflow(
        occupancy, stage_edges, stage_name, min_count, min_inflow
    )
    two_cfg = config["sensitivity_rules"]["two_step"]
    two_mass = float(two_cfg["two_step_mass"])
    two_name = two_cfg["name"]
    two_edges = build_one_step_edges(
        occupancy,
        events,
        lookup,
        two_name,
        source_l_main,
        probability_scale=1 - two_mass,
    )
    two_extra = build_two_step_edges(
        occupancy,
        events,
        theta.to_numpy(dtype=float),
        source_l_main,
        two_mass,
        two_name,
    )
    two_edges = pd.concat([two_edges, two_extra], ignore_index=True)
    two_result = aggregate_inflow(occupancy, two_edges, two_name, min_count, min_inflow)
    smooth_cfg = config["sensitivity_rules"]["smoothed"]
    smooth_name = smooth_cfg["name"]
    alpha = float(smooth_cfg["occupancy_alpha"])
    k = len(occupancy)
    n = int(occupancy["N_v"].sum())
    smooth_l = (
        (occupancy.set_index("state")["N_v"] + alpha) / (n + alpha * k)
    ).to_dict()
    smooth_edges = build_one_step_edges(
        occupancy, events, lookup, smooth_name, smooth_l
    )
    smooth_occupancy = occupancy.copy()
    smooth_occupancy["L_v"] = smooth_occupancy["state"].map(smooth_l)
    smooth_result = aggregate_inflow(
        smooth_occupancy, smooth_edges, smooth_name, min_count, min_inflow
    )
    return (
        {
            main_name: main,
            stage_name: stage_result,
            two_name: two_result,
            smooth_name: smooth_result,
        },
        {
            main_name: main_edges,
            stage_name: stage_edges,
            two_name: two_edges,
            smooth_name: smooth_edges,
        },
    )


def sensitivity_metrics(
    rule_tables: dict[str, pd.DataFrame], config: dict
) -> pd.DataFrame:
    main_name = config["main_rule"]["name"]
    main = rule_tables[main_name].set_index("state")
    top_k = int(config["thresholds"]["top_k"])
    rows = []
    for rule, table in rule_tables.items():
        if rule == main_name:
            continue
        other = table.set_index("state")
        joined = main[["F_hat", "N_v"]].join(
            other[["F_hat"]], how="inner", rsuffix="_other"
        )
        eligible = joined[
            (joined["N_v"] >= int(config["thresholds"]["minimum_state_count"]))
            & (joined["F_hat"] > 0)
            & (joined["F_hat_other"] > 0)
        ]
        rho = (
            float(spearmanr(eligible["F_hat"], eligible["F_hat_other"]).statistic)
            if len(eligible) >= 3
            else np.nan
        )
        main_top = set(joined[joined["F_hat"] > 0].nlargest(top_k, "F_hat").index)
        other_top = set(
            joined[joined["F_hat_other"] > 0].nlargest(top_k, "F_hat_other").index
        )
        overlap = len(main_top & other_top) / max(top_k, 1)
        relative_change = (
            np.abs(eligible["F_hat_other"] - eligible["F_hat"])
            / eligible["F_hat"].clip(lower=1e-12)
            if len(eligible)
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "comparison_rule": rule,
                "states_compared": len(eligible),
                "spearman_F_hat": rho,
                "top_k": top_k,
                "top_k_overlap": overlap,
                "median_absolute_relative_change": (
                    float(relative_change.median()) if len(relative_change) else np.nan
                ),
                "positive_inflow_states": int((other["F_hat"] > 0).sum()),
                "stable_states": int(other["stable_for_experiment5"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(
    dataset: str, rule_tables: dict[str, pd.DataFrame], config: dict
) -> pd.DataFrame:
    rows = []
    for rule, table in rule_tables.items():
        stable = table[table["stable_for_experiment5"]]
        positive = table[table["F_hat"] > 0]
        correlation_states = table[(table["F_hat"] > 0) & (table["N_v"] >= 5)]
        rho = (
            float(
                spearmanr(
                    correlation_states["L_v"], correlation_states["F_hat"]
                ).statistic
            )
            if len(correlation_states) >= 3
            else np.nan
        )
        rows.append(
            {
                "dataset_name": dataset,
                "rule": rule,
                "analysis_samples": int(table["N_v"].sum()),
                "observed_states": len(table),
                "positive_inflow_states": len(positive),
                "zero_inflow_states": int((table["F_hat"] <= 0).sum()),
                "stable_states": len(stable),
                "stable_sample_fraction": float(stable["L_v"].sum()),
                "median_positive_F_hat": (
                    float(positive["F_hat"].median()) if len(positive) else 0.0
                ),
                "max_F_hat": float(table["F_hat"].max()),
                "spearman_L_vs_F": rho,
            }
        )
    return pd.DataFrame(rows)


def top_edge_table(edges: pd.DataFrame, count: int) -> pd.DataFrame:
    if edges.empty:
        return edges
    return edges.nlargest(count, "inflow_contribution").copy()


def compact_inflow_edge_label(row: pd.Series) -> str:
    stage, genotype = str(row["source_state"]).split("::", 1)
    stage_label = "P" if stage == "primary" else "M"
    if genotype != "WT":
        events = genotype.split("+")
        if len(events) > 2:
            genotype = "+".join(events[:2]) + "+..."
    return f"{stage_label} | {genotype} -> +{row['event_added']}"


def canonical_inflow_edge(row: pd.Series) -> str:
    stage, genotype = str(row["source_state"]).split("::", 1)
    stage_label = "P" if stage == "primary" else "M"
    if genotype != "WT":
        genotype = "+".join(sorted(genotype.split("+")))
    return f"{stage_label} | {genotype} -> +{row['event_added']}"


def compact_canonical_edge(edge: str) -> str:
    prefix, event_added = edge.rsplit(" -> +", 1)
    stage, genotype = prefix.split(" | ", 1)
    if genotype != "WT":
        events = genotype.split("+")
        if len(events) > 3:
            genotype = "+".join(events[:3]) + "+..."
    return f"{stage} | {genotype} -> +{event_added}"


def _submission_short_name(dataset: str) -> str:
    return {"AACR_LUAD": "LUAD", "AACR_COAD": "COAD", "AACR_IDC": "IDC"}.get(
        dataset, dataset
    )


def _load_submission_tables(
    datasets: list[str], config: dict, result_root: Path
) -> dict[str, dict[str, pd.DataFrame]]:
    main_rule = config["main_rule"]["name"]
    stage_rule = config["sensitivity_rules"]["stage_bridge"]["name"]
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in datasets:
        root = result_root / dataset / "tables"
        tables[dataset] = {
            "main": pd.read_csv(root / f"inflow_table_{main_rule}.tsv", sep="\t"),
            "stage": pd.read_csv(root / f"inflow_table_{stage_rule}.tsv", sep="\t"),
            "edges": pd.read_csv(root / f"predecessor_edges_{main_rule}.tsv", sep="\t"),
            "sensitivity": pd.read_csv(root / "inflow_rule_sensitivity.tsv", sep="\t"),
        }
    return tables


def _dominant_edge_matrix(
    datasets: list[str], tables: dict[str, dict[str, pd.DataFrame]]
) -> tuple[list[str], np.ndarray]:
    per_cohort = 5
    all_tables: dict[str, pd.Series] = {}
    selected_edges: set[str] = set()
    for dataset in datasets:
        edges = tables[dataset]["edges"].copy()
        edges["canonical_edge"] = edges.apply(canonical_inflow_edge, axis=1)
        canonical = (
            edges.groupby("canonical_edge", as_index=False)["inflow_contribution"]
            .sum()
            .sort_values("inflow_contribution", ascending=False)
        )
        all_tables[dataset] = canonical.set_index("canonical_edge")[
            "inflow_contribution"
        ]
        selected_edges.update(canonical.head(per_cohort)["canonical_edge"])
    rows = []
    for edge in selected_edges:
        values = [float(all_tables[dataset].get(edge, 0.0)) for dataset in datasets]
        rows.append(
            {
                "edge": edge,
                "cohort_count": sum((value > 0 for value in values)),
                "maximum": max(values),
                "total": sum(values),
            }
        )
    edge_order = [
        row["edge"]
        for row in sorted(
            rows,
            key=lambda row: (row["cohort_count"], row["maximum"], row["total"]),
            reverse=True,
        )
    ][:15]
    matrix = np.asarray(
        [
            [float(all_tables[dataset].get(edge, 0.0)) for dataset in datasets]
            for edge in edge_order
        ],
        dtype=float,
    )
    return (edge_order, matrix)


def run_dataset(dataset: str, config: dict, result_root: Path) -> dict:
    dataset_root = result_root / dataset
    tables = dataset_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    occupancy, events, stage_qc = prepare_states(dataset, config)
    exp3_tables = Path(config["experiment_03_root"]) / dataset / "tables"
    transitions = pd.read_csv(exp3_tables / "genotype_transition.tsv", sep="\t")
    theta = pd.read_csv(exp3_tables / "theta.tsv", sep="\t", index_col=0)
    if theta.columns.astype(str).tolist() != events:
        raise ValueError(f"{dataset}: Experiment 3 theta event order mismatch")
    rule_tables, edge_tables = make_rule_outputs(
        dataset, config, occupancy, events, transitions, theta
    )
    sensitivity = sensitivity_metrics(rule_tables, config)
    summary = summary_metrics(dataset, rule_tables, config)
    occupancy.to_csv(tables / "state_occupancy_experiment4.tsv", sep="\t", index=False)
    stage_qc.to_csv(tables / "stage_inclusion_qc.tsv", sep="\t", index=False)
    for rule, table in rule_tables.items():
        table.to_csv(tables / f"inflow_table_{rule}.tsv", sep="\t", index=False)
    for rule, edges in edge_tables.items():
        edges.to_csv(tables / f"predecessor_edges_{rule}.tsv", sep="\t", index=False)
    sensitivity.to_csv(tables / "inflow_rule_sensitivity.tsv", sep="\t", index=False)
    summary.to_csv(tables / "experiment_04_metrics.tsv", sep="\t", index=False)
    record = summary[summary["rule"].eq(config["main_rule"]["name"])].iloc[0].to_dict()
    record["stage_excluded_samples"] = int(
        stage_qc.loc[~stage_qc["included"], "samples"].sum()
    )
    record["main_edges"] = len(edge_tables[config["main_rule"]["name"]])
    logging.info("%s complete: %s", dataset, record)
    return record


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    result_root = Path(config["result_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    setup_logging(result_root)
    records = []
    for dataset in datasets:
        print(f"[Experiment 4] Computing {dataset}...", flush=True)
        records.append(run_dataset(dataset, config, result_root))
        print(f"[Experiment 4] {dataset} complete.", flush=True)
    pd.DataFrame(records).to_csv(result_root / "experiment_04_summary.csv", index=False)


if __name__ == "__main__":
    main()
