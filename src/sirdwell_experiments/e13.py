"""Run Experiment 13: split-cohort replication of R* state signals.

The protocol asks whether top R* states reproduce outside the discovery cohort.
The current project data do not contain paired TCGA/GENIE cohorts, so this
implementation uses the protocol's internal split A/B route. Patient IDs are
split before occupancy is recalculated. The MHN-derived inflow backbone is kept
locked from Experiment 5, making this a held-out occupancy replication test
rather than a full external re-training claim.
"""

from __future__ import annotations
from .parameters import config_path as bundled_config_path
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import hypergeom, spearmanr

CONFIG_PATH = bundled_config_path("experiment_13.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 13.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--result-root")
    return parser.parse_args()


def canonical_state(state: object) -> str:
    text = str(state)
    if "::" not in text:
        return text
    stage, genotype = text.split("::", 1)
    if genotype == "WT" or genotype.strip() == "":
        return f"{stage}::WT"
    return f"{stage}::" + "+".join(sorted(genotype.split("+")))


def compact_state(state: str, max_events: int = 3) -> str:
    stage, genotype = str(state).split("::", 1)
    prefix = "P" if stage == "primary" else "M"
    if genotype == "WT":
        return f"{prefix} | WT"
    events = genotype.split("+")
    if len(events) > max_events:
        genotype = "+".join(events[:max_events]) + "+..."
    return f"{prefix} | {genotype}"


def read_scores(dataset: str, config: dict) -> pd.DataFrame:
    path = Path(config["experiment_05_root"]) / dataset / "tables" / "state_scores.tsv"
    scores = pd.read_csv(path, sep="\t")
    scores["canonical_state"] = scores["state"].map(canonical_state)
    keep = [
        "canonical_state",
        "state",
        "stage",
        "genotype",
        "event_count",
        "F_hat",
        "R_star",
        "log2_R_star",
        "eligible_experiment5",
        "high_confidence",
        "clinical_annotation",
        "interpretation_flag",
    ]
    return scores[keep].drop_duplicates("canonical_state")


def read_state_table(dataset: str, config: dict) -> pd.DataFrame:
    path = Path(config["experiment_ready_root"]) / dataset / "state_table.csv"
    table = pd.read_csv(path)
    usable = table["usable_for_relobstq"].astype("boolean").fillna(False)
    table = table[usable].copy()
    table["canonical_state"] = table["state_id"].map(canonical_state)
    return table


def patient_split(
    patient_ids: np.ndarray, fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted((str(pid) for pid in patient_ids)))
    rng.shuffle(ids)
    cut = int(round(len(ids) * fraction))
    cut = min(max(cut, 1), len(ids) - 1)
    return (set(ids[:cut]), set(ids[cut:]))


def compute_split_scores(
    state_table: pd.DataFrame, scores: pd.DataFrame, patient_ids: set[str], config: dict
) -> pd.DataFrame:
    thresholds = config["analysis"]
    sub = state_table[state_table["patient_id"].astype(str).isin(patient_ids)].copy()
    total = max(len(sub), 1)
    counts = sub.groupby("canonical_state", dropna=False).size().rename("N_split")
    frame = scores.merge(
        counts, left_on="canonical_state", right_index=True, how="left"
    )
    frame["N_split"] = frame["N_split"].fillna(0).astype(int)
    frame["L_split"] = frame["N_split"] / total
    frame["split_total_samples"] = int(total)
    frame["split_total_patients"] = int(len(patient_ids))
    frame["eligible_split"] = (
        frame["eligible_experiment5"].astype("boolean").fillna(False)
        & frame["N_split"].ge(int(thresholds["minimum_state_count"]))
        & frame["F_hat"].ge(float(thresholds["minimum_inflow"]))
    )
    epsilon = float(thresholds["epsilon"])
    frame["R_raw_split"] = frame["L_split"] / (frame["F_hat"] + epsilon)
    normalizer = frame.loc[frame["eligible_split"], "R_raw_split"].median()
    if not np.isfinite(normalizer) or normalizer <= 0:
        frame["R_star_split"] = np.nan
    else:
        frame["R_star_split"] = frame["R_raw_split"] / float(normalizer)
    frame["log2_R_star_split"] = np.log2(frame["R_star_split"].clip(lower=1e-12))
    return frame


def compare_splits(
    dataset: str,
    short: str,
    split_a: pd.DataFrame,
    split_b: pd.DataFrame,
    repeat: int,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    top_k = int(config["analysis"]["top_k"])
    a = split_a.rename(
        columns={
            "R_star_split": "R_star_A",
            "log2_R_star_split": "log2_R_star_A",
            "N_split": "N_A",
            "eligible_split": "eligible_A",
        }
    )
    b = split_b.rename(
        columns={
            "R_star_split": "R_star_B",
            "log2_R_star_split": "log2_R_star_B",
            "N_split": "N_B",
            "eligible_split": "eligible_B",
        }
    )
    common = a[
        [
            "canonical_state",
            "state",
            "stage",
            "genotype",
            "event_count",
            "clinical_annotation",
            "interpretation_flag",
            "R_star_A",
            "log2_R_star_A",
            "N_A",
            "eligible_A",
        ]
    ].merge(
        b[["canonical_state", "R_star_B", "log2_R_star_B", "N_B", "eligible_B"]],
        on="canonical_state",
        how="inner",
    )
    common["common_eligible"] = common["eligible_A"].astype(bool) & common[
        "eligible_B"
    ].astype(bool)
    eligible = common[common["common_eligible"]].copy()
    if len(eligible) >= int(config["analysis"]["minimum_common_states"]):
        rho = float(
            spearmanr(
                eligible["log2_R_star_A"], eligible["log2_R_star_B"], nan_policy="omit"
            ).correlation
        )
    else:
        rho = np.nan
    top_a = set(
        a[a["eligible_A"].astype(bool)].nlargest(top_k, "R_star_A")["canonical_state"]
    )
    top_b = set(
        b[b["eligible_B"].astype(bool)].nlargest(top_k, "R_star_B")["canonical_state"]
    )
    overlap = len(top_a.intersection(top_b))
    common_state_count = int(len(eligible))
    common_universe = set(eligible["canonical_state"])
    k_a = int(len(top_a))
    k_b = int(len(top_b))
    k_a_common = int(len(top_a.intersection(common_universe)))
    k_b_common = int(len(top_b.intersection(common_universe)))
    if common_state_count > 0 and k_a_common > 0 and (k_b_common > 0):
        null_mean = float(k_a_common * k_b_common / common_state_count)
        null_p = float(
            hypergeom.sf(overlap - 1, common_state_count, k_a_common, k_b_common)
        )
    else:
        null_mean = np.nan
        null_p = np.nan
    top_union = sorted(top_a.union(top_b))
    core = common[
        common["canonical_state"].isin(top_union) & common["common_eligible"]
    ].copy()
    if core.empty:
        direction_concordance = np.nan
        direction_reversals = np.nan
    else:
        same = core["R_star_A"].ge(1.0) == core["R_star_B"].ge(1.0)
        direction_concordance = float(same.mean())
        direction_reversals = int((~same).sum())
    metrics = {
        "dataset_name": dataset,
        "short_name": short,
        "repeat": repeat,
        "common_states": common_state_count,
        "spearman_rho": rho,
        "top_k": top_k,
        "top_overlap": int(overlap),
        "top_overlap_fraction": float(overlap / top_k),
        "top_overlap_null_mean": null_mean,
        "top_overlap_null_p_value": null_p,
        "top_overlap_enrichment": (
            float(overlap / null_mean)
            if np.isfinite(null_mean) and null_mean > 0
            else np.nan
        ),
        "direction_concordance": direction_concordance,
        "direction_reversals": direction_reversals,
        "top_A_count": k_a,
        "top_B_count": k_b,
        "top_A_common_count": k_a_common,
        "top_B_common_count": k_b_common,
    }
    common["dataset_name"] = dataset
    common["short_name"] = short
    common["repeat"] = repeat
    common["compact_state"] = common["state"].map(compact_state)
    common["in_top_A"] = common["canonical_state"].isin(top_a)
    common["in_top_B"] = common["canonical_state"].isin(top_b)
    return (metrics, common)


def run_analysis(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    representative_rows = []
    audit_rows = []
    seed0 = int(config["analysis"]["random_seed"])
    split_repeats = int(config["analysis"]["split_repeats"])
    for dataset, ds_cfg in config["datasets"].items():
        short = ds_cfg["short_name"]
        state_table = read_state_table(dataset, config)
        scores = read_scores(dataset, config)
        patients = state_table["patient_id"].astype(str).drop_duplicates().to_numpy()
        audit_rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "split_patients": int(len(patients)),
                "state_table_rows": int(len(state_table)),
                "scored_states": int(len(scores)),
                "locked_inflow_backbone": bool(
                    config["analysis"]["locked_inflow_backbone"]
                ),
            }
        )
        dataset_seed = seed0 + sum((ord(char) for char in dataset)) * 100
        for repeat in range(1, split_repeats + 1):
            split_a_ids, split_b_ids = patient_split(
                patients,
                float(config["analysis"]["split_fraction"]),
                dataset_seed + repeat,
            )
            split_a = compute_split_scores(state_table, scores, split_a_ids, config)
            split_b = compute_split_scores(state_table, scores, split_b_ids, config)
            metrics, common = compare_splits(
                dataset, short, split_a, split_b, repeat, config
            )
            metrics_rows.append(metrics)
            if repeat == 1:
                representative_rows.append(common)
    representative = pd.concat(representative_rows, ignore_index=True)
    return (pd.DataFrame(metrics_rows), representative, pd.DataFrame(audit_rows))


def summarize_metrics(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    criteria = config["success_criteria"]
    for dataset, sub in metrics.groupby("dataset_name", sort=False):
        median_rho = float(sub["spearman_rho"].median())
        median_overlap = float(sub["top_overlap"].median())
        median_common = float(sub["common_states"].median())
        evaluable = bool(
            np.isfinite(median_rho)
            and median_common >= int(config["analysis"]["minimum_common_states"])
        )
        rows.append(
            {
                "dataset_name": dataset,
                "short_name": sub["short_name"].iloc[0],
                "repeats": int(len(sub)),
                "evaluable": evaluable,
                "median_common_states": median_common,
                "median_spearman_rho": median_rho,
                "iqr_spearman_rho": float(
                    sub["spearman_rho"].quantile(0.75)
                    - sub["spearman_rho"].quantile(0.25)
                ),
                "median_top10_overlap": median_overlap,
                "iqr_top10_overlap": float(
                    sub["top_overlap"].quantile(0.75)
                    - sub["top_overlap"].quantile(0.25)
                ),
                "median_top10_null_mean": float(sub["top_overlap_null_mean"].median()),
                "median_top10_enrichment": float(
                    sub["top_overlap_enrichment"].median()
                ),
                "fraction_top10_above_null_p05": float(
                    sub["top_overlap_null_p_value"].lt(0.05).mean()
                ),
                "median_direction_concordance": float(
                    sub["direction_concordance"].median()
                ),
                "rho_status": (
                    "not_evaluable"
                    if not evaluable
                    else (
                        "good"
                        if median_rho >= float(criteria["spearman_good"])
                        else (
                            "acceptable"
                            if median_rho >= float(criteria["spearman_acceptable"])
                            else "below"
                        )
                    )
                ),
                "overlap_status": (
                    "not_evaluable"
                    if not evaluable
                    else (
                        "good"
                        if median_overlap >= float(criteria["top10_overlap_good"])
                        else (
                            "acceptable"
                            if median_overlap
                            >= float(criteria["top10_overlap_acceptable"])
                            else "below"
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.result_root:
        config["result_root"] = args.result_root
    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    metrics, representative, audit = run_analysis(config)
    summary = summarize_metrics(metrics, config)
    metrics.to_csv(tables / "split_replication_metrics.tsv", sep="\t", index=False)
    representative.to_csv(
        tables / "representative_split_state_scores.tsv", sep="\t", index=False
    )
    audit.to_csv(tables / "split_replication_audit.tsv", sep="\t", index=False)
    summary.to_csv(tables / "experiment_13_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
