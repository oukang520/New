"""Innovation-specific ablation and falsification analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.scoring import ScoreThresholds, compute_relative_dwell
from ..core.transitions import aggregate_inflow, same_stage_one_step_edges
from ..core.validation import require_columns


def _add_match_bins(scores: pd.DataFrame, bins: int) -> pd.DataFrame:
    work = scores.copy().reset_index(drop=True)
    work["log_N_v"] = np.log1p(pd.to_numeric(work["N_v"], errors="coerce"))
    work["log_F_hat"] = np.log10(pd.to_numeric(work["F_hat"], errors="coerce") + 1.0e-12)
    for source, target in [("log_N_v", "N_bin"), ("log_F_hat", "F_bin")]:
        values = work[source]
        if values.nunique(dropna=True) <= 1:
            work[target] = 0
        else:
            q = min(bins, int(values.nunique(dropna=True)))
            work[target] = pd.qcut(values.rank(method="first"), q=q, labels=False, duplicates="drop")
    return work


def matched_decoy_test(
    scores: pd.DataFrame,
    *,
    top_k: int = 10,
    quantile_bins: int = 4,
    minimum_decoys: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare top R* states with structurally matched real-state decoys."""

    require_columns(scores, ["state", "stage", "event_count", "N_v", "F_hat", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)]
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_star", "F_hat", "N_v"])
    eligible = _add_match_bins(eligible, quantile_bins)
    selected = eligible.nlargest(min(top_k, len(eligible)), "R_star")
    selected_states = set(selected["state"].astype(str))
    rows = []
    for rank, (_, target) in enumerate(selected.iterrows(), start=1):
        pool = eligible[~eligible["state"].astype(str).isin(selected_states)]
        tiers = [
            (
                "stage+events+Nbin+Fbin",
                pool[
                    pool["stage"].eq(target["stage"])
                    & pool["event_count"].eq(target["event_count"])
                    & pool["N_bin"].eq(target["N_bin"])
                    & pool["F_bin"].eq(target["F_bin"])
                ],
            ),
            (
                "stage+events+Fbin",
                pool[
                    pool["stage"].eq(target["stage"])
                    & pool["event_count"].eq(target["event_count"])
                    & pool["F_bin"].eq(target["F_bin"])
                ],
            ),
            ("stage+events", pool[pool["stage"].eq(target["stage"]) & pool["event_count"].eq(target["event_count"])]),
            ("all_eligible", pool),
        ]
        tier, decoys = tiers[-1]
        for candidate_tier, candidate in tiers:
            if len(candidate) >= minimum_decoys or candidate_tier == "all_eligible":
                tier, decoys = candidate_tier, candidate
                break
        target_r = float(target["R_star"])
        decoy_r = decoys["R_star"].astype(float)
        rows.append(
            {
                "rank": rank,
                "state": target["state"],
                "stage": target["stage"],
                "event_count": int(target["event_count"]),
                "R_star": target_r,
                "match_tier": tier,
                "decoy_count": len(decoys),
                "decoy_median_R_star": float(decoy_r.median()) if len(decoy_r) else np.nan,
                "matched_percentile": float((decoy_r <= target_r).mean()) if len(decoy_r) else np.nan,
                "log2_R_advantage": (
                    float(np.log2(target_r) - np.log2(decoy_r).median()) if len(decoy_r) and target_r > 0 else np.nan
                ),
            }
        )
    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "top_states_tested": len(details),
                "median_matched_percentile": details["matched_percentile"].median(),
                "fraction_above_decoy_q90": details["matched_percentile"].ge(0.90).mean(),
                "median_log2_R_advantage": details["log2_R_advantage"].median(),
            }
        ]
    )
    return details, summary


def inflow_pairing_falsification(
    scores: pd.DataFrame,
    *,
    top_k: int = 10,
    replicates: int = 400,
    seed: int = 20260630,
    epsilon: float = 1.0e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Break the learned L-F pairing within stage/event-count strata."""

    require_columns(scores, ["state", "stage", "event_count", "L_v", "F_hat", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)]
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["L_v", "F_hat", "R_star"]).reset_index(drop=True)
    top_k = min(top_k, len(eligible))
    observed_top = set(eligible.nlargest(top_k, "R_star")["state"].astype(str))
    strata = eligible["stage"].astype(str) + "|e" + eligible["event_count"].astype(str)
    groups = list(strata.groupby(strata).groups.values())
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, replicates + 1):
        shuffled = eligible["F_hat"].to_numpy(dtype=float).copy()
        for index in groups:
            positions = np.asarray(index, dtype=int)
            if len(positions) > 1:
                shuffled[positions] = rng.permutation(shuffled[positions])
        raw = eligible["L_v"].to_numpy(dtype=float) / (shuffled + epsilon)
        normalizer = np.nanmedian(raw)
        shuffled_r = raw / normalizer
        selected = np.argsort(shuffled_r)[-top_k:]
        overlap = len(observed_top.intersection(eligible.iloc[selected]["state"].astype(str)))
        rows.append({"replicate": replicate, "top_k": top_k, "top_overlap_fraction": overlap / max(top_k, 1)})
    replicates_table = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "replicates": replicates,
                "top_k": top_k,
                "observed_overlap_fraction": 1.0,
                "median_shuffled_overlap": replicates_table["top_overlap_fraction"].median(),
                "q1_shuffled_overlap": replicates_table["top_overlap_fraction"].quantile(0.25),
                "q3_shuffled_overlap": replicates_table["top_overlap_fraction"].quantile(0.75),
                "median_overlap_loss": 1.0 - replicates_table["top_overlap_fraction"].median(),
            }
        ]
    )
    return replicates_table, summary


def backbone_ablation(
    occupancy: pd.DataFrame,
    events: list[str],
    learned_probabilities,
    *,
    thresholds: ScoreThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare learned MHN inflow with a uniform one-step backbone."""

    thresholds = thresholds or ScoreThresholds()

    def uniform(genotype: str) -> dict[str, float]:
        present = set() if genotype == "WT" else set(str(genotype).split("+"))
        absent = [event for event in events if event not in present]
        return {event: 1.0 / len(absent) for event in absent} if absent else {}

    variants = []
    for name, provider in [("learned_mhn", learned_probabilities), ("uniform_one_step", uniform)]:
        edges = same_stage_one_step_edges(occupancy, events, provider, rule=name)
        inflow = aggregate_inflow(
            occupancy,
            edges,
            rule=name,
            minimum_state_count=thresholds.minimum_state_count,
            minimum_inflow=thresholds.minimum_inflow,
        )
        scores, _ = compute_relative_dwell(inflow, thresholds)
        variants.append(scores[["state", "N_v", "L_v", "F_hat", "R_star"]].assign(backbone=name))
    details = pd.concat(variants, ignore_index=True)
    pivot = details.pivot(index="state", columns="backbone", values="R_star").dropna()
    rho = spearmanr(pivot["learned_mhn"], pivot["uniform_one_step"]).statistic if len(pivot) >= 3 else np.nan
    learned_top = set(pivot.nlargest(min(10, len(pivot)), "learned_mhn").index)
    uniform_top = set(pivot.nlargest(min(10, len(pivot)), "uniform_one_step").index)
    summary = pd.DataFrame(
        [
            {
                "states_compared": len(pivot),
                "spearman_learned_vs_uniform": rho,
                "top10_jaccard": len(learned_top & uniform_top) / max(len(learned_top | uniform_top), 1),
            }
        ]
    )
    return details, summary
