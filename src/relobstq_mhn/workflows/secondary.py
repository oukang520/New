"""Compact table-only analyses derived from the primary state-score output."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from ..core.validation import require_columns
from ..evaluation.metrics import safe_rank_correlation


def inflow_computability_summary(scores: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether conditional inflow and R* are numerically usable."""

    require_columns(
        scores,
        ["state", "N_v", "L_v", "F_hat", "R_star", "count_eligible", "inflow_eligible", "eligible_relobstq"],
        "scores",
    )
    eligible = scores[scores["eligible_relobstq"].astype(bool)].copy()
    finite = np.isfinite(eligible[["L_v", "F_hat", "R_star"]].to_numpy(dtype=float)).all(axis=1)
    return pd.DataFrame(
        [
            {
                "observed_states": len(scores),
                "count_eligible_states": int(scores["count_eligible"].astype(bool).sum()),
                "positive_inflow_states": int(scores["inflow_eligible"].astype(bool).sum()),
                "rstar_eligible_states": len(eligible),
                "finite_eligible_states": int(finite.sum()),
                "finite_eligible_fraction": float(finite.mean()) if len(finite) else np.nan,
                "one_step_edges": len(edges),
                "total_conditional_inflow": float(pd.to_numeric(scores["F_hat"], errors="coerce").sum()),
                "median_positive_inflow": float(
                    pd.to_numeric(scores.loc[scores["F_hat"].gt(0), "F_hat"], errors="coerce").median()
                ),
            }
        ]
    )


def rstar_landscape_summary(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an eligible-state R* landscape and a compact cohort summary."""

    require_columns(scores, ["state", "stage", "genotype", "N_v", "L_v", "F_hat", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)].copy()
    eligible = eligible.replace([np.inf, -np.inf], np.nan).dropna(subset=["R_star", "L_v", "F_hat"])
    eligible = eligible.sort_values(["R_star", "N_v"], ascending=[False, False]).reset_index(drop=True)
    eligible.insert(0, "rstar_rank", np.arange(1, len(eligible) + 1))
    summary = pd.DataFrame(
        [
            {
                "eligible_states": len(eligible),
                "median_R_star": float(eligible["R_star"].median()) if len(eligible) else np.nan,
                "q1_R_star": float(eligible["R_star"].quantile(0.25)) if len(eligible) else np.nan,
                "q3_R_star": float(eligible["R_star"].quantile(0.75)) if len(eligible) else np.nan,
                "maximum_R_star": float(eligible["R_star"].max()) if len(eligible) else np.nan,
                "states_R_star_gt_2": int(eligible["R_star"].gt(2).sum()),
                "states_R_star_lt_0_5": int(eligible["R_star"].lt(0.5).sum()),
            }
        ]
    )
    return eligible, summary


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values preserving the input index."""

    values = pd.to_numeric(pvalues, errors="coerce")
    valid = values.dropna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return result
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    result.loc[order] = adjusted
    return result


def module_enrichment(
    scores: pd.DataFrame,
    event_to_module: dict[str, str],
    *,
    top_k: int = 20,
) -> pd.DataFrame:
    """Test a-priori event modules among top R* states using hypergeometric tails."""

    require_columns(scores, ["state", "genotype", "R_star"], "scores")
    eligible = scores.copy()
    if "eligible_relobstq" in eligible:
        eligible = eligible[eligible["eligible_relobstq"].astype(bool)].copy()
    top = eligible.nlargest(min(top_k, len(eligible)), "R_star")

    def events(frame: pd.DataFrame) -> set[str]:
        found: set[str] = set()
        for genotype in frame["genotype"].fillna("WT").astype(str):
            if genotype != "WT":
                found.update(genotype.split("+"))
        return found

    background = events(eligible)
    selected = events(top)
    modules = sorted(set(event_to_module.values()))
    rows = []
    for module in modules:
        members = {event for event, assigned in event_to_module.items() if assigned == module} & background
        overlap = members & selected
        pvalue = hypergeom.sf(len(overlap) - 1, len(background), len(members), len(selected)) if background else np.nan
        rows.append(
            {
                "module": module,
                "background_events": len(background),
                "module_events": len(members),
                "top_events": len(selected),
                "overlap_events": len(overlap),
                "overlap_event_names": "+".join(sorted(overlap)),
                "pvalue": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["qvalue"] = benjamini_hochberg(result["pvalue"])
    return result.sort_values(["qvalue", "pvalue", "module"], na_position="last").reset_index(drop=True)


def information_gain_summary(scores: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    """Quantify how R* reorders states relative to occupancy and inflow alone."""

    require_columns(scores, ["state", "L_v", "F_hat", "R_star"], "scores")
    work = scores.copy()
    if "eligible_relobstq" in work:
        work = work[work["eligible_relobstq"].astype(bool)]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["L_v", "F_hat", "R_star"])
    selected_k = min(top_k, len(work))
    rho_l, p_l = safe_rank_correlation(work["R_star"], work["L_v"])
    rho_f, p_f = safe_rank_correlation(work["R_star"], work["F_hat"])
    top_r = set(work.nlargest(selected_k, "R_star")["state"].astype(str))
    top_l = set(work.nlargest(selected_k, "L_v")["state"].astype(str))
    top_f = set(work.nlargest(selected_k, "F_hat")["state"].astype(str))
    return pd.DataFrame(
        [
            {
                "states": len(work),
                "top_k": selected_k,
                "spearman_R_vs_occupancy": rho_l,
                "spearman_R_vs_occupancy_p": p_l,
                "spearman_R_vs_inflow": rho_f,
                "spearman_R_vs_inflow_p": p_f,
                "top_k_overlap_R_occupancy": len(top_r & top_l),
                "top_k_overlap_R_inflow": len(top_r & top_f),
            }
        ]
    )


def attach_patient_scores(
    patient_states: pd.DataFrame,
    state_scores: pd.DataFrame,
    *,
    patient_state_column: str = "state_id",
) -> pd.DataFrame:
    """Attach state-level R* to patient records for clinical association tests."""

    require_columns(patient_states, [patient_state_column], "patient_states")
    require_columns(state_scores, ["state", "R_star"], "state_scores")
    lookup = state_scores[["state", "R_star"]].drop_duplicates("state")
    result = patient_states.merge(lookup, left_on=patient_state_column, right_on="state", how="left", validate="many_to_one")
    result["log2_R_star"] = np.log2(pd.to_numeric(result["R_star"], errors="coerce").clip(lower=1.0e-12))
    return result


def harrell_c_index(time: pd.Series, event: pd.Series, risk: pd.Series) -> float:
    """Harrell's concordance index for right-censored outcomes."""

    time = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(event, errors="coerce").to_numpy(dtype=float)
    risk = pd.to_numeric(risk, errors="coerce").to_numpy(dtype=float)
    comparable = concordant = 0.0
    for left in range(len(time)):
        for right in range(left + 1, len(time)):
            if not np.isfinite([time[left], time[right], event[left], event[right], risk[left], risk[right]]).all():
                continue
            if time[left] < time[right] and event[left] == 1:
                first, second = left, right
            elif time[right] < time[left] and event[right] == 1:
                first, second = right, left
            else:
                continue
            comparable += 1
            concordant += 1.0 if risk[first] > risk[second] else 0.5 if np.isclose(risk[first], risk[second]) else 0.0
    return float(concordant / comparable) if comparable else np.nan


def clinical_association(
    patients: pd.DataFrame,
    *,
    time_column: str = "survival_time",
    event_column: str = "survival_event",
    covariates: tuple[str, ...] = ("log2_R_star",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a Cox model and return coefficients plus a C-index audit.

    Continuous covariates are z-standardized within the supplied cohort.  This
    is an association analysis and must not be interpreted causally.
    """

    require_columns(patients, [time_column, event_column, *covariates], "patients")
    columns = [time_column, event_column, *covariates]
    frame = patients[columns].apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame[time_column] > 0) & frame[event_column].isin([0, 1])].copy()
    if len(frame) < max(20, len(covariates) * 10) or frame[event_column].sum() < 5:
        empty = pd.DataFrame(columns=["covariate", "coefficient", "hazard_ratio", "ci_low", "ci_high", "pvalue"])
        return empty, pd.DataFrame([{"patients": len(frame), "events": int(frame[event_column].sum()), "c_index": np.nan}])
    design = frame[list(covariates)].copy()
    for column in covariates:
        sd = float(design[column].std(ddof=0))
        design[column] = (design[column] - design[column].mean()) / sd if sd > 0 else 0.0
    try:
        from statsmodels.duration.hazard_regression import PHReg

        fit = PHReg(frame[time_column], design, status=frame[event_column]).fit(disp=0)
        parameters = np.asarray(fit.params, dtype=float)
        standard_errors = np.asarray(fit.bse, dtype=float)
        pvalues = np.asarray(fit.pvalues, dtype=float)
        rows = []
        for index, column in enumerate(covariates):
            coefficient = parameters[index]
            rows.append(
                {
                    "covariate": column,
                    "coefficient": coefficient,
                    "hazard_ratio": np.exp(coefficient),
                    "ci_low": np.exp(coefficient - 1.96 * standard_errors[index]),
                    "ci_high": np.exp(coefficient + 1.96 * standard_errors[index]),
                    "pvalue": pvalues[index],
                }
            )
        risk = design.to_numpy() @ parameters
        audit = pd.DataFrame(
            [
                {
                    "patients": len(frame),
                    "events": int(frame[event_column].sum()),
                    "c_index": harrell_c_index(frame[time_column], frame[event_column], pd.Series(risk)),
                }
            ]
        )
        return pd.DataFrame(rows), audit
    except Exception as exc:
        raise RuntimeError(f"Cox model fitting failed: {exc}") from exc
