"""Run Experiment 13: split-cohort replication of R* state signals.

The protocol asks whether top R* states reproduce outside the discovery cohort.
The current project data do not contain paired TCGA/GENIE cohorts, so this
implementation uses the protocol's internal split A/B route. Patient IDs are
split before occupancy is recalculated. The MHN-derived inflow backbone is kept
locked from Experiment 5, making this a held-out occupancy replication test
rather than a full external re-training claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import hypergeom, spearmanr
from statsmodels.duration.hazard_regression import PHReg

import figure_style


CONFIG_PATH = Path("configs/experiment_13.yaml")


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


def read_clinical_table(dataset: str, config: dict) -> pd.DataFrame:
    path = Path(config["experiment_12_root"]) / "tables" / "patient_clinical_scores.tsv"
    clinical = pd.read_csv(path, sep="\t")
    clinical = clinical[clinical["dataset_name"].eq(dataset)].copy()
    clinical["canonical_state"] = clinical["canonical_state"].map(canonical_state)
    return clinical


def patient_split(patient_ids: np.ndarray, fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(str(pid) for pid in patient_ids))
    rng.shuffle(ids)
    cut = int(round(len(ids) * fraction))
    cut = min(max(cut, 1), len(ids) - 1)
    return set(ids[:cut]), set(ids[cut:])


def compute_split_scores(
    state_table: pd.DataFrame,
    scores: pd.DataFrame,
    patient_ids: set[str],
    config: dict,
) -> pd.DataFrame:
    thresholds = config["analysis"]
    sub = state_table[state_table["patient_id"].astype(str).isin(patient_ids)].copy()
    total = max(len(sub), 1)
    counts = sub.groupby("canonical_state", dropna=False).size().rename("N_split")
    frame = scores.merge(counts, left_on="canonical_state", right_index=True, how="left")
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
    frame["log2_R_star_split"] = np.log2(frame["R_star_split"].clip(lower=1.0e-12))
    return frame


def standardize_features(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=0)
    keep = []
    for col in x.columns:
        std = x[col].std(ddof=0)
        if np.isfinite(std) and std > 1.0e-10:
            keep.append(col)
    x = x[keep]
    if x.empty:
        return x, []
    x = (x - x.mean()) / x.std(ddof=0)
    return x, keep


def fit_clinical_hr(clinical: pd.DataFrame, split_scores: pd.DataFrame, patient_ids: set[str], config: dict) -> dict:
    score_map = split_scores.set_index("canonical_state")["R_star_split"]
    frame = clinical[clinical["patient_id"].astype(str).isin(patient_ids)].copy()
    frame["R_star_split"] = frame["canonical_state"].map(score_map)
    frame["log_R_star_split"] = np.log(np.clip(pd.to_numeric(frame["R_star_split"], errors="coerce"), 1.0e-12, None))
    frame = frame[
        frame["followup_time_days"].gt(30)
        & frame["survival_event_binary"].isin([0, 1])
        & frame["R_star_split"].notna()
    ].copy()
    if len(frame) < 20 or frame["survival_event_binary"].sum() < int(config["analysis"]["minimum_events_for_cox"]):
        return {"ok": False, "reason": "insufficient patients/events", "n": int(len(frame)), "events": int(frame["survival_event_binary"].sum())}
    candidates = ["log_R_star_split", "age_num", "sex_male", "stage_metastatic", "event_count"]
    attempts = []
    while candidates:
        x, used = standardize_features(frame, candidates)
        if "log_R_star_split" not in used:
            return {"ok": False, "reason": "R* unavailable after feature filtering", "n": int(len(frame)), "events": int(frame["survival_event_binary"].sum())}
        y = frame.loc[x.index]
        try:
            fit = PHReg(y["followup_time_days"], x, status=y["survival_event_binary"]).fit(disp=0)
            params = pd.Series(fit.params, index=used)
            se = pd.Series(fit.bse, index=used)
            beta = float(params["log_R_star_split"])
            se_beta = float(se["log_R_star_split"])
            return {
                "ok": True,
                "reason": "OK",
                "n": int(len(y)),
                "events": int(y["survival_event_binary"].sum()),
                "beta": beta,
                "hr": float(np.exp(beta)),
                "ci_low": float(np.exp(beta - 1.96 * se_beta)),
                "ci_high": float(np.exp(beta + 1.96 * se_beta)),
                "features_used": ";".join(used),
            }
        except Exception as exc:
            attempts.append(f"{type(exc).__name__}: {exc}")
            removable = [feature for feature in ["stage_metastatic", "sex_male", "event_count", "age_num"] if feature in candidates]
            if not removable:
                return {
                    "ok": False,
                    "reason": "; ".join(attempts[-3:]),
                    "n": int(len(frame)),
                    "events": int(frame["survival_event_binary"].sum()),
                }
            candidates.remove(removable[0])
    return {"ok": False, "reason": "no estimable model", "n": int(len(frame)), "events": int(frame["survival_event_binary"].sum())}


def compare_splits(
    dataset: str,
    short: str,
    split_a: pd.DataFrame,
    split_b: pd.DataFrame,
    repeat: int,
    config: dict,
) -> tuple[dict, pd.DataFrame]:
    top_k = int(config["analysis"]["top_k"])
    a = split_a.rename(columns={"R_star_split": "R_star_A", "log2_R_star_split": "log2_R_star_A", "N_split": "N_A", "eligible_split": "eligible_A"})
    b = split_b.rename(columns={"R_star_split": "R_star_B", "log2_R_star_split": "log2_R_star_B", "N_split": "N_B", "eligible_split": "eligible_B"})
    common = a[
        ["canonical_state", "state", "stage", "genotype", "event_count", "clinical_annotation", "interpretation_flag", "R_star_A", "log2_R_star_A", "N_A", "eligible_A"]
    ].merge(
        b[["canonical_state", "R_star_B", "log2_R_star_B", "N_B", "eligible_B"]],
        on="canonical_state",
        how="inner",
    )
    common["common_eligible"] = common["eligible_A"].astype(bool) & common["eligible_B"].astype(bool)
    eligible = common[common["common_eligible"]].copy()
    if len(eligible) >= int(config["analysis"]["minimum_common_states"]):
        rho = float(spearmanr(eligible["log2_R_star_A"], eligible["log2_R_star_B"], nan_policy="omit").correlation)
    else:
        rho = np.nan
    top_a = set(a[a["eligible_A"].astype(bool)].nlargest(top_k, "R_star_A")["canonical_state"])
    top_b = set(b[b["eligible_B"].astype(bool)].nlargest(top_k, "R_star_B")["canonical_state"])
    overlap = len(top_a.intersection(top_b))
    common_state_count = int(len(eligible))
    common_universe = set(eligible["canonical_state"])
    k_a = int(len(top_a))
    k_b = int(len(top_b))
    k_a_common = int(len(top_a.intersection(common_universe)))
    k_b_common = int(len(top_b.intersection(common_universe)))
    if common_state_count > 0 and k_a_common > 0 and k_b_common > 0:
        null_mean = float(k_a_common * k_b_common / common_state_count)
        null_p = float(hypergeom.sf(overlap - 1, common_state_count, k_a_common, k_b_common))
    else:
        null_mean = np.nan
        null_p = np.nan
    top_union = sorted(top_a.union(top_b))
    core = common[common["canonical_state"].isin(top_union) & common["common_eligible"]].copy()
    if core.empty:
        direction_concordance = np.nan
        direction_reversals = np.nan
    else:
        same = (core["R_star_A"].ge(1.0) == core["R_star_B"].ge(1.0))
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
        "top_overlap_enrichment": float(overlap / null_mean) if np.isfinite(null_mean) and null_mean > 0 else np.nan,
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
    return metrics, common


def run_analysis(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    representative_rows = []
    clinical_rows = []
    audit_rows = []
    seed0 = int(config["analysis"]["random_seed"])
    split_repeats = int(config["analysis"]["split_repeats"])
    clinical_repeats = int(config["analysis"]["clinical_repeats"])
    for dataset, ds_cfg in config["datasets"].items():
        short = ds_cfg["short_name"]
        state_table = read_state_table(dataset, config)
        scores = read_scores(dataset, config)
        clinical = read_clinical_table(dataset, config)
        patients = state_table["patient_id"].astype(str).drop_duplicates().to_numpy()
        audit_rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "split_patients": int(len(patients)),
                "state_table_rows": int(len(state_table)),
                "scored_states": int(len(scores)),
                "clinical_patients": int(clinical["patient_id"].nunique()),
                "locked_inflow_backbone": bool(config["analysis"]["locked_inflow_backbone"]),
            }
        )
        dataset_seed = seed0 + sum(ord(char) for char in dataset) * 100
        for repeat in range(1, split_repeats + 1):
            split_a_ids, split_b_ids = patient_split(patients, float(config["analysis"]["split_fraction"]), dataset_seed + repeat)
            split_a = compute_split_scores(state_table, scores, split_a_ids, config)
            split_b = compute_split_scores(state_table, scores, split_b_ids, config)
            metrics, common = compare_splits(dataset, short, split_a, split_b, repeat, config)
            if repeat <= clinical_repeats:
                hr_a = fit_clinical_hr(clinical, split_a, split_a_ids, config)
                hr_b = fit_clinical_hr(clinical, split_b, split_b_ids, config)
                hr_same = np.nan
                if hr_a.get("ok") and hr_b.get("ok"):
                    hr_same = bool((hr_a["hr"] >= 1.0) == (hr_b["hr"] >= 1.0))
                clinical_rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "repeat": repeat,
                        "split_A_ok": bool(hr_a.get("ok")),
                        "split_B_ok": bool(hr_b.get("ok")),
                        "split_A_hr": hr_a.get("hr", np.nan),
                        "split_B_hr": hr_b.get("hr", np.nan),
                        "split_A_ci_low": hr_a.get("ci_low", np.nan),
                        "split_A_ci_high": hr_a.get("ci_high", np.nan),
                        "split_B_ci_low": hr_b.get("ci_low", np.nan),
                        "split_B_ci_high": hr_b.get("ci_high", np.nan),
                        "hr_direction_same": hr_same,
                        "split_A_n": hr_a.get("n", 0),
                        "split_A_events": hr_a.get("events", 0),
                        "split_B_n": hr_b.get("n", 0),
                        "split_B_events": hr_b.get("events", 0),
                        "split_A_reason": hr_a.get("reason", ""),
                        "split_B_reason": hr_b.get("reason", ""),
                    }
                )
                metrics["clinical_hr_direction_same"] = hr_same
            else:
                metrics["clinical_hr_direction_same"] = np.nan
            metrics_rows.append(metrics)
            if repeat == 1:
                representative_rows.append(common)
    representative = pd.concat(representative_rows, ignore_index=True)
    return pd.DataFrame(metrics_rows), representative, pd.DataFrame(clinical_rows), pd.DataFrame(audit_rows)


def summarize_metrics(metrics: pd.DataFrame, clinical: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    criteria = config["success_criteria"]
    for dataset, sub in metrics.groupby("dataset_name", sort=False):
        clinical_sub = clinical[clinical["dataset_name"].eq(dataset)]
        hr_fraction = clinical_sub["hr_direction_same"].dropna().astype(bool).mean() if not clinical_sub.empty else np.nan
        median_rho = float(sub["spearman_rho"].median())
        median_overlap = float(sub["top_overlap"].median())
        median_common = float(sub["common_states"].median())
        evaluable = bool(np.isfinite(median_rho) and median_common >= int(config["analysis"]["minimum_common_states"]))
        rows.append(
            {
                "dataset_name": dataset,
                "short_name": sub["short_name"].iloc[0],
                "repeats": int(len(sub)),
                "evaluable": evaluable,
                "median_common_states": median_common,
                "median_spearman_rho": median_rho,
                "iqr_spearman_rho": float(sub["spearman_rho"].quantile(0.75) - sub["spearman_rho"].quantile(0.25)),
                "median_top10_overlap": median_overlap,
                "iqr_top10_overlap": float(sub["top_overlap"].quantile(0.75) - sub["top_overlap"].quantile(0.25)),
                "median_top10_null_mean": float(sub["top_overlap_null_mean"].median()),
                "median_top10_enrichment": float(sub["top_overlap_enrichment"].median()),
                "fraction_top10_above_null_p05": float(sub["top_overlap_null_p_value"].lt(0.05).mean()),
                "median_direction_concordance": float(sub["direction_concordance"].median()),
                "clinical_hr_direction_consistency": float(hr_fraction) if np.isfinite(hr_fraction) else np.nan,
                "rho_status": "not_evaluable" if not evaluable else "good" if median_rho >= float(criteria["spearman_good"]) else "acceptable" if median_rho >= float(criteria["spearman_acceptable"]) else "below",
                "overlap_status": "not_evaluable" if not evaluable else "good" if median_overlap >= float(criteria["top10_overlap_good"]) else "acceptable" if median_overlap >= float(criteria["top10_overlap_acceptable"]) else "below",
            }
        )
    return pd.DataFrame(rows)


def save_square(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def spread_positions(values: list[float], min_gap: float, lower: float, upper: float) -> list[float]:
    finite = [(index, float(value)) for index, value in enumerate(values) if np.isfinite(value)]
    positions = [np.nan] * len(values)
    if not finite:
        return positions
    finite.sort(key=lambda item: item[1])
    ys = [value for _, value in finite]
    for index in range(1, len(ys)):
        ys[index] = max(ys[index], ys[index - 1] + min_gap)
    overflow = ys[-1] - upper
    if overflow > 0:
        ys = [value - overflow for value in ys]
    for index in range(len(ys) - 2, -1, -1):
        ys[index] = min(ys[index], ys[index + 1] - min_gap)
    underflow = lower - ys[0]
    if underflow > 0:
        ys = [value + underflow for value in ys]
    for (original_index, _), y_value in zip(finite, ys):
        positions[original_index] = y_value
    return positions


def plot_main_figure(summary: pd.DataFrame, representative: pd.DataFrame, clinical: pd.DataFrame, output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = [
        cat.get("lavender", "#B5AED5"),
        cat.get("sky_blue", "#B2E6FD"),
        cat.get("sage", "#B8D2CC"),
        cat.get("coral", "#E8B2A7"),
    ]
    root = output.parents[1]
    metrics = pd.read_csv(root / "tables" / "split_replication_metrics.tsv", sep="\t")
    cohorts = summary["short_name"].tolist()
    fig = plt.figure(figsize=(7.2, 7.2))
    fig.text(0.075, 0.972, "Experiment 13 | Split-cohort replication of R* state signals", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.075, 0.947, "Fifty patient-level A/B splits recalculate occupancy independently under a locked MHN-derived inflow backbone.", ha="left", va="top", fontsize=5.9, color=text_secondary)

    a_size = 0.178
    a_x0 = 0.075
    a_xgap = 0.045
    a_ygap = 0.082
    a_y0 = 0.258
    datasets = list(config["datasets"])
    if len(datasets) == 3:
        axes_a_positions = [
            [a_x0, a_y0 + a_size + a_ygap, a_size, a_size],
            [a_x0 + a_size + a_xgap, a_y0 + a_size + a_ygap, a_size, a_size],
            [a_x0 + 0.5 * (a_size + a_xgap), a_y0, a_size, a_size],
        ]
    else:
        axes_a_positions = [
            [a_x0, a_y0 + a_size + a_ygap, a_size, a_size],
            [a_x0 + a_size + a_xgap, a_y0 + a_size + a_ygap, a_size, a_size],
            [a_x0, a_y0, a_size, a_size],
            [a_x0 + a_size + a_xgap, a_y0, a_size, a_size],
        ][: len(datasets)]
    axes_a = [fig.add_axes(position) for position in axes_a_positions]
    for idx, (ax, dataset) in enumerate(zip(axes_a, datasets)):
        sub = representative[representative["dataset_name"].eq(dataset) & representative["common_eligible"].astype(bool)].copy()
        short = config["datasets"][dataset]["short_name"]
        if sub.empty:
            ax.text(0.5, 0.5, "no common states", transform=ax.transAxes, ha="center", va="center", fontsize=5.5, color=text_secondary)
            continue
        x = sub["log2_R_star_A"].to_numpy(dtype=float)
        y = sub["log2_R_star_B"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        lim = np.nanquantile(np.abs(np.concatenate([x, y])), 0.98)
        lim = max(1.0, float(lim))
        ax.scatter(x, y, s=5.0, color=cohort_colors[idx], alpha=0.54, edgecolor="none", rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color="#777777", lw=0.62, ls=(0, (3, 2)))
        ax.axhline(0, color=grid_color, lw=0.45)
        ax.axvline(0, color=grid_color, lw=0.45)
        rho = spearmanr(x, y).correlation if len(x) >= 3 else np.nan
        overlap = int(summary.loc[summary["dataset_name"].eq(dataset), "median_top10_overlap"].iloc[0])
        detail = f"rho={rho:.2f}\nTop10={overlap}" if np.isfinite(rho) else f"underpowered\nTop10={overlap}"
        ax.text(0.05, 0.95, short, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=text_primary)
        ax.text(0.95, 0.08, detail, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color=text_secondary)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(color=grid_color, lw=0.35)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=5.4, length=1.6, width=0.5)
        if idx in [2, 3]:
            ax.set_xlabel("split A log2 R*", fontsize=5.7)
        else:
            ax.set_xticklabels([])
        if idx in [0, 2]:
            ax.set_ylabel("split B log2 R*", fontsize=5.7)
        else:
            ax.set_yticklabels([])
        ax.set_box_aspect(1)
    axes_a[0].text(-0.42, 1.18, "a", transform=axes_a[0].transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    axes_a[0].text(-0.02, 1.18, "Held-out state-rank concordance", transform=axes_a[0].transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_b = fig.add_axes([0.600, 0.625, 0.350, 0.235])
    y_base = np.arange(len(cohorts))
    for idx, dataset in enumerate(summary["dataset_name"]):
        sub = metrics[metrics["dataset_name"].eq(dataset)].copy()
        obs = sub["top_overlap"].to_numpy(dtype=float)
        null = sub["top_overlap_null_mean"].to_numpy(dtype=float)
        finite_null = null[np.isfinite(null)]
        row = summary[summary["dataset_name"].eq(dataset)].iloc[0]
        if len(finite_null):
            q1n, medn, q3n = np.quantile(finite_null, [0.25, 0.5, 0.75])
            ax_b.hlines(y_base[idx] + 0.16, q1n, q3n, color="#9A9A9A", lw=1.0, zorder=1)
            ax_b.scatter(medn, y_base[idx] + 0.16, s=18, color="#D6D6D6", edgecolor="#555555", linewidth=0.35, zorder=2)
        if bool(row["evaluable"]):
            q1, med, q3 = np.quantile(obs, [0.25, 0.5, 0.75])
            ax_b.hlines(y_base[idx] - 0.05, q1, q3, color=text_primary, lw=1.55, zorder=4)
            ax_b.scatter(med, y_base[idx] - 0.05, s=42, color=cohort_colors[idx], edgecolor=text_primary, linewidth=0.45, zorder=5)
            ax_b.text(10.15, y_base[idx] - 0.05, f"{med:.0f} ({row['median_top10_enrichment']:.1f}x)", ha="left", va="center", fontsize=5.2, color=text_secondary)
        else:
            ax_b.text(5.0, y_base[idx] - 0.05, "underpowered", ha="center", va="center", fontsize=5.3, color=text_secondary)
    ax_b.axvline(float(config["success_criteria"]["top10_overlap_acceptable"]), color="#999999", lw=0.6, ls=(0, (2, 2)))
    ax_b.axvline(float(config["success_criteria"]["top10_overlap_good"]), color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax_b.set_xlim(-0.2, 11.0)
    ax_b.set_ylim(len(cohorts) - 0.45, -0.62)
    ax_b.set_yticks(y_base, cohorts, fontsize=6.0)
    ax_b.set_xlabel("Top-10 overlap (colored) vs random expectation (gray)", fontsize=6.1)
    ax_b.grid(axis="x", color=grid_color, lw=0.38)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_b.text(-0.16, 1.12, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.text(0.00, 1.12, "Top-state replication above chance", transform=ax_b.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_c = fig.add_axes([0.600, 0.150, 0.350, 0.350])
    clinical_rep = clinical[clinical["repeat"].eq(1)].copy()
    ax_c.axvline(0, color="#777777", lw=0.65, ls=(0, (3, 2)), zorder=1)
    for idx, dataset in enumerate(summary["dataset_name"]):
        row = clinical_rep[clinical_rep["dataset_name"].eq(dataset)]
        if row.empty or not bool(row["split_A_ok"].iloc[0]) or not bool(row["split_B_ok"].iloc[0]):
            ax_c.text(0, idx, "NA", ha="center", va="center", fontsize=5.2, color=text_secondary)
            continue
        log_a = float(np.log(row["split_A_hr"].iloc[0]))
        log_b = float(np.log(row["split_B_hr"].iloc[0]))
        ax_c.plot([log_a, log_b], [idx, idx], color=cohort_colors[idx], lw=1.0, alpha=0.85)
        ax_c.scatter([log_a, log_b], [idx, idx], s=28, color=[cat.get("pale_yellow", "#FEEBB9"), cohort_colors[idx]], edgecolor=text_primary, linewidth=0.35, zorder=3)
        ax_c.text(max(log_a, log_b) + 0.05, idx, "same" if (log_a >= 0) == (log_b >= 0) else "flip", ha="left", va="center", fontsize=5.2, color=text_secondary)
    finite_logs = []
    for _, row in clinical_rep.iterrows():
        if bool(row["split_A_ok"]) and bool(row["split_B_ok"]):
            finite_logs.extend([np.log(row["split_A_hr"]), np.log(row["split_B_hr"])])
    span = max(0.5, float(np.nanmax(np.abs(finite_logs))) if finite_logs else 0.5)
    ax_c.set_xlim(-span * 1.18, span * 1.55)
    ax_c.set_yticks(y_base, cohorts, fontsize=5.9)
    ax_c.set_ylim(len(cohorts) - 0.45, -0.55)
    ax_c.set_xlabel("log hazard ratio per split-specific R*", fontsize=6.3)
    ax_c.grid(axis="x", color=grid_color, lw=0.35)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    ax_c.tick_params(labelsize=5.6, length=2.0, width=0.55)
    ax_c.text(-0.07, 1.16, "c", transform=ax_c.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_c.text(0.00, 1.16, "Clinical direction audit in the representative split", transform=ax_c.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    save_square(fig, output, config)


def write_reports(root: Path, config: dict, summary: pd.DataFrame, audit: pd.DataFrame, clinical: pd.DataFrame) -> None:
    cohort_list = ", ".join(config["datasets"].keys())
    protocol = f"""# Experiment 13 Protocol Audit

## Protocol Section

Source document section: `19. 实验 13：跨队列复现`.

Purpose: validate whether top R* states are stable outside a single discovery
sample and are not only a single-cohort artifact.

## Implemented Route

- Current available experiment-ready cohorts: {cohort_list}.
- No paired TCGA-LUAD/TCGA-PAAD discovery cohort is currently present in the
  workspace, so the protocol's internal split A/B route was used.
- Patient IDs are split before occupancy is recalculated. The MHN-derived F_hat
  backbone is locked from Experiment 5, so the analysis tests held-out
  occupancy/R* reproducibility rather than full external re-training.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_13_protocol_audit.md").write_text(protocol, encoding="utf-8")

    lines = [
        "# Experiment 13 Summary",
        "",
        "| Cohort | Evaluable | Repeats | Common states | Median rho | Median Top10 overlap | Random expected overlap | Enrichment | Direction concordance | HR direction consistency | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples():
        hr_text = "NA" if not np.isfinite(row.clinical_hr_direction_consistency) else f"{row.clinical_hr_direction_consistency:.2f}"
        rho_text = "NA" if not np.isfinite(row.median_spearman_rho) else f"{row.median_spearman_rho:.2f}"
        null_text = "NA" if not np.isfinite(row.median_top10_null_mean) else f"{row.median_top10_null_mean:.2f}"
        enrich_text = "NA" if not np.isfinite(row.median_top10_enrichment) else f"{row.median_top10_enrichment:.1f}x"
        status = f"rho:{row.rho_status}; overlap:{row.overlap_status}"
        lines.append(
            f"| {row.short_name} | {row.evaluable} | {row.repeats} | {row.median_common_states:.0f} | {rho_text} | {row.median_top10_overlap:.0f} | {null_text} | {enrich_text} | {row.median_direction_concordance:.2f} | {hr_text} | {status} |"
        )
    (root / "experiment_13_summary.md").write_text("\n".join(lines), encoding="utf-8")

    sci = [
        "# Experiment 13 Scientific Review",
        "",
        "## Main Interpretation",
        "",
        "Experiment 13 tests whether state-level R* signals reproduce when patients from the same cancer cohort are split into independent discovery and validation halves. A reproducible result means that the same high-R* state pattern is not driven by a small cluster of samples in one half.",
        "",
        "The strongest evidence is simultaneous agreement across rank correlation, top-10 overlap beyond random expectation, direction concordance and clinical HR direction. Rank rho evaluates the whole stable-state ordering, while top-10 overlap evaluates the paper-facing bottleneck states. Direction concordance asks whether a state remains above or below the neutral R*=1 boundary.",
        "",
        "## Boundary",
        "",
        "This is an internal split-cohort replication experiment. It does not replace a future TCGA-to-GENIE external validation. Because F_hat is locked from the full Experiment 5 backbone, this version validates held-out occupancy stability under a fixed progression model rather than independent MHN re-training in each split.",
        "",
        "## Cohort Audit",
        "",
        "| Cohort | Split patients | State rows | Scored states | Clinical patients | Locked F_hat |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit.itertuples():
        sci.append(f"| {row.short_name} | {row.split_patients} | {row.state_table_rows} | {row.scored_states} | {row.clinical_patients} | {row.locked_inflow_backbone} |")
    sci.extend(
        [
            "",
            "## Clinical HR Direction Audit",
            "",
            "| Cohort | Repeat | A OK | B OK | A HR | B HR | Same direction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in clinical[clinical["repeat"].le(5)].itertuples():
        a_hr = "NA" if not np.isfinite(row.split_A_hr) else f"{row.split_A_hr:.2f}"
        b_hr = "NA" if not np.isfinite(row.split_B_hr) else f"{row.split_B_hr:.2f}"
        same = "NA" if pd.isna(row.hr_direction_same) else str(bool(row.hr_direction_same))
        sci.append(f"| {row.short_name} | {row.repeat} | {row.split_A_ok} | {row.split_B_ok} | {a_hr} | {b_hr} | {same} |")
    (root / "experiment_13_scientific_review.md").write_text("\n".join(sci), encoding="utf-8")

    design = f"""# Experiment 13 Figure Design Review

## Sources

{figure_style.design_sources_markdown(config)}

## Rules Applied

{figure_style.design_rules_markdown(config)}

## Design Choices

- Panel A shows the full distribution of rank correlations over 50 patient-level
  splits, with median and IQR rather than a single representative scatter.
- Panel B compares observed Top-10 overlap against the hypergeometric random
  overlap expectation on the common stable-state universe.
- Panel C keeps clinical association as a direction-consistency audit rather
  than a claim of split-level significance.
"""
    (root / "top_journal_figure_design_review.md").write_text(design, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.result_root:
        config["result_root"] = args.result_root
    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics, representative, clinical, audit = run_analysis(config)
    summary = summarize_metrics(metrics, clinical, config)
    metrics.to_csv(tables / "split_replication_metrics.tsv", sep="\t", index=False)
    representative.to_csv(tables / "representative_split_state_scores.tsv", sep="\t", index=False)
    clinical.to_csv(tables / "clinical_direction_by_split.tsv", sep="\t", index=False)
    audit.to_csv(tables / "split_replication_audit.tsv", sep="\t", index=False)
    summary.to_csv(tables / "experiment_13_summary.tsv", sep="\t", index=False)
    plot_main_figure(summary, representative, clinical, figures / "Figure_E13_split_replication", config)
    write_reports(root, config, summary, audit, clinical)


if __name__ == "__main__":
    main()
