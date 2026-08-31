"""Run Experiment 14: ablation and backbone replacement.

This experiment tests whether the MHN-derived inflow component is necessary.
It compares the full Rel-ObsTQ-MHN score against occupancy-only, uniform inflow,
event-frequency inflow and shuffled-MHN backbones on real cohorts. A simulation
positive-control digest from Experiment 6 is included as a supporting panel.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right, insort
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from statsmodels.duration.hazard_regression import PHReg

import figure_style


CONFIG_PATH = Path("configs/experiment_14.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 14.")
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


def genotype_events(genotype: str) -> list[str]:
    return [] if genotype == "WT" or not str(genotype).strip() else str(genotype).split("+")


def read_events(dataset: str, config: dict) -> list[str]:
    theta_path = Path(config["experiment_03_root"]) / dataset / "tables" / "theta.tsv"
    theta = pd.read_csv(theta_path, sep="\t", index_col=0)
    return theta.columns.astype(str).tolist()


def read_inputs(dataset: str, config: dict) -> dict[str, pd.DataFrame | list[str]]:
    exp4 = Path(config["experiment_04_root"]) / dataset / "tables"
    occupancy = pd.read_csv(exp4 / "state_occupancy_experiment4.tsv", sep="\t")
    inflow = pd.read_csv(exp4 / "inflow_table_rule_a_one_step.tsv", sep="\t")
    edges = pd.read_csv(exp4 / "predecessor_edges_rule_a_one_step.tsv", sep="\t")
    scores = pd.read_csv(Path(config["experiment_05_root"]) / dataset / "tables" / "state_scores.tsv", sep="\t")
    state_table = pd.read_csv(Path(config["experiment_ready_root"]) / dataset / "state_table.csv")
    clinical = pd.read_csv(Path(config["experiment_12_root"]) / "tables" / "patient_clinical_scores.tsv", sep="\t")
    clinical = clinical[clinical["dataset_name"].eq(dataset)].copy()
    for frame in [occupancy, inflow, scores]:
        frame["canonical_state"] = frame["state"].map(canonical_state)
    state_table["canonical_state"] = state_table["state_id"].map(canonical_state)
    clinical["canonical_state"] = clinical["canonical_state"].map(canonical_state)
    return {
        "occupancy": occupancy,
        "inflow": inflow,
        "edges": edges,
        "scores": scores,
        "state_table": state_table,
        "clinical": clinical,
        "events": read_events(dataset, config),
    }


def event_frequencies(state_table: pd.DataFrame, events: list[str]) -> dict[str, float]:
    usable = state_table[state_table["usable_for_relobstq"].astype("boolean").fillna(False)]
    total = max(len(usable), 1)
    frequencies = {}
    signatures = usable["genotype_signature"].fillna("WT").astype(str)
    for event in events:
        frequencies[event] = float(signatures.map(lambda text: event in set(genotype_events(text))).sum() / total)
    floor = 1.0 / (10.0 * total)
    return {event: max(value, floor) for event, value in frequencies.items()}


def prepare_edges(edges: pd.DataFrame, events: list[str], event_freq: dict[str, float], seed: int) -> pd.DataFrame:
    work = edges.copy()
    p = len(events)
    event_set = set(events)
    uniform_probs = []
    frequency_probs = []
    for row in work.itertuples():
        source_genotype = str(row.source_state).split("::", 1)[1]
        present = set(genotype_events(source_genotype))
        absent = sorted(event_set.difference(present))
        uniform_probs.append(1.0 / max(p - len(present), 1))
        denom = sum(event_freq[event] for event in absent)
        frequency_probs.append(event_freq.get(str(row.event_added), 0.0) / denom if denom > 0 else 1.0 / max(len(absent), 1))
    rng = np.random.default_rng(seed)
    shuffled = work["edge_probability"].to_numpy(dtype=float).copy()
    rng.shuffle(shuffled)
    work["prob_full_mhn"] = work["edge_probability"].astype(float)
    work["prob_uniform_inflow"] = uniform_probs
    work["prob_frequency_inflow"] = frequency_probs
    work["prob_shuffled_mhn"] = shuffled
    work["canonical_source"] = work["source_state"].map(canonical_state)
    work["canonical_target"] = work["target_state"].map(canonical_state)
    return work


def compute_inflow_from_edges(edges: pd.DataFrame, source_l: pd.Series, prob_column: str) -> pd.Series:
    frame = edges[["canonical_source", "canonical_target", prob_column]].copy()
    frame["source_L"] = frame["canonical_source"].map(source_l).fillna(0.0)
    frame["contribution"] = frame["source_L"] * frame[prob_column].astype(float)
    return frame.groupby("canonical_target")["contribution"].sum()


def normalize_score(frame: pd.DataFrame, score_col: str, eligible: pd.Series) -> pd.Series:
    values = pd.to_numeric(frame[score_col], errors="coerce")
    normalizer = values[eligible].median()
    if not np.isfinite(normalizer) or normalizer <= 0:
        return pd.Series(np.nan, index=frame.index)
    return values / float(normalizer)


def build_variant_scores(dataset: str, config: dict) -> pd.DataFrame:
    inputs = read_inputs(dataset, config)
    occupancy = inputs["occupancy"]
    inflow = inputs["inflow"]
    state_table = inputs["state_table"]
    events = inputs["events"]
    edges = prepare_edges(inputs["edges"], events, event_frequencies(state_table, events), int(config["analysis"]["random_seed"]) + sum(ord(c) for c in dataset))
    thresholds = config["analysis"]
    minimum_count = int(thresholds["minimum_state_count"])
    minimum_inflow = float(thresholds["minimum_inflow"])
    epsilon = float(thresholds["epsilon"])
    base = occupancy[["canonical_state", "state", "stage", "genotype", "event_count", "N_v", "L_v"]].copy()
    base = base.merge(inflow[["canonical_state", "F_hat"]], on="canonical_state", how="left")
    source_l = base.set_index("canonical_state")["L_v"]
    base["F_uniform_inflow"] = base["canonical_state"].map(compute_inflow_from_edges(edges, source_l, "prob_uniform_inflow")).fillna(0.0)
    base["F_frequency_inflow"] = base["canonical_state"].map(compute_inflow_from_edges(edges, source_l, "prob_frequency_inflow")).fillna(0.0)
    base["F_shuffled_mhn"] = base["canonical_state"].map(compute_inflow_from_edges(edges, source_l, "prob_shuffled_mhn")).fillna(0.0)
    rows = []
    for variant in config["variants"]:
        work = base.copy()
        if variant == "occupancy_only":
            work["raw_score"] = work["L_v"]
            eligible = work["N_v"].ge(minimum_count)
            work["inflow_value"] = np.nan
        else:
            f_col = "F_hat" if variant == "full_mhn" else f"F_{variant}"
            work["inflow_value"] = work[f_col]
            work["raw_score"] = work["L_v"] / (work["inflow_value"] + epsilon)
            eligible = work["N_v"].ge(minimum_count) & work["inflow_value"].ge(minimum_inflow)
        work["variant"] = variant
        work["variant_display"] = config["variants"][variant]["display_name"]
        work["eligible_variant"] = eligible
        work["score"] = normalize_score(work, "raw_score", eligible)
        work["log_score"] = np.log(np.clip(work["score"], 1.0e-12, None))
        work["log2_score"] = np.log2(np.clip(work["score"], 1.0e-12, None))
        work["dataset_name"] = dataset
        rows.append(work)
    return pd.concat(rows, ignore_index=True)


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
    return (x - x.mean()) / x.std(ddof=0), keep


def harrell_c_index(time: pd.Series, event: pd.Series, risk: pd.Series) -> float:
    df = pd.DataFrame({"time": time, "event": event, "risk": risk}).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return np.nan
    df = df.sort_values("time", ascending=False)
    later_risks: list[float] = []
    concordant = 0.0
    comparable = 0
    for _, group in df.groupby("time", sort=False):
        events = group[group["event"].eq(1)]
        for value in events["risk"].astype(float):
            lower = bisect_left(later_risks, value)
            equal_low = bisect_left(later_risks, value)
            equal_high = bisect_right(later_risks, value)
            concordant += lower + 0.5 * (equal_high - equal_low)
            comparable += len(later_risks)
        for value in group["risk"].astype(float):
            insort(later_risks, value)
    return float(concordant / comparable) if comparable else np.nan


def fit_clinical_model(clinical: pd.DataFrame, scores: pd.DataFrame, dataset: str, short: str, variant: str, config: dict) -> dict:
    score_map = scores[scores["variant"].eq(variant)].set_index("canonical_state")["score"]
    frame = clinical.copy()
    frame["variant_score"] = frame["canonical_state"].map(score_map)
    frame["log_variant_score"] = np.log(np.clip(pd.to_numeric(frame["variant_score"], errors="coerce"), 1.0e-12, None))
    frame = frame[
        frame["followup_time_days"].gt(30)
        & frame["survival_event_binary"].isin([0, 1])
        & frame["variant_score"].notna()
    ].copy()
    result = {
        "dataset_name": dataset,
        "short_name": short,
        "variant": variant,
        "variant_display": config["variants"][variant]["display_name"],
        "n": int(len(frame)),
        "events": int(frame["survival_event_binary"].sum()) if len(frame) else 0,
        "ok": False,
        "hr": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "beta": np.nan,
        "c_index": np.nan,
        "reason": "",
    }
    if len(frame) < 20 or result["events"] < int(config["analysis"]["minimum_events_for_cox"]):
        result["reason"] = "insufficient patients/events"
        return result
    candidates = ["log_variant_score", "age_num", "sex_male", "stage_metastatic", "event_count"]
    attempts = []
    while candidates:
        x, used = standardize_features(frame, candidates)
        if "log_variant_score" not in used:
            result["reason"] = "score unavailable after feature filtering"
            return result
        y = frame.loc[x.index]
        try:
            fit = PHReg(y["followup_time_days"], x, status=y["survival_event_binary"]).fit(disp=0)
            params = pd.Series(fit.params, index=used)
            se = pd.Series(fit.bse, index=used)
            beta = float(params["log_variant_score"])
            se_beta = float(se["log_variant_score"])
            risk = pd.Series(np.dot(x.to_numpy(dtype=float), fit.params), index=x.index)
            result.update(
                {
                    "ok": True,
                    "hr": float(np.exp(beta)),
                    "ci_low": float(np.exp(beta - 1.96 * se_beta)),
                    "ci_high": float(np.exp(beta + 1.96 * se_beta)),
                    "beta": beta,
                    "c_index": harrell_c_index(y["followup_time_days"], y["survival_event_binary"], risk),
                    "features_used": ";".join(used),
                    "reason": "OK",
                }
            )
            return result
        except Exception as exc:
            attempts.append(f"{type(exc).__name__}: {exc}")
            removable = [feature for feature in ["stage_metastatic", "sex_male", "event_count", "age_num"] if feature in candidates]
            if not removable:
                result["reason"] = "; ".join(attempts[-3:])
                return result
            candidates.remove(removable[0])
    result["reason"] = "no estimable model"
    return result


def split_top_overlap(state_table: pd.DataFrame, variant_scores: pd.DataFrame, dataset: str, short: str, config: dict) -> pd.DataFrame:
    rows = []
    patients = np.array(sorted(state_table["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]) + 1414 + sum(ord(c) for c in dataset))
    repeats = int(config["analysis"]["split_repeats"])
    top_k = int(config["analysis"]["top_k"])
    for repeat in range(1, repeats + 1):
        ids = patients.copy()
        rng.shuffle(ids)
        cut = min(max(int(round(len(ids) * float(config["analysis"]["split_fraction"]))), 1), len(ids) - 1)
        a_ids = set(ids[:cut])
        b_ids = set(ids[cut:])
        for variant in config["variants"]:
            full = variant_scores[variant_scores["variant"].eq(variant)].copy()
            a_counts = state_table[state_table["patient_id"].astype(str).isin(a_ids)].groupby("canonical_state").size()
            b_counts = state_table[state_table["patient_id"].astype(str).isin(b_ids)].groupby("canonical_state").size()
            a = full.copy()
            b = full.copy()
            a["N_split"] = a["canonical_state"].map(a_counts).fillna(0).astype(int)
            b["N_split"] = b["canonical_state"].map(b_counts).fillna(0).astype(int)
            a["split_score"] = a["score"] * (a["N_split"].ge(int(config["analysis"]["minimum_state_count"])))
            b["split_score"] = b["score"] * (b["N_split"].ge(int(config["analysis"]["minimum_state_count"])))
            top_a = set(a[a["split_score"].gt(0)].nlargest(top_k, "split_score")["canonical_state"])
            top_b = set(b[b["split_score"].gt(0)].nlargest(top_k, "split_score")["canonical_state"])
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "variant": variant,
                    "variant_display": config["variants"][variant]["display_name"],
                    "repeat": repeat,
                    "top_overlap": int(len(top_a.intersection(top_b))),
                    "top_A_count": int(len(top_a)),
                    "top_B_count": int(len(top_b)),
                }
            )
    return pd.DataFrame(rows)


def summarize_real(clinical: pd.DataFrame, stability: pd.DataFrame, config: dict) -> pd.DataFrame:
    full = clinical[clinical["variant"].eq("full_mhn")][["dataset_name", "c_index"]].rename(columns={"c_index": "full_c_index"})
    clinical = clinical.merge(full, on="dataset_name", how="left")
    clinical["delta_c_index_vs_full"] = clinical["c_index"] - clinical["full_c_index"]
    stability_summary = (
        stability.groupby(["dataset_name", "variant"], as_index=False)
        .agg(median_top_overlap=("top_overlap", "median"), iqr_top_overlap=("top_overlap", lambda s: s.quantile(0.75) - s.quantile(0.25)))
    )
    return clinical.merge(stability_summary, on=["dataset_name", "variant"], how="left")


def load_simulation_digest(config: dict) -> pd.DataFrame:
    path = Path(config["experiment_06_root"]) / "tables" / "performance_summary_table.tsv"
    table = pd.read_csv(path, sep="\t")
    keep = table[table["endpoint"].isin(["Spearman", "Bottleneck ROC AUC", "Top-5 precision", "Recall@5"])].copy()
    return keep


def eligible_scores(scores: pd.DataFrame, dataset: str, variant: str) -> pd.DataFrame:
    frame = scores[scores["dataset_name"].eq(dataset) & scores["variant"].eq(variant)].copy()
    frame = frame[frame["eligible_variant"].astype(bool)]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame[frame["score"].notna() & frame["score"].gt(0)]
    return frame


def build_relative_dwell_ablation_tables(scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_k = int(config["analysis"]["top_k"])
    decomposition_rows = []
    retention_rows = []
    rank_lift_rows = []
    for dataset, ds_cfg in config["datasets"].items():
        short = ds_cfg["short_name"]
        full = eligible_scores(scores, dataset, "full_mhn")
        if full.empty:
            continue
        full = full.sort_values("score", ascending=False).reset_index(drop=True)
        n_eligible = int(full["canonical_state"].nunique())
        random_expected = float(top_k * top_k / max(n_eligible, 1))
        top_full = full.head(top_k).copy()
        top_full["state_rank"] = np.arange(1, len(top_full) + 1)
        for row in top_full.itertuples():
            decomposition_rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "state_rank": int(row.state_rank),
                    "canonical_state": row.canonical_state,
                    "state": row.state,
                    "event_count": int(row.event_count),
                    "N_v": int(row.N_v),
                    "L_v": float(row.L_v),
                    "F_MHN": float(row.inflow_value),
                    "R_raw": float(row.raw_score),
                    "R_star": float(row.score),
                    "log10_L_v": float(np.log10(max(float(row.L_v), 1.0e-12))),
                    "log10_F_MHN": float(np.log10(max(float(row.inflow_value), 1.0e-12))),
                    "log10_R_raw": float(np.log10(max(float(row.raw_score), 1.0e-12))),
                }
            )
        full_top_set = set(top_full["canonical_state"])
        for variant in config["variants"]:
            if variant == "full_mhn":
                continue
            variant_frame = eligible_scores(scores, dataset, variant).sort_values("score", ascending=False).head(top_k)
            variant_top = set(variant_frame["canonical_state"])
            union = full_top_set.union(variant_top)
            overlap = len(full_top_set.intersection(variant_top))
            retention_rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "variant": variant,
                    "variant_display": config["variants"][variant]["display_name"],
                    "top_k": top_k,
                    "eligible_states": n_eligible,
                    "retained_top_k": int(overlap),
                    "random_expected_overlap": random_expected,
                    "enrichment_vs_random": float(overlap / random_expected) if random_expected > 0 else np.nan,
                    "retention_fraction": float(overlap / max(len(full_top_set), 1)),
                    "jaccard": float(overlap / len(union)) if union else np.nan,
                }
            )
        rank_frame = full.copy()
        n_states = len(rank_frame)
        denom = max(n_states - 1, 1)
        rank_frame["rank_R"] = rank_frame["score"].rank(ascending=False, method="min")
        rank_frame["rank_L"] = rank_frame["L_v"].rank(ascending=False, method="min")
        rank_frame["percentile_R"] = 100.0 * (1.0 - (rank_frame["rank_R"] - 1.0) / denom)
        rank_frame["percentile_L"] = 100.0 * (1.0 - (rank_frame["rank_L"] - 1.0) / denom)
        rank_frame["rank_lift_percentile"] = rank_frame["percentile_R"] - rank_frame["percentile_L"]
        top_rank = rank_frame[rank_frame["canonical_state"].isin(full_top_set)].copy()
        for row in top_rank.sort_values("rank_R").itertuples():
            rank_lift_rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "canonical_state": row.canonical_state,
                    "state": row.state,
                    "rank_R": int(row.rank_R),
                    "rank_L": int(row.rank_L),
                    "percentile_R": float(row.percentile_R),
                    "percentile_L": float(row.percentile_L),
                    "rank_lift_percentile": float(row.rank_lift_percentile),
                    "R_star": float(row.score),
                    "L_v": float(row.L_v),
                    "F_MHN": float(row.inflow_value),
                }
            )
    return pd.DataFrame(decomposition_rows), pd.DataFrame(retention_rows), pd.DataFrame(rank_lift_rows)


def run_analysis(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_scores = []
    clinical_rows = []
    stability_rows = []
    audit_rows = []
    for dataset, ds_cfg in config["datasets"].items():
        short = ds_cfg["short_name"]
        scores = build_variant_scores(dataset, config)
        all_scores.append(scores)
        inputs = read_inputs(dataset, config)
        clinical = inputs["clinical"]
        state_table = inputs["state_table"]
        for variant in config["variants"]:
            clinical_rows.append(fit_clinical_model(clinical, scores, dataset, short, variant, config))
        stability_rows.append(split_top_overlap(state_table, scores, dataset, short, config))
        audit_rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "states": int(scores["canonical_state"].nunique()),
                "patients": int(state_table["patient_id"].nunique()),
                "clinical_patients": int(clinical["patient_id"].nunique()),
            }
        )
    clinical = pd.DataFrame(clinical_rows)
    stability = pd.concat(stability_rows, ignore_index=True)
    summary = summarize_real(clinical, stability, config)
    return pd.concat(all_scores, ignore_index=True), clinical, stability, summary, pd.DataFrame(audit_rows)


def save_square(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def plot_main_figure(
    scores: pd.DataFrame,
    decomposition: pd.DataFrame,
    retention: pd.DataFrame,
    rank_lift: pd.DataFrame,
    simulation: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = [cat.get("lavender", "#B5AED5"), cat.get("sky_blue", "#B2E6FD"), cat.get("sage", "#B8D2CC"), cat.get("coral", "#E8B2A7")]
    color_by_short = {cfg["short_name"]: color for color, cfg in zip(cohort_colors, config["datasets"].values())}
    top_k = int(config["analysis"]["top_k"])

    fig = plt.figure(figsize=(7.2, 7.2))
    fig.text(0.075, 0.972, "Experiment 14 | Ablation and backbone replacement", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.075, 0.947, "Ablating the expected-inflow denominator of the relative dwell-time signal R* = L / F.", ha="left", va="top", fontsize=5.9, color=text_secondary)

    ax_a = fig.add_axes([0.075, 0.585, 0.385, 0.300])
    full_states = scores[
        scores["variant"].eq("full_mhn")
        & scores["eligible_variant"].astype(bool)
        & pd.to_numeric(scores["L_v"], errors="coerce").gt(0)
        & pd.to_numeric(scores["inflow_value"], errors="coerce").gt(0)
    ].copy()
    full_states["log10_F_MHN"] = np.log10(pd.to_numeric(full_states["inflow_value"], errors="coerce"))
    full_states["log10_L_v"] = np.log10(pd.to_numeric(full_states["L_v"], errors="coerce"))
    full_states = full_states.replace([np.inf, -np.inf], np.nan).dropna(subset=["log10_F_MHN", "log10_L_v"])
    ax_a.scatter(full_states["log10_F_MHN"], full_states["log10_L_v"], s=3.0, color="#C9D1D3", alpha=0.26, linewidth=0, rasterized=True)
    for dataset, ds_cfg in config["datasets"].items():
        short = ds_cfg["short_name"]
        color = color_by_short[short]
        sub = decomposition[decomposition["dataset_name"].eq(dataset)].sort_values("state_rank")
        ax_a.scatter(sub["log10_F_MHN"], sub["log10_L_v"], s=17, color=color, edgecolor=text_primary, linewidth=0.28, alpha=0.96, zorder=4, label=short)
    combined = pd.concat([full_states[["log10_F_MHN", "log10_L_v"]], decomposition[["log10_F_MHN", "log10_L_v"]]], ignore_index=True)
    lo = float(np.floor(np.nanmin(combined.to_numpy()) * 2.0) / 2.0 - 0.05)
    hi = float(np.ceil(np.nanmax(combined.to_numpy()) * 2.0) / 2.0 + 0.05)
    ax_a.plot([lo, hi], [lo, hi], color="#777777", lw=0.65, ls=(0, (3, 2)), zorder=1)
    ax_a.text(lo + 0.15, lo + 0.35, "L = F_MHN", fontsize=4.8, color=text_secondary, rotation=34, ha="left", va="bottom")
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Expected inflow log10(F_MHN)", fontsize=6.3)
    ax_a.set_ylabel("Observed dwell log10(L)", fontsize=6.3)
    ax_a.grid(color=grid_color, lw=0.35)
    ax_a.legend(loc="lower right", frameon=False, fontsize=4.9, handletextpad=0.25, borderpad=0.1, markerscale=0.8)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)
    ax_a.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_a.text(-0.14, 1.12, "a", transform=ax_a.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_a.text(0.00, 1.12, "Observed versus expected dwell", transform=ax_a.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_b = fig.add_axes([0.590, 0.585, 0.350, 0.300])
    variant_order = [variant for variant in config["variants"] if variant != "full_mhn"]
    variant_labels = [config["variants"][variant]["display_name"] for variant in variant_order]
    cohort_order = list(config["datasets"])
    cohort_labels = [config["datasets"][dataset]["short_name"] for dataset in cohort_order]
    variant_short = {
        "occupancy_only": "Occ.",
        "uniform_inflow": "Unif.",
        "frequency_inflow": "Freq.",
        "shuffled_mhn": "Shuf.",
    }
    rows = []
    y_cursor = 0.0
    for dataset in cohort_order:
        short = config["datasets"][dataset]["short_name"]
        sub = retention[retention["dataset_name"].eq(dataset)].set_index("variant")
        for variant_index, variant in enumerate(variant_order):
            row = sub.loc[variant]
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "variant": variant,
                    "variant_short": f"{short}  {variant_short[variant]}" if variant_index == 0 else f"      {variant_short[variant]}",
                    "observed": float(row["retained_top_k"]),
                    "expected": float(row["random_expected_overlap"]),
                    "enrichment": float(row["enrichment_vs_random"]),
                    "color": color_by_short[short],
                    "y": y_cursor,
                }
            )
            y_cursor += 1.0
        if dataset != cohort_order[-1]:
            ax_b.axhline(y_cursor - 0.45, color=grid_color, lw=0.42)
            y_cursor += 0.42
    for row in rows:
        y = row["y"]
        obs = row["observed"]
        exp = row["expected"]
        ax_b.plot([exp, obs], [y, y], color=row["color"], lw=0.8, alpha=0.95, solid_capstyle="round", zorder=2)
        ax_b.scatter(exp, y, s=12, facecolor="white", edgecolor="#6F777A", linewidth=0.45, zorder=3)
        ax_b.scatter(obs, y, s=18, facecolor=row["color"], edgecolor=text_primary, linewidth=0.28, zorder=4)
        ax_b.text(10.22, y, f"{int(obs)}/{top_k}", ha="left", va="center", fontsize=4.7, color=text_secondary)
    ax_b.axvline(0, color="#B9C1C3", lw=0.45)
    ax_b.axvline(top_k, color="#B9C1C3", lw=0.45)
    ax_b.set_xlim(-0.55, 10.95)
    ax_b.set_ylim(y_cursor - 0.42, -0.58)
    ax_b.set_yticks([row["y"] for row in rows], [row["variant_short"] for row in rows], fontsize=4.8)
    ax_b.set_xticks([0, 2, 4, 6, 8, 10])
    ax_b.set_xlabel(f"Retained Full R* Top-{top_k} states", fontsize=6.3)
    ax_b.grid(axis="x", color=grid_color, lw=0.35)
    ax_b.text(0.995, 1.015, "open: random expectation; filled: observed", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=4.6, color=text_secondary)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_b.text(-0.16, 1.12, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.text(0.00, 1.12, "Top-state overlap above random expectation", transform=ax_b.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_c = fig.add_axes([0.075, 0.170, 0.385, 0.300])
    rng = np.random.default_rng(14014)
    lift_values = []
    for idx, (dataset, ds_cfg) in enumerate(config["datasets"].items()):
        short = ds_cfg["short_name"]
        color = color_by_short[short]
        sub = rank_lift[rank_lift["dataset_name"].eq(dataset)].sort_values("rank_R")
        values = sub["rank_lift_percentile"].to_numpy(dtype=float)
        lift_values.extend(values.tolist())
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax_c.scatter(np.full(len(values), idx) + jitter, values, s=13, color=color, edgecolor=text_primary, linewidth=0.25, alpha=0.92, zorder=3)
        if len(values):
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax_c.vlines(idx, q1, q3, color=text_primary, lw=0.75, zorder=4)
            ax_c.hlines(med, idx - 0.22, idx + 0.22, color=text_primary, lw=1.0, zorder=5)
            ax_c.text(idx, max(values) + 1.6, f"{med:.1f}", ha="center", va="bottom", fontsize=5.0, color=text_secondary)
    ax_c.axhline(0, color="#777777", lw=0.65, ls=(0, (3, 2)))
    ax_c.set_xticks(np.arange(len(cohort_labels)), cohort_labels, fontsize=5.8)
    ax_c.set_ylabel("R* percentile minus L percentile", fontsize=6.3)
    if lift_values:
        ymin = min(-5.0, float(np.nanmin(lift_values)) - 6.0)
        ymax = max(10.0, float(np.nanmax(lift_values)) + 8.0)
        ax_c.set_ylim(ymin, ymax)
    ax_c.grid(axis="y", color=grid_color, lw=0.38)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    ax_c.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_c.text(-0.14, 1.12, "c", transform=ax_c.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_c.text(0.00, 1.12, "Rank lift after MHN inflow correction", transform=ax_c.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_d = fig.add_axes([0.590, 0.170, 0.350, 0.300])
    endpoints = ["Spearman", "Bottleneck ROC AUC", "Top-5 precision", "Recall@5"]
    y = np.arange(len(endpoints))
    sim = simulation.set_index("endpoint").reindex(endpoints)
    for idx, endpoint in enumerate(endpoints):
        occ = float(sim.loc[endpoint, "occupancy_median"])
        full = float(sim.loc[endpoint, "R_star_median"])
        ax_d.plot([occ, full], [idx, idx], color=text_primary, lw=0.9, zorder=2)
        ax_d.scatter(occ, idx, s=26, color=cat.get("pale_yellow", "#FEEBB9"), edgecolor=text_primary, linewidth=0.35, zorder=3)
        ax_d.scatter(full, idx, s=30, color=cat.get("lavender", "#B5AED5"), edgecolor=text_primary, linewidth=0.35, zorder=3)
        ax_d.text(full + 0.035, idx, f"+{full - occ:.2f}", ha="left", va="center", fontsize=5.2, color=text_secondary)
    ax_d.set_yticks(y, ["Spearman", "AUC", "Top5", "Recall@5"], fontsize=5.9)
    ax_d.set_xlim(0, 1.08)
    ax_d.set_xlabel("Simulation recovery metric", fontsize=6.3)
    ax_d.grid(axis="x", color=grid_color, lw=0.38)
    ax_d.invert_yaxis()
    for spine in ["top", "right"]:
        ax_d.spines[spine].set_visible(False)
    ax_d.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_d.text(-0.16, 1.12, "d", transform=ax_d.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_d.text(0.00, 1.12, "Simulation positive control", transform=ax_d.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)
    ax_d.text(0.02, -0.20, "Yellow: occupancy; lavender: Full R*.", transform=ax_d.transAxes, ha="left", va="top", fontsize=5.0, color=text_secondary)
    save_square(fig, output, config)


def write_reports(
    root: Path,
    config: dict,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    simulation: pd.DataFrame,
    decomposition: pd.DataFrame,
    retention: pd.DataFrame,
    rank_lift: pd.DataFrame,
) -> None:
    protocol = f"""# Experiment 14 Protocol Audit

## Protocol Section

Source document section: `20. 实验 14：消融与骨架替换`.

Purpose: prove that the MHN-derived inflow is a necessary component of R*, while
also showing that the innovation is not equivalent to MHN alone.

## Implemented Variants

- Full Rel-ObsTQ-MHN: R=L/F_hat_MHN.
- Occupancy-only: score=L.
- Uniform inflow: one-step predecessor inflow with uniform next-event probabilities.
- Frequency inflow: one-step predecessor inflow weighted by event marginal frequencies.
- Shuffled-MHN: one-step predecessor inflow after permuting MHN edge probabilities.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}

## Main-Figure Logic

- Panel A places all eligible full-model states on an observed-versus-expected
  dwell plane and highlights the Top-{config["analysis"]["top_k"]} R* states.
- Panel B replaces the denominator and asks how many full-model Top-{config["analysis"]["top_k"]}
  states are retained.
- Panel C quantifies how much the full relative dwell score promotes a state
  beyond its occupancy-only percentile.
- Panel D keeps the truth-based simulation positive control.
"""
    (root / "experiment_14_protocol_audit.md").write_text(protocol, encoding="utf-8")

    ret_pivot = retention.pivot(index="short_name", columns="variant_display", values="retained_top_k")
    lift_summary = rank_lift.groupby("short_name", as_index=False).agg(
        median_rank_lift=("rank_lift_percentile", "median"),
        max_rank_lift=("rank_lift_percentile", "max"),
    )
    lines = [
        "# Experiment 14 Summary",
        "",
        "## Core Relative-Dwell Ablation",
        "",
        "| Cohort | Occupancy retained | Uniform retained | Frequency retained | Shuffled retained | Median rank lift | Max rank lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in lift_summary.iterrows():
        short = row["short_name"]
        values = ret_pivot.loc[short] if short in ret_pivot.index else pd.Series(dtype=float)
        lines.append(
            f"| {short} | {int(values.get('Occupancy', 0))} | {int(values.get('Uniform', 0))} | {int(values.get('Frequency', 0))} | {int(values.get('Shuffled', 0))} | {row['median_rank_lift']:.1f} | {row['max_rank_lift']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Clinical Supplement",
            "",
            "| Cohort | Variant | C-index | Delta vs Full | HR | Split Top10 overlap |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples():
        lines.append(f"| {row.short_name} | {row.variant_display} | {row.c_index:.3f} | {row.delta_c_index_vs_full:+.3f} | {row.hr:.2f} | {row.median_top_overlap:.1f} |")
    (root / "experiment_14_summary.md").write_text("\n".join(lines), encoding="utf-8")

    sci = [
        "# Experiment 14 Scientific Review",
        "",
        "## Main Interpretation",
        "",
        "Experiment 14 evaluates whether the relative dwell-state signal depends on the MHN-derived inflow denominator. The main figure is now centered on the core innovation: relative dwell time, not clinical prediction. Panel A places all eligible states on the observed-versus-expected dwell plane and highlights the Top R* states above the `L = F_MHN` reference. Panel B compares denominator-replacement Top-state overlap against the random-set expectation. Panel C measures how much MHN inflow correction promotes states beyond occupancy alone. Panel D anchors the interpretation with truth-based simulation recovery.",
        "",
        "Uniform and frequency inflow variants test whether any generic inflow normalization is enough. Shuffled-MHN tests whether the learned transition structure matters beyond the marginal edge-probability distribution. Clinical C-index and HR are retained as supplements, but they are not treated as the primary evidence for the dwell-time innovation.",
        "",
        "## Denominator Replacement Summary",
        "",
        "| Cohort | Replacement | Full Top states retained | Jaccard |",
        "|---|---|---:|---:|",
    ]
    for row in retention.itertuples():
        sci.append(f"| {row.short_name} | {row.variant_display} | {row.retained_top_k}/{row.top_k} | {row.jaccard:.2f} |")
    sci.extend(
        [
            "",
            "## Rank-Lift Summary",
            "",
            "| Cohort | Median lift | IQR | Max lift |",
            "|---|---:|---:|---:|",
        ]
    )
    lift_stats = rank_lift.groupby("short_name")["rank_lift_percentile"].agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        max="max",
    )
    for short, row in lift_stats.iterrows():
        sci.append(f"| {short} | {row['median']:.1f} | {row['q75'] - row['q25']:.1f} | {row['max']:.1f} |")
    sci.extend(
        [
            "",
            "## Top Relative-Dwell State Decomposition",
            "",
            f"The decomposition table contains {len(decomposition)} top-state rows across the {len(config['datasets'])} selected cohorts. It records `L_v`, `F_MHN`, raw `L/F_MHN`, and normalized `R*` for each full-model top state.",
        "",
        "## Dataset Audit",
        "",
        "| Cohort | States | Patients | Clinical patients |",
        "|---|---:|---:|---:|",
        ]
    )
    for row in audit.itertuples():
        sci.append(f"| {row.short_name} | {row.states} | {row.patients} | {row.clinical_patients} |")
    sci.extend(
        [
            "",
            "## Simulation Positive Control",
            "",
            "| Endpoint | Full R* median | Occupancy median | Delta | Paired p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in simulation.itertuples():
        sci.append(f"| {row.endpoint} | {row.R_star_median:.3f} | {row.occupancy_median:.3f} | {row.paired_delta_median:+.3f} | {row.paired_p_value:.2g} |")
    sci.extend(
        [
            "",
            "## Boundary",
            "",
            "This experiment uses the existing real-cohort MHN backbone and does not claim that all alternative backbones have been exhaustively searched. The oMHN-backbone sensitivity specified in the long protocol is not implemented because an oMHN-trained backbone is not present in the current workspace.",
        ]
    )
    (root / "experiment_14_scientific_review.md").write_text("\n".join(sci), encoding="utf-8")

    design = f"""# Experiment 14 Figure Design Review

## Sources

{figure_style.design_sources_markdown(config)}

## Rules Applied

{figure_style.design_rules_markdown(config)}

## Design Choices

- Panels A-C were redesigned after scientific review to focus directly on the
  relative dwell-time mechanism instead of clinical supplement metrics.
- Panel A uses an observed-versus-expected scatter with an `L = F_MHN`
  reference line, avoiding the self-confirming appearance of a Top-state-only
  ratio decomposition.
- Panel B uses a random-expectation lollipop display: open points mark the
  expected Top-state overlap from random Top-{config["analysis"]["top_k"]} sets and filled
  points mark the observed overlap after denominator replacement.
- Panel C uses dot-plus-median summaries to show state-level rank promotion
  without hiding the individual top states.
- Panel D imports the controlled simulation positive-control summary from
  Experiment 6 to anchor the real-cohort ablation in truth-based recovery.
- All auxiliary per-state scores and per-repeat stability values are kept in
  tables rather than crowded into the main figure.
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
    scores, clinical, stability, summary, audit = run_analysis(config)
    simulation = load_simulation_digest(config)
    decomposition, retention, rank_lift = build_relative_dwell_ablation_tables(scores, config)
    scores.to_csv(tables / "variant_state_scores.tsv", sep="\t", index=False)
    clinical.to_csv(tables / "variant_clinical_results.tsv", sep="\t", index=False)
    stability.to_csv(tables / "variant_split_stability.tsv", sep="\t", index=False)
    summary.to_csv(tables / "experiment_14_summary.tsv", sep="\t", index=False)
    audit.to_csv(tables / "experiment_14_audit.tsv", sep="\t", index=False)
    simulation.to_csv(tables / "simulation_positive_control.tsv", sep="\t", index=False)
    decomposition.to_csv(tables / "relative_dwell_decomposition.tsv", sep="\t", index=False)
    retention.to_csv(tables / "backbone_top_state_retention.tsv", sep="\t", index=False)
    rank_lift.to_csv(tables / "relative_dwell_rank_lift.tsv", sep="\t", index=False)
    plot_main_figure(scores, decomposition, retention, rank_lift, simulation, figures / "Figure_E14_ablation_backbone", config)
    write_reports(root, config, summary, audit, simulation, decomposition, retention, rank_lift)


if __name__ == "__main__":
    main()
