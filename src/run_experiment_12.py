"""Run Experiment 12: clinical association validation.

Experiment 12 validates whether patient-level R* carries clinical information.
Survival/follow-up is used only as an external validation endpoint, not as a
time anchor for cancer progression. For AACR-GENIE cohorts the available raw
time fields are intervals from date of birth, so we derive follow-up after the
sequencing-report age before fitting clinical models.
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
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg

import figure_style


CONFIG_PATH = Path("configs/experiment_12.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 12.")
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


def read_scores(dataset: str, config: dict) -> pd.DataFrame:
    path = Path(config["experiment_05_root"]) / dataset / "tables" / "state_scores.tsv"
    scores = pd.read_csv(path, sep="\t")
    scores["canonical_state"] = scores["state"].map(canonical_state)
    scores = scores.sort_values(["eligible_experiment5", "high_confidence", "N_v"], ascending=[False, False, False])
    keep = [
        "canonical_state",
        "state",
        "L_v",
        "F_hat",
        "R_star",
        "O_star",
        "eligible_experiment5",
        "high_confidence",
        "R_star_ci_low",
        "R_star_ci_high",
        "clinical_annotation",
        "interpretation_flag",
    ]
    return scores[keep].drop_duplicates("canonical_state")


def prepare_patient_table(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit_rows = []
    for dataset, ds_cfg in config["datasets"].items():
        state_path = Path(config["experiment_ready_root"]) / dataset / "state_table.csv"
        state_table = pd.read_csv(state_path)
        scores = read_scores(dataset, config)
        state_table["canonical_state"] = state_table["state_id"].map(canonical_state)
        merged = state_table.merge(scores, on="canonical_state", how="left", suffixes=("", "_score"))
        time_raw = pd.to_numeric(merged["survival_time"], errors="coerce")
        age = pd.to_numeric(merged["age"], errors="coerce")
        merged["followup_time_days"] = time_raw - age * 365.25
        time_origin = config["analysis"]["aacr_time_origin"]
        merged["survival_event_binary"] = pd.to_numeric(merged["survival_event"], errors="coerce")
        merged["age_num"] = age
        merged["sex_male"] = merged["sex"].astype(str).str.lower().str.startswith("m").astype(float)
        stage = merged["stage_group"].astype(str).str.lower()
        merged["stage_metastatic"] = stage.eq("metastatic").astype(float)
        merged["stage_unknown"] = stage.isin(["unknown", "nan", ""]).astype(float)
        merged["stage_binary"] = np.where(merged["stage_metastatic"].eq(1), "metastatic", "non-metastatic")
        for col in ["L_v", "F_hat", "R_star", "O_star"]:
            merged[f"log_{col}"] = np.log(np.clip(pd.to_numeric(merged[col], errors="coerce"), 1.0e-12, None))
        merged["log2_R_star"] = np.log2(np.clip(pd.to_numeric(merged["R_star"], errors="coerce"), 1.0e-12, None))
        valid = (
            merged["followup_time_days"].gt(float(config["analysis"]["minimum_followup_days"]))
            & merged["survival_event_binary"].isin([0, 1])
            & merged["R_star"].notna()
            & merged["eligible_experiment5"].astype("boolean").fillna(False)
        )
        eligible = merged[valid].copy()
        eligible["sort_high_confidence"] = eligible["high_confidence"].astype("boolean").fillna(False).astype(int)
        eligible["sort_known_stage"] = 1 - eligible["stage_unknown"].astype(int)
        eligible = eligible.sort_values(
            ["patient_id", "sort_high_confidence", "sort_known_stage", "event_count", "followup_time_days", "sample_id"],
            ascending=[True, False, False, False, False, True],
        )
        patients = eligible.drop_duplicates("patient_id").copy()
        patients["dataset_name"] = dataset
        patients["short_name"] = ds_cfg["short_name"]
        patients["display_name"] = ds_cfg["display_name"]
        median_r = patients["R_star"].median()
        patients["R_group"] = np.where(patients["R_star"].ge(median_r), "High R*", "Low R*")
        rows.append(patients)
        audit_rows.append(
            {
                "dataset_name": dataset,
                "short_name": ds_cfg["short_name"],
                "time_origin": time_origin,
                "state_table_rows": int(len(state_table)),
                "unique_patients_raw": int(state_table["patient_id"].nunique()),
                "rows_with_score": int(merged["R_star"].notna().sum()),
                "eligible_rows": int(len(eligible)),
                "analysis_patients": int(len(patients)),
                "events": int(patients["survival_event_binary"].sum()),
                "median_followup_days": float(patients["followup_time_days"].median()) if len(patients) else np.nan,
                "high_confidence_fraction": float(patients["high_confidence"].astype(bool).mean()) if len(patients) else np.nan,
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audit_rows)


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


def fit_cox(frame: pd.DataFrame, features: list[str], required_feature: str | None = None) -> dict:
    candidates = list(features)
    fallback_order = ["stage_metastatic", "sex_male", "event_count", "age_num", "log_F_hat", "log_L_v"]
    attempts = []
    while candidates:
        x, used = standardize_features(frame, candidates)
        if required_feature and required_feature not in used:
            return {"ok": False, "reason": f"required feature {required_feature} unavailable", "features": used}
        y = frame.loc[x.index].copy()
        if len(y) < 20 or y["survival_event_binary"].sum() < 5:
            return {"ok": False, "reason": "insufficient outcome after feature filtering", "features": used}
        try:
            fit = PHReg(y["followup_time_days"], x, status=y["survival_event_binary"]).fit(disp=0)
            params = pd.Series(fit.params, index=used)
            se = pd.Series(fit.bse, index=used)
            pvals = pd.Series(fit.pvalues, index=used)
            risk = np.dot(x.to_numpy(dtype=float), fit.params)
            return {
                "ok": True,
                "fit": fit,
                "features": used,
                "n": int(len(y)),
                "events": int(y["survival_event_binary"].sum()),
                "params": params,
                "se": se,
                "pvalues": pvals,
                "risk": pd.Series(risk, index=x.index),
            }
        except Exception as exc:  # singular matrices are handled by dropping nonessential terms.
            attempts.append(f"{type(exc).__name__}: {exc}")
            removable = [f for f in fallback_order if f in candidates and f != required_feature]
            if not removable:
                return {"ok": False, "reason": "; ".join(attempts[-3:]), "features": candidates}
            candidates.remove(removable[0])
    return {"ok": False, "reason": "no estimable feature set", "features": []}


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


def cox_results(patients: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit = []
    base_features = ["age_num", "sex_male", "stage_metastatic", "event_count", "log_R_star"]
    for dataset, sub in patients.groupby("dataset_name", sort=False):
        short = sub["short_name"].iloc[0]
        for model_name, features in [
            ("R_star_unadjusted", ["log_R_star"]),
            ("R_star_adjusted", base_features),
        ]:
            result = fit_cox(sub, features, required_feature="log_R_star")
            audit.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "model": model_name,
                    "ok": bool(result["ok"]),
                    "features_used": ";".join(result.get("features", [])),
                    "reason": result.get("reason", "OK" if result["ok"] else ""),
                }
            )
            if result["ok"]:
                beta = float(result["params"]["log_R_star"])
                se = float(result["se"]["log_R_star"])
                rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "model": model_name,
                        "n": result["n"],
                        "events": result["events"],
                        "beta_log_R_star_per_sd": beta,
                        "hr_per_sd": float(np.exp(beta)),
                        "ci_low": float(np.exp(beta - 1.96 * se)),
                        "ci_high": float(np.exp(beta + 1.96 * se)),
                        "p_value": float(result["pvalues"]["log_R_star"]),
                        "features_used": ";".join(result["features"]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(audit)


def cindex_results(patients: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    audit = []
    for dataset, sub in patients.groupby("dataset_name", sort=False):
        short = sub["short_name"].iloc[0]
        for model_name, features in config["analysis"]["cindex_models"].items():
            result = fit_cox(sub, list(features))
            audit.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "model": f"cindex_{model_name}",
                    "ok": bool(result["ok"]),
                    "features_used": ";".join(result.get("features", [])),
                    "reason": result.get("reason", "OK" if result["ok"] else ""),
                }
            )
            if result["ok"]:
                y = sub.loc[result["risk"].index]
                rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "model": model_name,
                        "n": result["n"],
                        "events": result["events"],
                        "c_index": harrell_c_index(y["followup_time_days"], y["survival_event_binary"], result["risk"]),
                        "features_used": ";".join(result["features"]),
                    }
                )
    cindex = pd.DataFrame(rows)
    if not cindex.empty:
        clinical = cindex[cindex["model"].eq("clinical_only")][["dataset_name", "c_index"]].rename(columns={"c_index": "clinical_c_index"})
        cindex = cindex.merge(clinical, on="dataset_name", how="left")
        cindex["delta_vs_clinical"] = cindex["c_index"] - cindex["clinical_c_index"]
    return cindex, pd.DataFrame(audit)


def km_curve(time: pd.Series, event: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"time": time, "event": event}).dropna().sort_values("time")
    if df.empty:
        return pd.DataFrame([{"time": 0.0, "survival": 1.0}])
    grouped = (
        df.groupby("time", sort=True)
        .agg(total=("event", "size"), deaths=("event", lambda s: int(np.sum(np.asarray(s) == 1))))
        .reset_index()
    )
    survival = 1.0
    rows = [{"time": 0.0, "survival": 1.0}]
    removed_before = 0
    n_total = len(df)
    for row in grouped.itertuples():
        t = float(row.time)
        at_risk = n_total - removed_before
        deaths = int(row.deaths)
        if at_risk > 0 and deaths > 0:
            survival *= 1.0 - deaths / at_risk
            rows.append({"time": t, "survival": float(survival)})
        removed_before += int(row.total)
    return pd.DataFrame(rows)


def survival_at(curve: pd.DataFrame, horizon_days: float) -> float:
    if curve.empty:
        return np.nan
    sub = curve[curve["time"].le(float(horizon_days))]
    if sub.empty:
        return 1.0
    return float(sub.sort_values("time")["survival"].iloc[-1])


def rmst_from_curve(curve: pd.DataFrame, horizon_days: float) -> float:
    if curve.empty:
        return np.nan
    curve = curve.sort_values("time")
    tau = float(horizon_days)
    last_time = 0.0
    last_surv = 1.0
    area = 0.0
    for row in curve.itertuples():
        t = float(row.time)
        if t <= 0:
            last_surv = float(row.survival)
            continue
        if t >= tau:
            area += max(tau - last_time, 0.0) * last_surv
            return float(area)
        area += max(t - last_time, 0.0) * last_surv
        last_time = t
        last_surv = float(row.survival)
    area += max(tau - last_time, 0.0) * last_surv
    return float(area)


def group_survival_metric(frame: pd.DataFrame, horizon_days: float, metric: str) -> float:
    curve = km_curve(frame["followup_time_days"], frame["survival_event_binary"])
    if metric == "survival":
        return survival_at(curve, horizon_days)
    if metric == "rmst":
        return rmst_from_curve(curve, horizon_days)
    raise ValueError(metric)


def bootstrap_group_difference(sub: pd.DataFrame, horizon_days: float, metric: str, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    high = sub[sub["R_group"].eq("High R*")].reset_index(drop=True)
    low = sub[sub["R_group"].eq("Low R*")].reset_index(drop=True)
    if high.empty or low.empty:
        return np.nan, np.nan
    values = []
    for _ in range(reps):
        h = high.iloc[rng.integers(0, len(high), len(high))]
        l = low.iloc[rng.integers(0, len(low), len(low))]
        diff = group_survival_metric(h, horizon_days, metric) - group_survival_metric(l, horizon_days, metric)
        if np.isfinite(diff):
            values.append(float(diff))
    if not values:
        return np.nan, np.nan
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def clinical_effect_results(patients: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rmst_rows = []
    landmark_rows = []
    reps = int(config["analysis"].get("bootstrap_replicates", 200))
    rmst_horizon = float(config["analysis"]["rmst_horizon_years"]) * 365.25
    for dataset, sub in patients.groupby("dataset_name", sort=False):
        short = sub["short_name"].iloc[0]
        high = sub[sub["R_group"].eq("High R*")]
        low = sub[sub["R_group"].eq("Low R*")]
        high_rmst = group_survival_metric(high, rmst_horizon, "rmst")
        low_rmst = group_survival_metric(low, rmst_horizon, "rmst")
        seed = 20261200 + sum(ord(ch) for ch in dataset)
        ci_low, ci_high = bootstrap_group_difference(sub, rmst_horizon, "rmst", reps, seed=seed)
        rmst_rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "horizon_years": float(config["analysis"]["rmst_horizon_years"]),
                "high_rmst_days": high_rmst,
                "low_rmst_days": low_rmst,
                "delta_rmst_days": high_rmst - low_rmst,
                "delta_rmst_months": (high_rmst - low_rmst) / 30.4375,
                "ci_low_months": ci_low / 30.4375 if np.isfinite(ci_low) else np.nan,
                "ci_high_months": ci_high / 30.4375 if np.isfinite(ci_high) else np.nan,
                "bootstrap_replicates": reps,
            }
        )
        for year in config["analysis"]["landmark_years"]:
            horizon = float(year) * 365.25
            high_max = float(high["followup_time_days"].max()) if len(high) else np.nan
            low_max = float(low["followup_time_days"].max()) if len(low) else np.nan
            high_at_risk = int(high["followup_time_days"].ge(horizon).sum())
            low_at_risk = int(low["followup_time_days"].ge(horizon).sum())
            estimable = bool(high_max >= horizon and low_max >= horizon)
            high_s = group_survival_metric(high, horizon, "survival") if estimable else np.nan
            low_s = group_survival_metric(low, horizon, "survival") if estimable else np.nan
            landmark_rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "horizon_years": float(year),
                    "estimable": estimable,
                    "high_at_risk": high_at_risk,
                    "low_at_risk": low_at_risk,
                    "high_max_followup_years": high_max / 365.25 if np.isfinite(high_max) else np.nan,
                    "low_max_followup_years": low_max / 365.25 if np.isfinite(low_max) else np.nan,
                    "high_survival": high_s,
                    "low_survival": low_s,
                    "delta_survival": high_s - low_s,
                    "delta_percentage_points": 100.0 * (high_s - low_s),
                }
            )
    return pd.DataFrame(rmst_rows), pd.DataFrame(landmark_rows)


def logrank_p(time: pd.Series, event: pd.Series, group: pd.Series) -> float:
    df = pd.DataFrame({"time": time, "event": event, "group": group}).dropna()
    levels = list(df["group"].dropna().unique())
    if len(levels) != 2:
        return np.nan
    g1 = levels[0]
    observed = expected = variance = 0.0
    for t in sorted(df.loc[df["event"].eq(1), "time"].unique()):
        risk = df["time"].ge(t)
        events = df["time"].eq(t) & df["event"].eq(1)
        n = int(risk.sum())
        d = int(events.sum())
        n1 = int((risk & df["group"].eq(g1)).sum())
        d1 = int((events & df["group"].eq(g1)).sum())
        if n <= 1:
            continue
        observed += d1
        expected += d * n1 / n
        variance += (n1 / n) * (1 - n1 / n) * d * (n - d) / (n - 1)
    if variance <= 0:
        return np.nan
    stat = (observed - expected) ** 2 / variance
    return float(chi2.sf(stat, 1))


def km_results(patients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    summary = []
    for dataset, sub in patients.groupby("dataset_name", sort=False):
        short = sub["short_name"].iloc[0]
        p_value = logrank_p(sub["followup_time_days"], sub["survival_event_binary"], sub["R_group"])
        for group, part in sub.groupby("R_group"):
            curve = km_curve(part["followup_time_days"], part["survival_event_binary"])
            curve["dataset_name"] = dataset
            curve["short_name"] = short
            curve["R_group"] = group
            curves.append(curve)
            summary.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "R_group": group,
                    "n": int(len(part)),
                    "events": int(part["survival_event_binary"].sum()),
                    "median_followup_days": float(part["followup_time_days"].median()),
                    "logrank_p_value": p_value,
                }
            )
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


def subgroup_results(patients: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for (dataset, stage_binary), sub in patients.groupby(["dataset_name", "stage_binary"], sort=False):
        short = sub["short_name"].iloc[0]
        min_n = int(config["analysis"]["minimum_subgroup_patients"])
        min_events = int(config["analysis"]["minimum_subgroup_events"])
        if len(sub) < min_n or sub["survival_event_binary"].sum() < min_events:
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "subgroup": stage_binary,
                    "n": int(len(sub)),
                    "events": int(sub["survival_event_binary"].sum()),
                    "ok": False,
                    "reason": "insufficient subgroup size/events",
                }
            )
            continue
        result = fit_cox(sub, ["age_num", "sex_male", "event_count", "log_R_star"], required_feature="log_R_star")
        if result["ok"]:
            beta = float(result["params"]["log_R_star"])
            se = float(result["se"]["log_R_star"])
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "subgroup": stage_binary,
                    "n": result["n"],
                    "events": result["events"],
                    "ok": True,
                    "hr_per_sd": float(np.exp(beta)),
                    "ci_low": float(np.exp(beta - 1.96 * se)),
                    "ci_high": float(np.exp(beta + 1.96 * se)),
                    "p_value": float(result["pvalues"]["log_R_star"]),
                    "features_used": ";".join(result["features"]),
                    "reason": "OK",
                }
            )
        else:
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": short,
                    "subgroup": stage_binary,
                    "n": int(len(sub)),
                    "events": int(sub["survival_event_binary"].sum()),
                    "ok": False,
                    "reason": result.get("reason", "fit failed"),
                }
            )
    return pd.DataFrame(rows)


def square_save(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def spread_label_positions(values: list[float], min_gap: float, lower: float, upper: float) -> list[float]:
    """Return nearby y-label positions separated enough to avoid text collisions."""
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


def plot_main_figure(
    patients: pd.DataFrame,
    km_curves: pd.DataFrame,
    km_summary: pd.DataFrame,
    landmark: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    high_color = cat.get("coral", "#E8B2A7")
    low_color = cat.get("sky_blue", "#B2E6FD")
    cohort_colors = [
        cat.get("lavender", "#B5AED5"),
        cat.get("sky_blue", "#B2E6FD"),
        cat.get("sage", "#B8D2CC"),
        cat.get("coral", "#E8B2A7"),
    ]
    fig = plt.figure(figsize=(7.2, 5.2))
    dataset_names = list(config["datasets"])
    a_left = 0.075
    a_right = 0.965
    a_bottom = 0.545
    a_height = 0.265
    a_gap = 0.052
    a_width = (a_right - a_left - a_gap * (len(dataset_names) - 1)) / len(dataset_names)
    axes_a = [
        fig.add_axes([a_left + idx * (a_width + a_gap), a_bottom, a_width, a_height])
        for idx in range(len(dataset_names))
    ]
    for idx, (ax, dataset) in enumerate(zip(axes_a, dataset_names)):
        short = config["datasets"][dataset]["short_name"]
        p_value = km_summary.loc[km_summary["dataset_name"].eq(dataset), "logrank_p_value"].dropna()
        p_label = f"p={p_value.iloc[0]:.2g}" if len(p_value) and np.isfinite(p_value.iloc[0]) else "p=NA"
        for group, color in [("Low R*", low_color), ("High R*", high_color)]:
            curve = km_curves[km_curves["dataset_name"].eq(dataset) & km_curves["R_group"].eq(group)]
            if not curve.empty:
                ax.step(curve["time"] / 365.25, curve["survival"], where="post", color=color, lw=1.0, label=group)
        ax.text(0.05, 0.95, short, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=text_primary)
        ax.text(0.95, 0.95, p_label, transform=ax.transAxes, ha="right", va="top", fontsize=5.2, color=text_secondary)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ax.grid(color=grid_color, lw=0.34)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=5.4, length=1.6, width=0.5)
        ax.set_xlabel("follow-up (years)", fontsize=5.7)
        if idx == 0:
            ax.set_ylabel("survival", fontsize=5.7)
        else:
            ax.set_yticklabels([])
        if idx == 1:
            ax.legend(frameon=False, loc="lower left", fontsize=5.0, handlelength=1.4)
        ax.set_box_aspect(1)
    axes_a[0].text(-0.34, 1.22, "a", transform=axes_a[0].transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    axes_a[0].text(-0.02, 1.22, "Median R* clinical separation", transform=axes_a[0].transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    ax_b = fig.add_axes([0.090, 0.070, 0.790, 0.265])
    years = np.asarray([float(year) for year in config["analysis"]["landmark_years"]], dtype=float)
    landmark_span = np.nanmax(np.abs(landmark["delta_percentage_points"].to_numpy(dtype=float)))
    landmark_span = max(4.0, float(landmark_span))
    y_min = -landmark_span * 1.22
    y_max = landmark_span * 1.22
    landmark_profiles: dict[str, np.ndarray] = {}
    for color, dataset in zip(cohort_colors, config["datasets"]):
        sub = (
            landmark[landmark["dataset_name"].eq(dataset)]
            .set_index("horizon_years")
            .reindex(years)
            .reset_index()
        )
        values = sub["delta_percentage_points"].to_numpy(dtype=float)
        short = config["datasets"][dataset]["short_name"]
        landmark_profiles[dataset] = values
        ax_b.plot(years, values, color=color, lw=1.05, marker="o", ms=3.9, mec=text_primary, mew=0.28, zorder=3)
    label_records = []
    for dataset in config["datasets"]:
        values = landmark_profiles[dataset]
        finite = np.where(np.isfinite(values))[0]
        if len(finite):
            last_index = int(finite[-1])
            label_records.append(
                {
                    "dataset": dataset,
                    "year": float(years[last_index]),
                    "value": float(values[last_index]),
                    "short": config["datasets"][dataset]["short_name"],
                }
            )
    for year in sorted({record["year"] for record in label_records}):
        records = [record for record in label_records if np.isclose(record["year"], year)]
        y_positions = spread_label_positions([record["value"] for record in records], min_gap=0.72, lower=y_min + 0.50, upper=y_max - 0.50)
        for record, y_text in zip(records, y_positions):
            year_note = "" if np.isclose(record["year"], years[-1]) else f" ({record['year']:.0f}y)"
            ax_b.text(record["year"] + 0.16, y_text, f"{record['short']} {record['value']:+.1f}{year_note}", ha="left", va="center", fontsize=5.2, color=text_secondary)
    ax_b.axhline(0, color="#777777", lw=0.68, ls=(0, (3, 2)), zorder=1)
    ax_b.set_xlim(years.min() - 0.18, years.max() + 0.92)
    ax_b.set_ylim(y_min, y_max)
    ax_b.set_xlabel("Landmark after sequencing (years)", fontsize=6.3)
    ax_b.set_ylabel("High-Low R* survival difference (pp)", fontsize=6.3)
    ax_b.set_xticks(np.arange(int(years.min()), int(years.max()) + 1), [str(year) for year in range(int(years.min()), int(years.max()) + 1)], fontsize=5.8)
    ax_b.grid(axis="both", color=grid_color, lw=0.38)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(labelsize=5.6, length=2.0, width=0.55)
    ax_b.text(-0.09, 1.12, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.text(0.00, 1.12, "Landmark survival profile", transform=ax_b.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    fig.text(0.075, 0.972, "Experiment 12 | Clinical validation of relative dwell-state signal", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.075, 0.947, "Patient-level validation; survival/follow-up is used only as an external endpoint, not as a progression-time anchor.", ha="left", va="top", fontsize=5.9, color=text_secondary)
    square_save(fig, output, config)


def plot_single_figures(
    km_curves: pd.DataFrame,
    km_summary: pd.DataFrame,
    landmark: pd.DataFrame,
    root: Path,
    config: dict,
) -> None:
    """Render atomic E12 panels directly from clinical result tables."""
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    high_color = cat.get("coral", "#E8B2A7")
    low_color = cat.get("sky_blue", "#B2E6FD")
    cohort_colors = [
        cat.get("lavender", "#B5AED5"),
        cat.get("sky_blue", "#B2E6FD"),
        cat.get("sage", "#B8D2CC"),
        cat.get("coral", "#E8B2A7"),
    ]
    single_dir = root / "single_figures"
    single_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = list(config["datasets"])

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(6.9, 2.35), sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.90, wspace=0.24)
    for idx, (ax, dataset) in enumerate(zip(axes, dataset_names)):
        short = config["datasets"][dataset]["short_name"]
        p_value = km_summary.loc[km_summary["dataset_name"].eq(dataset), "logrank_p_value"].dropna()
        p_label = f"p={p_value.iloc[0]:.2g}" if len(p_value) and np.isfinite(p_value.iloc[0]) else "p=NE"
        for group, color in [("Low R*", low_color), ("High R*", high_color)]:
            curve = km_curves[km_curves["dataset_name"].eq(dataset) & km_curves["R_group"].eq(group)]
            if not curve.empty:
                ax.step(curve["time"] / 365.25, curve["survival"], where="post", color=color, lw=1.0, label=group)
        ax.text(0.05, 0.95, short, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, fontweight="bold", color=text_primary)
        ax.text(0.95, 0.95, p_label, transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=text_secondary)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Follow-up (years)", fontsize=6.2)
        if idx == 0:
            ax.set_ylabel("Survival", fontsize=6.2)
        ax.grid(color=grid_color, lw=0.34)
        ax.tick_params(labelsize=5.6, length=1.8, width=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[min(1, len(axes) - 1)].legend(frameon=False, loc="lower left", fontsize=5.2, handlelength=1.35)
    figure_style.save_figure(
        fig,
        single_dir / "Figure_E12_clinical_validation__km_three_cohorts",
        config,
        pad_inches=0.02,
    )

    fig, ax = plt.subplots(figsize=(4.2, 2.75))
    fig.subplots_adjust(left=0.16, right=0.82, bottom=0.20, top=0.94)
    years = np.asarray([float(year) for year in config["analysis"]["landmark_years"]], dtype=float)
    landmark_span = np.nanmax(np.abs(landmark["delta_percentage_points"].to_numpy(dtype=float)))
    landmark_span = max(4.0, float(landmark_span))
    y_min = -landmark_span * 1.22
    y_max = landmark_span * 1.22
    profiles: dict[str, np.ndarray] = {}
    for color, dataset in zip(cohort_colors, dataset_names):
        sub = (
            landmark[landmark["dataset_name"].eq(dataset)]
            .set_index("horizon_years")
            .reindex(years)
            .reset_index()
        )
        values = sub["delta_percentage_points"].to_numpy(dtype=float)
        profiles[dataset] = values
        ax.plot(years, values, color=color, lw=1.05, marker="o", ms=3.8, mec=text_primary, mew=0.28, zorder=3)
    labels = []
    for dataset in dataset_names:
        values = profiles[dataset]
        finite = np.where(np.isfinite(values))[0]
        if len(finite):
            last_index = int(finite[-1])
            labels.append(
                {
                    "dataset": dataset,
                    "year": float(years[last_index]),
                    "value": float(values[last_index]),
                    "short": config["datasets"][dataset]["short_name"],
                }
            )
    for year in sorted({record["year"] for record in labels}):
        records = [record for record in labels if np.isclose(record["year"], year)]
        y_positions = spread_label_positions([record["value"] for record in records], min_gap=0.72, lower=y_min + 0.50, upper=y_max - 0.50)
        for record, y_text in zip(records, y_positions):
            year_note = "" if np.isclose(record["year"], years[-1]) else f" ({record['year']:.0f}y)"
            ax.text(record["year"] + 0.16, y_text, f"{record['short']} {record['value']:+.1f}{year_note}", ha="left", va="center", fontsize=5.2, color=text_secondary)
    ax.axhline(0, color="#777777", lw=0.68, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(years.min() - 0.18, years.max() + 0.92)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Landmark after sequencing (years)", fontsize=6.3)
    ax.set_ylabel("High-Low R* survival difference (pp)", fontsize=6.3)
    ax.set_xticks(np.arange(int(years.min()), int(years.max()) + 1))
    ax.grid(axis="both", color=grid_color, lw=0.38)
    ax.tick_params(labelsize=5.7, length=2.0, width=0.55)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    figure_style.save_figure(
        fig,
        single_dir / "Figure_E12_clinical_validation__landmark_survival_profile",
        config,
        pad_inches=0.02,
    )


def write_reports(
    root: Path,
    config: dict,
    audit: pd.DataFrame,
    cox: pd.DataFrame,
    cindex: pd.DataFrame,
    km_summary: pd.DataFrame,
    subgroup: pd.DataFrame,
    fit_audit: pd.DataFrame,
    rmst: pd.DataFrame,
    landmark: pd.DataFrame,
) -> None:
    protocol = f"""# Experiment 12 Protocol Audit

## Protocol Section

Source document section: `18. 实验 12：临床关联验证`.

Purpose: validate whether patient-level R* has clinical explanatory value.
Survival/follow-up is used only as a validation endpoint and is not used as an
external time anchor for progression.

## Endpoint Handling

- AACR cohorts: raw clinical intervals are from date of birth. Follow-up after
  sequencing is derived as `INT_DOD/INT_CONTACT - AGE_AT_SEQ_REPORT * 365.25`.
- One representative scored state is retained per patient to avoid duplicating
  survival records from multiple samples.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_12_protocol_audit.md").write_text(protocol, encoding="utf-8")

    lines = [
        "# Experiment 12 Summary",
        "",
    ]
    years = [float(year) for year in config["analysis"]["landmark_years"]]
    delta_headers = " | ".join(f"Delta {int(year)}-year pp" for year in years)
    lines.append(f"| Cohort | Patients | Events | Median follow-up days | KM log-rank p | {delta_headers} |")
    lines.append("|---|---:|---:|---:|---:|" + "---:|" * len(years))
    km_p = km_summary.groupby("dataset_name")["logrank_p_value"].first()
    landmark_wide = landmark.pivot(index="dataset_name", columns="horizon_years", values="delta_percentage_points")
    for row in audit.itertuples():
        deltas = []
        for year in years:
            if row.dataset_name in landmark_wide.index and year in landmark_wide.columns:
                value = landmark_wide.loc[row.dataset_name, year]
                deltas.append(f"{value:+.1f}" if np.isfinite(value) else "NA")
            else:
                deltas.append("NA")
        delta_text = " | ".join(deltas)
        lines.append(
            f"| {row.short_name} | {row.analysis_patients} | {row.events} | {row.median_followup_days:.0f} | {km_p.loc[row.dataset_name]:.2g} | {delta_text} |"
        )
    (root / "experiment_12_summary.md").write_text("\n".join(lines), encoding="utf-8")

    sci = [
        "# Experiment 12 Scientific Review",
        "",
        "## Main Interpretation",
        "",
        "Experiment 12 evaluates whether the relative dwell-state score R* has patient-level clinical association. The main figure now focuses on the two most interpretable survival-validation views: high/low KM separation and landmark survival advantage over fixed endpoints.",
        "",
        "A hazard ratio below 1 means that higher R* is associated with lower observed death hazard in the validation endpoint. This is compatible with high-R* states representing slower or chronic relative dwell states, rather than necessarily aggressive states. Interpretation must remain coupled to O* and the biological annotation of the state.",
        "",
        "Restricted mean survival time is retained as a supplementary effect-size table, but it is removed from the primary figure to avoid duplicating the same survival-association message shown by the KM and landmark analyses.",
        "",
        "## Model Fit Audit",
        "",
        "| Cohort | Model | OK | Features used | Note |",
        "|---|---|---:|---|---|",
    ]
    for row in fit_audit.itertuples():
        sci.append(f"| {row.short_name} | {row.model} | {row.ok} | {row.features_used} | {row.reason} |")
    sci.extend(
        [
            "",
            "## Supplementary Cox and Stage-Stratified Results",
            "",
            "| Cohort | Subgroup | N | Events | HR | 95% CI | p |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in subgroup.itertuples():
        if bool(row.ok):
            sci.append(f"| {row.short_name} | {row.subgroup} | {row.n} | {row.events} | {row.hr_per_sd:.2f} | {row.ci_low:.2f}-{row.ci_high:.2f} | {row.p_value:.2g} |")
        else:
            sci.append(f"| {row.short_name} | {row.subgroup} | {row.n} | {row.events} | NA | NA | NA |")
    sci.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a validation experiment. It does not prove that R* is a clock or an absolute residence time. The AACR time origin is sequencing-report follow-up reconstructed from GENIE DOB-relative intervals, so the endpoint should be interpreted as post-sequencing survival association.",
        ]
    )
    (root / "experiment_12_scientific_review.md").write_text("\n".join(sci), encoding="utf-8")

    design = f"""# Experiment 12 Figure Design Review

## Sources

{figure_style.design_sources_markdown(config)}

## Rules Applied

{figure_style.design_rules_markdown(config)}

## Design Choices

- Panel A follows the clinical validation convention: KM curves with direct
  p-value annotation in compact cohort small multiples.
- Panel B reports yearly landmark survival differences as a compact
  profile rather than a heatmap, so the sign and time trend are directly visible.
- The previous quintile-gradient panel was removed from the main figure to keep
  the clinical validation centered on the primary survival endpoint contrast.
- Cox, C-index and subgroup analyses are retained as supplementary audit tables
  rather than the main visual emphasis.
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

    patients, audit = prepare_patient_table(config)
    cox, cox_audit = cox_results(patients, config)
    cindex, cindex_audit = cindex_results(patients, config)
    km_curves, km_summary = km_results(patients)
    subgroup = subgroup_results(patients, config)
    rmst, landmark = clinical_effect_results(patients, config)
    fit_audit = pd.concat([cox_audit, cindex_audit], ignore_index=True)

    patients.to_csv(tables / "patient_clinical_scores.tsv", sep="\t", index=False)
    audit.to_csv(tables / "clinical_endpoint_audit.tsv", sep="\t", index=False)
    cox.to_csv(tables / "cox_rstar_results.tsv", sep="\t", index=False)
    cindex.to_csv(tables / "cindex_comparison.tsv", sep="\t", index=False)
    km_curves.to_csv(tables / "km_curves.tsv", sep="\t", index=False)
    km_summary.to_csv(tables / "km_group_summary.tsv", sep="\t", index=False)
    subgroup.to_csv(tables / "stage_subgroup_cox.tsv", sep="\t", index=False)
    rmst.to_csv(tables / "rmst_difference.tsv", sep="\t", index=False)
    landmark.to_csv(tables / "landmark_survival_difference.tsv", sep="\t", index=False)
    obsolete_quintile = tables / "rstar_quintile_survival.tsv"
    if obsolete_quintile.exists():
        obsolete_quintile.unlink()
    obsolete_tertile = tables / "rstar_tertile_survival.tsv"
    if obsolete_tertile.exists():
        obsolete_tertile.unlink()
    fit_audit.to_csv(tables / "model_fit_audit.tsv", sep="\t", index=False)

    plot_main_figure(patients, km_curves, km_summary, landmark, figures / "Figure_E12_clinical_validation", config)
    plot_single_figures(km_curves, km_summary, landmark, root, config)
    write_reports(root, config, audit, cox, cindex, km_summary, subgroup, fit_audit, rmst, landmark)


if __name__ == "__main__":
    main()
