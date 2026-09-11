"""Numerical summaries and dominant-predecessor route tables for E17."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from relobstq_mhn.core.topology import build_dominant_predecessor_path, event_added
from .parameters import config_path


def genotype_events(genotype: object) -> set[str]:
    if pd.isna(genotype):
        return set()
    text = str(genotype).strip()
    if text == "" or text.upper() == "WT":
        return set()
    return {event for event in text.split("+") if event}


def genotype_similarity(row: pd.Series) -> float:
    early = genotype_events(row["early_genotype"])
    late = genotype_events(row["late_genotype"])
    if not early and (not late):
        return 1.0
    union = early | late
    return float(len(early & late) / len(union)) if union else 1.0


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]), float(values[0]))
    means = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = sample.mean()
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def prepare_predictions(config: dict) -> pd.DataFrame:
    source = (
        Path(config["source_result_root"])
        / "tables"
        / "dwell_persistence_predictions_all.tsv"
    )
    predictions = pd.read_csv(source, sep="\t")
    included = set(config["included_studies"].keys())
    score_col = config["analysis"]["primary_score_column"]
    data = predictions[predictions["study_id"].isin(included)].copy()
    filters = config["analysis"]["filters"]
    if filters.get("require_pair_qc_pass", True):
        data = data[data["pair_qc_pass"].eq(True)]
    data = data[data[score_col].notna()].copy()
    if filters.get("require_exact_heldout_state_score", True):
        data = data[data["score_source"].eq("exact_state")].copy()
    data["genotype_similarity_fixed"] = data.apply(genotype_similarity, axis=1)
    data["minimum_dwell_proxy"] = pd.to_numeric(
        data["minimum_observed_dwell_interval"], errors="coerce"
    ).fillna(0.0)
    data["empirical_persistent"] = pd.to_numeric(
        data["empirical_persistent"], errors="coerce"
    ).astype(int)
    bins = int(config["analysis"]["rstar_quantile_bins"])
    labels = [f"Q{i}" for i in range(1, bins + 1)]
    parts = []
    for study_id, group in data.groupby("study_id", sort=False):
        ranked = group[score_col].rank(method="first")
        group = group.copy()
        group["rstar_bin"] = pd.qcut(ranked, bins, labels=labels)
        group["rstar_bin_index"] = (
            group["rstar_bin"].astype(str).str.extract("Q(\\d+)").astype(int)
        )
        max_proxy = float(group["minimum_dwell_proxy"].max())
        group["minimum_dwell_proxy_scaled"] = (
            group["minimum_dwell_proxy"] / max_proxy if max_proxy > 0 else 0.0
        )
        group["study_short_name"] = config["included_studies"][study_id]["short_name"]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def summarize_bins(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["random_seed"]))
    replicates = int(config["analysis"]["bootstrap_replicates"])
    rows = []
    score_col = config["analysis"]["primary_score_column"]
    for (study_id, bin_label), group in data.groupby(
        ["study_id", "rstar_bin"], observed=True
    ):
        row = {
            "study_id": study_id,
            "study_short_name": group["study_short_name"].iloc[0],
            "rstar_bin": str(bin_label),
            "rstar_bin_index": int(group["rstar_bin_index"].iloc[0]),
            "n_pairs": int(len(group)),
            "log2_R_min": float(group[score_col].min()),
            "log2_R_median": float(group[score_col].median()),
            "log2_R_max": float(group[score_col].max()),
        }
        for column, prefix in [
            ("empirical_persistent", "persistence_rate"),
            ("genotype_similarity_fixed", "genotype_similarity"),
            ("minimum_dwell_proxy_scaled", "minimum_dwell_proxy_scaled"),
        ]:
            mean, low, high = bootstrap_mean_ci(
                group[column].to_numpy(dtype=float), rng, replicates
            )
            row[prefix] = mean
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["study_id", "rstar_bin_index"])


def summarize_metrics(
    data: pd.DataFrame, bins: pd.DataFrame, config: dict
) -> pd.DataFrame:
    rows = []
    score_col = config["analysis"]["primary_score_column"]
    for study_id, group in data.groupby("study_id", sort=False):
        study_bins = bins[bins["study_id"] == study_id].sort_values("rstar_bin_index")
        low = study_bins.iloc[0]
        high = study_bins.iloc[-1]
        row = {
            "study_id": study_id,
            "study_short_name": config["included_studies"][study_id]["short_name"],
            "n_evaluable_pairs": int(len(group)),
            "persistent_pairs": int(group["empirical_persistent"].sum()),
            "changed_pairs": int((1 - group["empirical_persistent"]).sum()),
            "top_bottom_persistence_delta": float(
                high["persistence_rate"] - low["persistence_rate"]
            ),
            "top_bottom_similarity_delta": float(
                high["genotype_similarity"] - low["genotype_similarity"]
            ),
            "top_bottom_minimum_dwell_proxy_delta_scaled": float(
                high["minimum_dwell_proxy_scaled"] - low["minimum_dwell_proxy_scaled"]
            ),
            "top_bin_persistence_rate": float(high["persistence_rate"]),
            "bottom_bin_persistence_rate": float(low["persistence_rate"]),
            "spearman_persistence": float(
                spearmanr(group[score_col], group["empirical_persistent"]).statistic
            ),
            "spearman_persistence_p": float(
                spearmanr(group[score_col], group["empirical_persistent"]).pvalue
            ),
            "spearman_similarity": float(
                spearmanr(
                    group[score_col], group["genotype_similarity_fixed"]
                ).statistic
            ),
            "spearman_similarity_p": float(
                spearmanr(group[score_col], group["genotype_similarity_fixed"]).pvalue
            ),
            "spearman_minimum_dwell_proxy": float(
                spearmanr(group[score_col], group["minimum_dwell_proxy"]).statistic
            ),
            "spearman_minimum_dwell_proxy_p": float(
                spearmanr(group[score_col], group["minimum_dwell_proxy"]).pvalue
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_float(value: object, digits: int = 2, missing: str = "NE") -> str:
    try:
        number = float(value)
    except Exception:
        return missing
    if not np.isfinite(number):
        return missing
    return f"{number:.{digits}f}"


def write_integrated_metrics_table(root: Path, config: dict) -> pd.DataFrame:
    source_root = Path(config["source_result_root"])
    core = pd.read_csv(source_root / "tables" / "core_metric_table.tsv", sep="\t")
    metrics = pd.read_csv(root / "tables" / "rstar_calibration_metrics.tsv", sep="\t")
    included = config["included_studies"]
    core_lookup = core.set_index("cohort")
    metric_lookup = metrics.set_index("study_id")
    primary_ids = {"difg_glass", "coadread_mskcc"}
    rows = []
    for study_id, info in included.items():
        short_name = info["short_name"]
        core_row = (
            core_lookup.loc[short_name]
            if short_name in core_lookup.index
            else pd.Series(dtype=object)
        )
        metric_row = (
            metric_lookup.loc[study_id]
            if study_id in metric_lookup.index
            else pd.Series(dtype=object)
        )
        rows.append(
            {
                "study_id": study_id,
                "cohort": short_name,
                "evidence_role": (
                    "primary" if study_id in primary_ids else "supplementary"
                ),
                "n_P_C": core_row.get("n_P_C", "NE"),
                "auc_95ci": core_row.get("AUC_95CI", "NE"),
                "ap_lift": core_row.get("AP_lift", "NE"),
                "tertile_delta_persistence_95ci": core_row.get(
                    "Delta_persist_95CI", "NE"
                ),
                "tertile_rho_minimum_dwell_95ci": core_row.get(
                    "rho_minimum_dwell_95CI", "NE"
                ),
                "quartile_delta_persistence": _fmt_float(
                    metric_row.get("top_bottom_persistence_delta")
                ),
                "quartile_delta_similarity": _fmt_float(
                    metric_row.get("top_bottom_similarity_delta")
                ),
                "quartile_delta_minimum_dwell_scaled": _fmt_float(
                    metric_row.get("top_bottom_minimum_dwell_proxy_delta_scaled")
                ),
                "spearman_persistence": _fmt_float(
                    metric_row.get("spearman_persistence")
                ),
                "spearman_persistence_p": _fmt_float(
                    metric_row.get("spearman_persistence_p"), 3
                ),
                "spearman_similarity": _fmt_float(
                    metric_row.get("spearman_similarity")
                ),
                "spearman_similarity_p": _fmt_float(
                    metric_row.get("spearman_similarity_p"), 3
                ),
                "spearman_minimum_dwell_proxy": _fmt_float(
                    metric_row.get("spearman_minimum_dwell_proxy")
                ),
                "spearman_minimum_dwell_proxy_p": _fmt_float(
                    metric_row.get("spearman_minimum_dwell_proxy_p"), 3
                ),
            }
        )
    table_df = pd.DataFrame(rows)
    table_path = source_root / "tables" / "integrated_longitudinal_metrics_table.tsv"
    table_df.to_csv(table_path, sep="\t", index=False)
    return table_df


def route_table(scores, *, top_paths=6, max_depth=8):
    """Use the selected panel's exact eligibility, target ordering and path rule."""
    eligible = scores[scores["eligible_relobstq"].astype(bool)].copy()
    if eligible.empty:
        eligible = scores[np.isfinite(scores["R_star"])].copy()
    selected = eligible.sort_values(["R_star", "N_v"], ascending=[False, False]).head(
        top_paths
    )
    lookup = scores.set_index("state").to_dict(orient="index")
    rows = []
    for lane, (_, target) in enumerate(selected.iterrows(), 1):
        path = build_dominant_predecessor_path(
            str(target["state"]), lookup, max_depth=max_depth
        )
        for position, state in enumerate(path):
            info = lookup.get(state, {})
            rows.append(
                {
                    "route": lane,
                    "target_state": target["state"],
                    "position": position,
                    "state": state,
                    "event_added": (
                        event_added(path[position - 1], state) if position else ""
                    ),
                    "N_v": info.get("N_v", np.nan),
                    "R_star": info.get("R_star", np.nan),
                    "log2_R_star": info.get("log2_R_star", np.nan),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "route",
            "target_state",
            "position",
            "state",
            "event_added",
            "N_v",
            "R_star",
            "log2_R_star",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(config_path("longitudinal_tables.yaml"))
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["result_root"])
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    data = prepare_predictions(config)
    bins = summarize_bins(data, config)
    metrics = summarize_metrics(data, bins, config)
    data.to_csv(tables / "rstar_calibration_pair_level.tsv", sep="\t", index=False)
    bins.to_csv(tables / "rstar_calibration_bins.tsv", sep="\t", index=False)
    metrics.to_csv(tables / "rstar_calibration_metrics.tsv", sep="\t", index=False)
    write_integrated_metrics_table(root, config)
    for study in config["included_studies"]:
        source = (
            Path(config["source_result_root"]) / study / "tables" / "state_scores.tsv"
        )
        routes = route_table(
            pd.read_csv(source, sep="\t"),
            top_paths=int(config["analysis"]["top_paths_per_study"]),
            max_depth=int(config["analysis"]["maximum_predecessor_depth"]),
        )
        routes.to_csv(tables / f"topology_routes_{study}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
