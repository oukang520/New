"""Validate Experiment 6 numerical identities, success rules and figure layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.stats import wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_06.yaml")
    return parser.parse_args()


def paired_pvalue(table: pd.DataFrame, first: str, second: str) -> float:
    valid = table[[first, second]].dropna()
    if valid.empty or np.allclose(valid[first], valid[second]):
        return np.nan
    return float(wilcoxon(valid[first], valid[second], alternative="greater").pvalue)


def check(condition: bool, name: str, records: list[dict], detail: str) -> None:
    records.append(
        {
            "check": name,
            "passed": bool(condition),
            "detail": detail,
        }
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    figure = root / "figures" / "Figure_E6_bottleneck_recovery"
    required = [
        "true_theta.tsv",
        "truth_states.tsv",
        "lambda_cv.tsv",
        "repeat_metrics.tsv",
        "repeat_curves.tsv",
        "state_recovery_long.tsv",
        "estimated_theta_long.tsv",
        "representative_state_scores.tsv",
        "experiment_06_summary.tsv",
    ]
    records: list[dict] = []
    missing = [name for name in required if not (tables / name).exists()]
    check(not missing, "required_tables", records, "OK" if not missing else ";".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_06_validation.csv", index=False)
        raise SystemExit(1)

    simulation = config["simulation"]
    success = config["success"]
    scoring = config["state_scoring"]
    truth = pd.read_csv(tables / "truth_states.tsv", sep="\t")
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    states = pd.read_csv(tables / "state_recovery_long.tsv", sep="\t")
    curves = pd.read_csv(tables / "repeat_curves.tsv", sep="\t")
    theta = pd.read_csv(tables / "true_theta.tsv", sep="\t", index_col=0)
    estimated_theta = pd.read_csv(tables / "estimated_theta_long.tsv", sep="\t")
    representative = pd.read_csv(tables / "representative_state_scores.tsv", sep="\t")

    check(
        theta.shape == (15, 15),
        "true_theta_shape",
        records,
        f"shape={theta.shape}",
    )
    interaction_fraction = (
        np.count_nonzero(theta.to_numpy() - np.diag(np.diag(theta.to_numpy())))
        / (15 * 14)
    )
    check(
        0.08 <= interaction_fraction <= 0.18,
        "mixed_topology_sparsity",
        records,
        f"directed_off_diagonal_fraction={interaction_fraction:.4f}",
    )
    check(
        len(truth[truth["truth_class"] == "bottleneck"]) == 3
        and len(truth[truth["truth_class"] == "fast"]) == 3,
        "truth_state_counts",
        records,
        truth["truth_class"].value_counts().to_dict().__str__(),
    )
    check(
        set(np.round(truth["D_true"], 6)) == {
            float(simulation["bottleneck_dwell"]),
            float(simulation["fast_dwell"]),
        },
        "truth_dwell_values",
        records,
        f"values={sorted(truth['D_true'].unique())}",
    )
    check(
        len(metrics) == int(simulation["repeats"]),
        "repeat_count",
        records,
        f"observed={len(metrics)}, expected={simulation['repeats']}",
    )
    check(
        len(estimated_theta) == int(simulation["repeats"]) * 15 * 15,
        "mhn_refit_count",
        records,
        f"theta_rows={len(estimated_theta)}",
    )
    eligible = states["eligible"].astype(bool)
    identity = states.loc[eligible, "L_v"] / (
        states.loc[eligible, "F_hat"] + float(scoring["epsilon"])
    )
    normalized_identity = states.loc[eligible, "R_star"] / states.loc[eligible, "R_raw"]
    check(
        np.allclose(identity, states.loc[eligible, "R_raw"], rtol=1e-9, atol=1e-12),
        "R_raw_identity",
        records,
        f"maximum_error={np.max(np.abs(identity - states.loc[eligible, 'R_raw'])):.3g}",
    )
    median_by_repeat = states.loc[eligible].groupby("repeat")["R_star"].median()
    check(
        np.allclose(median_by_repeat, 1.0, rtol=1e-8, atol=1e-8),
        "R_star_median_normalization",
        records,
        f"range={median_by_repeat.min():.6f}-{median_by_repeat.max():.6f}",
    )
    check(
        np.isfinite(normalized_identity).all()
        and (states.loc[eligible, ["R_star", "occupancy_star"]] > 0).all().all(),
        "finite_positive_scores",
        records,
        f"eligible_rows={eligible.sum()}",
    )
    expected_curve_rows = int(simulation["repeats"]) * 2 * 101
    check(
        len(curves) == expected_curve_rows,
        "curve_grid_completeness",
        records,
        f"observed={len(curves)}, expected={expected_curve_rows}",
    )

    median_spearman = float(metrics["spearman_R_star"].median())
    median_auc = float(metrics["bottleneck_auc_R_star"].median())
    median_top5 = float(metrics["top5_precision_R_star"].median())
    check(
        median_spearman >= float(success["median_spearman_minimum"]),
        "spearman_success",
        records,
        f"median={median_spearman:.4f}, threshold={success['median_spearman_minimum']}",
    )
    check(
        median_auc >= float(success["median_bottleneck_auc_minimum"]),
        "auc_success",
        records,
        f"median={median_auc:.4f}, threshold={success['median_bottleneck_auc_minimum']}",
    )
    check(
        median_top5 >= float(success["median_top5_precision_minimum"]),
        "top5_success",
        records,
        f"median={median_top5:.4f}, threshold={success['median_top5_precision_minimum']}",
    )
    auc_p = paired_pvalue(
        metrics, "bottleneck_auc_R_star", "bottleneck_auc_occupancy"
    )
    check(
        np.isfinite(auc_p)
        and auc_p < float(success["paired_test_alpha"])
        and metrics["bottleneck_auc_R_star"].median()
        > metrics["bottleneck_auc_occupancy"].median(),
        "occupancy_baseline_improvement",
        records,
        (
            f"R*_median={metrics['bottleneck_auc_R_star'].median():.4f}, "
            f"occupancy_median={metrics['bottleneck_auc_occupancy'].median():.4f}, "
            f"p={auc_p:.3g}"
        ),
    )
    check(
        metrics["stable_bottlenecks"].median() == 3,
        "truth_state_observability",
        records,
        f"median_stable_bottlenecks={metrics['stable_bottlenecks'].median():.1f}",
    )
    check(
        set(representative["truth_class"]).issuperset({"bottleneck", "fast"}),
        "representative_truth_coverage",
        records,
        representative["truth_class"].value_counts().to_dict().__str__(),
    )

    figure_problems = []
    for suffix in [".png", ".pdf"]:
        if not figure.with_suffix(suffix).exists():
            figure_problems.append(f"missing_{suffix[1:]}")
    if figure.with_suffix(".png").exists():
        with Image.open(figure.with_suffix(".png")) as image:
            width, height = image.size
            if width < 4500 or height < 3200:
                figure_problems.append(f"low_resolution_{width}x{height}")
            array = np.asarray(image.convert("RGB"))
            edge_nonwhite = np.mean(
                np.concatenate(
                    [
                        array[:8].reshape(-1, 3),
                        array[-8:].reshape(-1, 3),
                        array[:, :8].reshape(-1, 3),
                        array[:, -8:].reshape(-1, 3),
                    ]
                )
                < 245
            )
            if edge_nonwhite > 0.02:
                figure_problems.append(f"possible_edge_clipping={edge_nonwhite:.4f}")
    check(
        not figure_problems,
        "figure_files_and_boundaries",
        records,
        "OK" if not figure_problems else ";".join(figure_problems),
    )

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_06_validation.csv", index=False)
    lines = [
        "# Experiment 6 Validation",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            f"- Overall: {'PASS' if result['passed'].all() else 'FAIL'}",
            f"- Passed checks: {int(result['passed'].sum())}/{len(result)}",
        ]
    )
    (root / "experiment_06_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
