"""Generate result tables for falsification, information-gain, and topology analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from relobstq_mhn.io import ResultWriter, load_yaml
from relobstq_mhn.workflows.controls import inflow_pairing_falsification, matched_decoy_test
from relobstq_mhn.workflows.secondary import information_gain_summary
from relobstq_mhn.workflows.topology import topology_route_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/secondary.yaml")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    analysis = raw["analysis"]
    for cohort_index, cohort in enumerate(raw["cohorts"]):
        source = Path(raw["cross_sectional_root"]) / cohort / "tables" / "state_scores.tsv"
        scores = pd.read_csv(source, sep="\t")
        writer = ResultWriter(Path(raw["output_root"]) / cohort)
        details, decoy_summary = matched_decoy_test(
            scores,
            top_k=int(analysis["top_k"]),
            quantile_bins=int(analysis["matched_decoy_quantile_bins"]),
            minimum_decoys=int(analysis["minimum_decoys"]),
        )
        shuffled, shuffle_summary = inflow_pairing_falsification(
            scores,
            top_k=int(analysis["top_k"]),
            replicates=int(analysis["inflow_shuffle_replicates"]),
            seed=int(analysis["random_seed"]) + cohort_index,
        )
        writer.table("matched_decoy_details", details)
        writer.table("matched_decoy_summary", decoy_summary)
        writer.table("inflow_shuffle_replicates", shuffled)
        writer.table("inflow_shuffle_summary", shuffle_summary)
        writer.table("information_gain_summary", information_gain_summary(scores, top_k=int(analysis["top_k"])))
        writer.table("topology_routes", topology_route_table(scores, target_count=int(analysis["topology_routes"])))
        writer.manifest()
        print(f"completed secondary validations: {cohort}")


if __name__ == "__main__":
    main()
