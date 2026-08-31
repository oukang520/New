"""Evaluate leakage-controlled R* predictions in public longitudinal cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.longitudinal import LongitudinalConfig, evaluate_longitudinal_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longitudinal.yaml")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    missing = [Path(path) for path in raw["pair_tables"] if not Path(path).is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing prepared longitudinal pair tables:\n{joined}")
    pairs = pd.concat([pd.read_csv(path, sep="\t") for path in raw["pair_tables"]], ignore_index=True)
    columns = raw["columns"]
    analysis = raw["analysis"]
    config = LongitudinalConfig(
        study_column=columns["study"],
        score_column=columns["score"],
        persistence_column=columns["persistence"],
        dwell_proxy_column=columns["minimum_dwell"],
        lower_quantile=float(analysis["lower_quantile"]),
        upper_quantile=float(analysis["upper_quantile"]),
        bootstrap_replicates=int(analysis["bootstrap_replicates"]),
        random_seed=int(analysis["random_seed"]),
    )
    evaluate_longitudinal_pairs(pairs, output_dir=raw["output_root"], config=config)
    print(f"completed longitudinal workflow: {raw['output_root']}")


if __name__ == "__main__":
    main()
