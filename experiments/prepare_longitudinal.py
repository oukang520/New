"""Create patient-grouped out-of-fold longitudinal pair predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from relobstq_mhn.core.mhn import MhnFitConfig
from relobstq_mhn.core.scoring import ScoreThresholds
from relobstq_mhn.io import load_yaml
from relobstq_mhn.workflows.longitudinal_preparation import (
    LongitudinalPreparationConfig,
    prepare_longitudinal_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/longitudinal.yaml")
    parser.add_argument("--study", action="append", help="Prepare only the named study; repeatable.")
    args = parser.parse_args()
    raw = load_yaml(args.config)
    crossfit = raw["crossfit"]
    thresholds = ScoreThresholds(
        minimum_state_count=int(crossfit["minimum_state_count"]),
        minimum_inflow=float(crossfit["minimum_inflow"]),
        high_confidence_state_count=int(crossfit["high_confidence_state_count"]),
    )
    mhn = MhnFitConfig(
        lambda_multipliers=tuple(float(value) for value in crossfit["lambda_multipliers"]),
        cv_folds=int(crossfit["cv_folds"]),
        max_iterations=int(crossfit["max_iterations"]),
        relative_tolerance=float(crossfit["relative_tolerance"]),
        random_seed=int(crossfit["random_seed"]),
    )
    config = LongitudinalPreparationConfig(
        folds=int(crossfit["folds"]),
        random_seed=int(crossfit["random_seed"]),
        exclude_event_loss_pairs=bool(crossfit["exclude_event_loss_pairs"]),
        thresholds=thresholds,
        mhn=mhn,
    )
    studies = args.study or list(raw["prepared_inputs"])
    for study in studies:
        source = Path(raw["prepared_inputs"][study])
        metadata = pd.read_csv(source / "sample_metadata.tsv", sep="\t")
        matrix = pd.read_csv(source / "event_matrix.tsv", sep="\t")
        prepare_longitudinal_pairs(
            metadata,
            matrix,
            study_id=study,
            output_dir=Path(raw["preparation_root"]) / study,
            config=config,
            input_files=[source / "sample_metadata.tsv", source / "event_matrix.tsv"],
        )
        print(f"completed longitudinal cross-fitting: {study}")


if __name__ == "__main__":
    main()
