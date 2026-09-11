# SIRDwell: numerical manuscript experiments

This release follows the selected `figure_assembly_5fig_42panels` snapshot:
**42 experimental figures and one metric table, covering 14 experiment numbers**.
It supersedes older, conflicting experiment mappings and parameters. The exact
inventory, PDF hashes, numerical entry points, parameters and output tables are
in [the artifact map](docs/FIGURE_EXPERIMENT_MAP.tsv).

Only numerical methods, input/output handling, parameters, tests and supporting
contracts are retained. There are no plotters, rendering imports or image exports.
Raw patient records are not included. Existing aggregate reference results remain
under `reference_results/` for comparison.

## Method

For a stage/genotype state `v=(s,G)`:

```text
L_v     = N_v / N
F_hat_v = sum_u L_u * P(u -> v)
R_raw_v = L_v / (F_hat_v + epsilon)
R_star  = R_raw_v / median(R_raw among eligible states)
```

The primary predecessor set contains observed same-stage one-event predecessors.
Conditional next-event probabilities provide relative inflow mass; the score is
a dimensionless cohort-relative dwell tendency, not absolute elapsed time.
E3's cohort-weighted genotype edge support pools stages and is distinct from a
single predecessor-specific inflow term. That distinction is preserved in code.

## Installation

Use Python 3.11 or 3.12 with the official `mhn==1.2.3` backend:

```shell
python -m pip install -e ".[mhn,test]"
python -m pytest -q
```

No graphical dependency is required, including for E17.

## Running the selected experiments

Arrange the authorized inputs described in [DATA_AVAILABILITY](docs/DATA_AVAILABILITY.md).
Inspect the full ordered plan before executing it:

```shell
python experiments/run_all.py --dry-run
python experiments/run_all.py --workspace /path/to/data-and-results
```

Individual numerical contracts can also be run directly:

```shell
python -m sirdwell_experiments e03 --workspace /path/to/data-and-results
python -m sirdwell_experiments e09 --workspace /path/to/data-and-results
python -m sirdwell_experiments e13 --workspace /path/to/data-and-results
python experiments/run_cross_sectional.py --config configs/cross_sectional.yaml
python experiments/run_simulation.py --config configs/simulation.yaml
python experiments/run_secondary.py --config configs/secondary.yaml
python experiments/run_longitudinal.py --config configs/longitudinal.yaml
python -m sirdwell_experiments e17_tables --workspace /path/to/data-and-results
```

Relative data/output paths are resolved from the working directory (or `--workspace`).
Bundled numerical experiment parameters can be overridden with `--config`;
E1/E2 use `--experiment-config` and `--dataset-config`. Cross-sectional preparation
has a separate harmonized-data input contract. The ordered plan does not download
or silently synthesize missing data.

## Version-specific contracts

- E3/E4 preserve the selected historical transition-interface and sensitivity calculations.
- E5/E6/E10/E11/E14/E15B/E16 use the current frozen core workflows, including
  **5,000 samples per repeat and 60 repeats** for E6.
- E9 and E13 retain their resolved run parameters. E13 resamples patients with
  a locked inflow backbone. `e05_split_input` produces its original upstream
  score schema; those scores do not replace the current E5 panel's calculation.
- E15 contains only the retained inflow-pairing control (400 shuffles per cohort).
- E17 retains the selected GLASS, CRC-triplets and MNM-WashU analysis and its
  explicitly configured frequency/co-occurrence backbone. Its metric-table and
  dominant-predecessor route calculations are available without drawing code.

See [experiment mapping](docs/EXPERIMENT_MAPPING.md) and
[result contracts](docs/RESULT_CONTRACTS.md) for input dependencies and schemas.
Threshold differences between these contracts are intentional. No blanket
replacement of seeds, denominators or eligibility thresholds has been applied.

## Verification boundary

The test suite checks core calculations, numerical-only execution, stage-aggregation
semantics, locked-inflow splitting and coverage of all 43 selected artifacts.
[Numerical replay checks](docs/NUMERICAL_REPLAY_CHECKS.json) compare retained
calculations with the selected existing result tables. They do not claim a fresh
full cMHN refit, a raw longitudinal-cohort rerun or a full 60-repeat E6 rerun.
