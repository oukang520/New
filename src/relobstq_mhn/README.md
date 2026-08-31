# Rel-ObsTQ-MHN

Core method code for estimating relative state dwell/accumulation from
cross-sectional mutation states with an MHN-derived evolutionary inflow
backbone.

This folder is designed to be uploaded as a standalone method-code supplement.
It contains the reusable method implementation, configuration snapshots,
documentation and minimal tests. The original project-level experiment scripts
remain outside this folder and are not required to inspect the core method API.

## Directory Layout

```text
relobstq_mhn/
  __init__.py
  core/
    states.py          # state/genotype parsing and occupancy primitives
    transitions.py     # MHN event-addition probabilities and F_hat inflow
    scoring.py         # R*, O* and qualitative state classification
    bootstrap.py       # bootstrap uncertainty and top-state stability
    topology.py        # dominant-predecessor paths and topology targets
    validation.py      # lightweight schema/audit helpers
    pipeline.py        # compact orchestration from occupancy + theta to scores
  data/
    processing.py      # method-facing preprocessing utilities
  simulation/
    generator.py       # cMHN-like simulation and dwell-truth generator
  configs/             # configuration snapshots used by the experiments
  docs/                # method API and figure-style notes
  tests/               # minimal method-level tests
  requirements.txt
```

## Core Method

For state `v`, the method uses:

- `L_v`: observed cross-sectional occupancy.
- `F_hat_v`: MHN-derived expected inflow from predecessor states.
- `R_raw_v = L_v / (F_hat_v + epsilon)`.
- `R*_v = R_raw_v / median(R_raw among eligible states)`.
- `O*_v = L_v / (Lhat_progression_v + epsilon)` as an auxiliary
  observation-enrichment residual.

`R*` is the primary innovation. It highlights states that are enriched relative
to their model-derived evolutionary accessibility, making it a relative
dwell/accumulation proxy rather than a raw frequency statistic.

## Minimal Import

Place the parent directory of `relobstq_mhn/` on `PYTHONPATH`, then:

```python
from relobstq_mhn import (
    ScoreThresholds,
    aggregate_inflow,
    compute_relative_dwell,
    probability_provider_from_theta,
    same_stage_one_step_edges,
)

provider = probability_provider_from_theta(theta, events)
edges = same_stage_one_step_edges(occupancy, events, provider)
inflow = aggregate_inflow(occupancy, edges, minimum_state_count=5)
scores, normalizer = compute_relative_dwell(inflow, ScoreThresholds())
```

## Data Processing

```python
from relobstq_mhn import build_experiment_ready_tables

tables = build_experiment_ready_tables(
    metadata,
    mutations,
    max_events=25,
    min_event_frequency=0.01,
    min_state_count=5,
)
```

## Simulation

```python
from relobstq_mhn import (
    SimulationConfig,
    create_sparse_theta,
    implant_dwell_truth,
    simulate_cohort_with_audit,
)

events = [f"E{i}" for i in range(15)]
theta = create_sparse_theta(events, sparsity=0.15, seed=1)
_, pilot_snapshot = simulate_cohort_with_audit(theta, events, config=SimulationConfig(samples=1000))
dwell_by_mask, truth = implant_dwell_truth(pilot_snapshot)
trajectory, snapshot = simulate_cohort_with_audit(theta, events, dwell_by_mask)
```

## Tests

From the parent directory of `relobstq_mhn/`:

```bash
python -m pytest relobstq_mhn/tests -q
```

The tests check state/genotype conversion, inflow aggregation, R* identity,
bootstrap output, topology paths, data processing and simulation output.

## Boundary

This folder contains reusable method code. It intentionally excludes full
experiment report generation and plotting-heavy analysis scripts. Those belong
to the project-level reproducibility layer.
