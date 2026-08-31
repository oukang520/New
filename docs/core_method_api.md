# Rel-ObsTQ-MHN Core Method Code

This document describes the experiment-independent method package added under
`src/relobstq_mhn/`. The original `run_experiment_*.py` scripts are unchanged
and remain the audited reproduction path for Experiments 1-16. The new package
is a clean method layer intended for SCI submission, reuse and future refactors.

## Package Layout

| Module | Purpose |
|---|---|
| `states.py` | Canonical state/genotype parsing, binary genotype conversion and occupancy table construction. |
| `data_processing.py` | Method-facing preprocessing: mutation normalization, event-panel selection, event matrix and state table construction. |
| `transitions.py` | MHN event-addition probabilities, one-step predecessor edges and state-level inflow `F_hat`. |
| `scoring.py` | Relative dwell score `R*`, observation enrichment `O*`, and interpretation flags. |
| `bootstrap.py` | Multinomial bootstrap uncertainty and top-state stability for `R*`. |
| `topology.py` | Dominant-predecessor paths and topology target-state selection. |
| `simulation.py` | cMHN-like sparse theta generation, dwell-truth implantation and audited trajectory/snapshot simulation. |
| `validation.py` | Lightweight schema and audit helpers. |
| `pipeline.py` | Minimal composition helper from occupancy plus cMHN `theta` to scored states. |

## Core Mathematical Objects

For state `v`:

- `L_v`: observed cross-sectional occupancy.
- `F_hat_v`: MHN-derived relative inflow from observed predecessor states.
- `R_raw_v = L_v / (F_hat_v + epsilon)`.
- `R*_v = R_raw_v / median(R_raw among eligible states)`.
- `O*_v = L_v / (Lhat_progression_v + epsilon)`, where `Lhat_progression`
  comes from a progression-only expected occupancy model.

`R*` is the primary innovation: it estimates relative dwell/accumulation after
controlling for model-derived evolutionary accessibility.

## Minimal Usage

```python
import pandas as pd
from relobstq_mhn import (
    ScoreThresholds,
    aggregate_inflow,
    compute_relative_dwell,
    probability_provider_from_theta,
    same_stage_one_step_edges,
)

events = ["TP53", "KRAS", "EGFR"]
occupancy = pd.read_csv("state_occupancy.tsv", sep="\t")
theta = ...  # cMHN theta matrix in the same event order

provider = probability_provider_from_theta(theta, events)
edges = same_stage_one_step_edges(occupancy, events, provider)
inflow = aggregate_inflow(occupancy, edges, minimum_state_count=5)
scores, normalizer = compute_relative_dwell(inflow, ScoreThresholds())
```

## Data Processing Layer

The preprocessing layer converts raw analysis-unit metadata and long mutation
tables into method-ready objects:

```python
from relobstq_mhn import build_experiment_ready_tables

tables = build_experiment_ready_tables(
    metadata,
    mutations,
    max_events=25,
    min_event_frequency=0.01,
    min_state_count=5,
)

events = tables["events"]
event_matrix = tables["event_matrix"]
state_table = tables["state_table"]
state_occupancy = tables["state_occupancy"]
```

This code is intentionally generic. Cohort-specific extraction, OncoTree
filtering and file naming remain in the experiment/data scripts.

## Simulation Layer

The simulation layer supports controlled validation where true dwell
multipliers are known:

```python
from relobstq_mhn import (
    SimulationConfig,
    create_sparse_theta,
    implant_dwell_truth,
    simulate_cohort_with_audit,
)

events = [f"E{i}" for i in range(15)]
theta = create_sparse_theta(events, sparsity=0.15, seed=1)

pilot_trajectory, pilot_snapshot = simulate_cohort_with_audit(
    theta,
    events,
    config=SimulationConfig(samples=1000, random_seed=2),
)
dwell_by_mask, truth = implant_dwell_truth(pilot_snapshot)

trajectory, snapshot = simulate_cohort_with_audit(
    theta,
    events,
    dwell_by_mask,
    config=SimulationConfig(samples=1000, random_seed=3),
)
```

The two exported tables have different roles: `trajectory` audits the simulated
evolutionary path and dwell duration of each synthetic sample, while `snapshot`
is the cross-sectional input used by downstream scoring.

## Input Schema

The core inflow functions expect an occupancy table with:

- `state`: canonical `stage::genotype` identifier.
- `stage`: compartment label such as `primary` or `metastatic`.
- `genotype`: `WT` or a `+`-joined event set.
- `N_v`: observed count.
- `L_v`: observed occupancy fraction.

Predecessor edge tables contain:

- `source_state`
- `target_state`
- `edge_probability`
- `inflow_contribution`
- `predecessor_type`

## Submission Boundary

This package deliberately excludes:

- plotting code;
- cohort-specific thresholds and display choices;
- file paths tied to the current workstation;
- experiment-specific report generation.

Those remain in the experiment scripts. The method package is the reusable
scientific core; the experiment scripts are the audited analysis applications.
