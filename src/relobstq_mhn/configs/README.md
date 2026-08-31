# Configuration Snapshots

This directory contains YAML configuration snapshots from the full Rel-ObsTQ-MHN
analysis chain. They are included so that a standalone upload of the
`relobstq_mhn` folder preserves method thresholds, cohort definitions and figure
style choices used by the experiments.

Important files:

- `selected_experiment_datasets.yaml`: retained real-cohort set.
- `figure_style.yaml`: shared figure-style configuration.
- `experiment_04.yaml`: relative inflow settings.
- `experiment_05.yaml`: R*/O* state-score settings.
- `experiment_06*.yaml`, `experiment_07*.yaml`, `experiment_09.yaml`: simulation
  validation settings.
- `experiment_16.yaml`: real topology display settings.

These files are snapshots for transparency. The reusable Python methods do not
require a specific file layout or global config object.
