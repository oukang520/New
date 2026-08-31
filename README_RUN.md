# Rel-ObsTQ-MHN Data Audit Pipeline

This project prepares cross-sectional tumor datasets for later Rel-ObsTQ-MHN analysis.
The current pipeline does not train MHN, does not compute R*, does not run survival
models, and does not download external data.

## Directory Contract

- Raw downloaded data stay under `data/`.
- Generated reports are written to `reports/`.
- Generated model inputs are written to `processed/`.
- Reusable scripts live under `src/`.
- Logs are written to `logs/`.

## Run The Full First-Pass Pipeline

Use a Python environment with pandas, numpy, and pyyaml installed.

```powershell
python src/audit_datasets.py --data-dir data --output-dir .
python src/build_event_matrix.py --input processed/standardized_mutations.csv --clinical processed/standardized_clinical.csv --id-level patient --min-frequency 0.03 --top-k-events 15 --include-cna no --output-dir .
python src/build_state_table.py --event-matrix processed/event_matrix.csv --metadata processed/sample_metadata.csv --id-level patient --min-state-count 5 --output-dir .
```

On this machine, `python` may be a Windows Store placeholder. The working command is:

```powershell
& 'D:\ai\Scripts\conda.exe' run python src/audit_datasets.py --data-dir data --output-dir .
& 'D:\ai\Scripts\conda.exe' run python src/build_event_matrix.py --input processed/standardized_mutations.csv --clinical processed/standardized_clinical.csv --id-level patient --min-frequency 0.03 --top-k-events 15 --include-cna no --output-dir .
& 'D:\ai\Scripts\conda.exe' run python src/build_state_table.py --event-matrix processed/event_matrix.csv --metadata processed/sample_metadata.csv --id-level patient --min-state-count 5 --output-dir .
```

## Outputs

Required audit outputs:

- `reports/file_inventory.csv`
- `reports/dataset_feasibility_table.csv`
- `reports/dataset_feasibility_report.md`
- `reports/column_mapping_report.csv`
- `reports/final_feasibility_summary.md`
- `processed/data_dictionary.csv`

Required model-prep outputs:

- `processed/event_matrix.csv`
- `processed/event_frequency.csv`
- `processed/sample_metadata.csv`
- `processed/state_table.csv`
- `processed/state_occupancy.csv`
- `reports/event_matrix_qc.md`
- `reports/state_table_qc.md`

Additional reproducibility outputs:

- `processed/standardized_mutations.csv`
- `processed/standardized_clinical.csv`
- `logs/*.log`

## Smoke Test

```powershell
python -m pytest tests/test_pipeline_smoke.py
```

If pytest is unavailable, install the requirements first.

## MHN Official Component

The official `mhn` Python package source snapshot has been downloaded under:

- `external/mhn/pypi/mhn-1.2.3.tar.gz`
- `external/mhn/github/LearnMHN-v1.2.3/LearnMHN-1.2.3/`

See `external/mhn/MHN_SOURCE.md` for source URLs, SHA256 hashes, license, and compatibility notes.
`mhn==1.2.3` requires Python `>=3.8,<3.13`, so create a Python 3.12 or 3.11 environment before
the later MHN training step.

## Build Experiment-Ready Datasets

This prepares the three AACR fine cancer-type subsets plus the five ICGC datasets into one
uniform schema for MHN and Rel-ObsTQ-MHN experiments:

```powershell
& 'D:\ai\Scripts\conda.exe' run python src/build_experiment_ready_datasets.py --data-dir data --processed-dir processed --output-dir processed\experiment_ready --min-frequency 0.03 --top-k 0 --min-state-count 5 --chunksize 250000
& 'D:\ai\Scripts\conda.exe' run python src/validate_experiment_ready.py --input-dir processed\experiment_ready
```

Per-dataset outputs live under `processed/experiment_ready/{DATASET}/`:

- `analysis_metadata.csv`
- `mutations_long.csv`
- `event_matrix.csv`
- `mhn_training_matrix.csv`
- `mhn_row_index_map.csv`
- `event_frequency.csv`
- `state_table.csv`
- `state_occupancy.csv`
- `dataset_manifest.json`
- `qc_report.md`

`mhn_training_matrix.csv` is the file to pass to `mhn.optimizers.Optimizer.load_data_from_csv()`.
It intentionally contains only binary event columns and no ID or clinical fields.

## Active Experiment Cohorts

The active cohort selection is stored in:

- `configs/selected_experiment_datasets.yaml`
- `reports/selected_experiment_datasets.md`

Selected datasets are `AACR_LUAD`, `AACR_COAD`, `AACR_IDC`, and `PACA-CA`.
The remaining processed datasets are retained for reproducibility but are excluded from the active experiments.
Each selected dataset is an independent full experiment: it receives its own MHN fit, transition
outputs, Rel-ObsTQ calculations, validation analyses, figures, metrics, and result directory.

## Run Experiments 1 And 2

Experiments 1 and 2 are run independently for `AACR_LUAD`, `AACR_COAD`, `AACR_IDC`,
and `PACA-CA`:

```powershell
& 'D:\ai\Scripts\conda.exe' run python src/run_experiments_01_02.py --experiment-config configs/experiments_01_02.yaml --dataset-config configs/selected_experiment_datasets.yaml
& 'D:\ai\Scripts\conda.exe' run python src/validate_experiments_01_02.py --experiment-config configs/experiments_01_02.yaml --dataset-config configs/selected_experiment_datasets.yaml
```

Results are written to `results/experiments_01_02/{DATASET}/`, with separate
`experiment_01_data_preparation` and `experiment_02_stage_sensitivity` directories.
Figures are exported as editable PDF and 600 dpi PNG.
