
# Data availability

The cross-sectional analysis uses AACR Project GENIE v18.0-public-derived LUAD,
COAD, and IDC cohorts. GENIE files must be obtained from the authorized
provider under its data-use terms; patient-level records are not redistributed.

The public canonical preparation entry begins from two cohort-filtered,
provider-authorized tables under `data/cross_sectional_harmonized/COHORT/`:

| file | required columns | unit |
|---|---|---|
| `analysis_metadata.csv` | `analysis_id`, `patient_id`, `sample_id`, `stage_group` | one tumor analysis unit/sample per row |
| `mutations_long.csv` | `analysis_id`, `gene`; optional `consequence` | one mutation event per row |

`analysis_id` must be unique in metadata. Multiple samples from one patient are
allowed and must not be described as unique patients. Mutation events are
upper-cased, functional consequences are retained, and duplicate
analysis-unit/gene records are collapsed. Stage text is normalized to the
method stage groups by `standardize_stage_group`.

Run:

```powershell
python experiments/prepare_cross_sectional.py
```

This generates `mhn_training_matrix.csv`, `mhn_row_index_map.csv`, and
`state_table.csv` under `outputs/prepared_cross_sectional/COHORT/`, plus a
fixed event-panel table, QC, resolved config, input SHA-256 hashes, environment
metadata, and output manifest.

The provider-raw-to-harmonized extraction layer remains dataset/provider
specific and cannot be executed without licensed raw data. It must filter
GENIE v18.0-public to LUAD, COAD, and IDC, retain tumor samples as analysis
units, preserve patient/sample identifiers, normalize clinical stage, and
construct the two tables above. This boundary is `REQUIRES_DATA`, not silently
claimed as reproduced by the public package.

The selected external longitudinal consistency analysis uses GLASS,
colorectal primary-metastasis triplets, and MNM-WashU. Place their cBioPortal
exports under
`Data/longitudinal_public/cbioportal/difg_glass/`,
`Data/longitudinal_public/cbioportal/coadread_mskcc/`, and
`Data/longitudinal_public/cbioportal/mnm_washu_2016/`. Each directory must at
least provide `data_mutations.txt`, `data_clinical_sample.txt`, and
`data_clinical_patient.txt`; study-specific timeline or panel files are used
when present. The selected legacy runner constructs its event matrices and
ordered pairs from these files. Raw patient-level records are not
redistributed.

The screening preprocessing retained 25, 25, and 17 events for LUAD, COAD,
and IDC respectively. The final independently fitted cMHN models use a
prespecified 15-event cancer-specific panel for each cohort. These are two
different selection stages and must not be reported as one event count.
