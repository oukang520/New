# Phase-0 Reproducibility Audit

## Scope and repository state

- Historical workspace: `D:/project/Rel_ObsHN` (not a Git repository).
- Frozen public repository: `public_release/RelObsTQ_MHN_reproducible_code`.
- Original frozen commit and initial HEAD: `e4215608dd394581da19e1ae0d8a0206c4d33798`.
- Initial frozen-repository state: clean; no post-frozen commit existed.
- Historical `results/`, scripts and configs have no locally verifiable commit and are labeled legacy where applicable.
- All reruns were isolated under `outputs/phase0_reaudit_2026/`; historical results were not overwritten.

## Issue resolution table

| issue | status | evidence | action | remaining caveat |
|---|---|---|---|---|
| Declared preprocessing package absent from frozen Git tree | RESOLVED | manifest hashes and package imports | restored exact manifest-matching files | original commit remains non-self-contained |
| Python 3.11/3.12 + official MHN execution | RESOLVED_WITH_CAVEAT | `REPRODUCIBILITY_TEST_REPORT.md` | built `mhn==1.2.3`, ran tests and smoke fits in both versions | Windows may need x64 compiler setup |
| No canonical fixed-panel preparation | RESOLVED | exact p15 SHA matches | added preparation workflow/config/runner | provider-raw extraction requires licensed data |
| Screening versus model-panel event counts conflated | RESOLVED | manifests and E1 p15 tables | documented 25/25/17 screening versus 15/15/15 model panels | none |
| Result directories lacked runtime/input provenance | RESOLVED | `run_metadata.json` in new outputs | added commit/environment/input-hash metadata | Git dirty state can be null if Git executable is unavailable |
| E5 200 versus 500 bootstrap conflict | RESOLVED_WITH_CAVEAT | exact 200 reproduction and 500 reruns | preserved both, recommended 500 | intervals remain conditional on fixed theta |
| `F_hat` described ambiguously | RESOLVED | code-level transition audit | fixed terminology/documentation | definition intentionally remains conditional, same-stage, one-step |
| E17 fallback versus OOF official cMHN conflict | RESOLVED_WITH_CAVEAT | old fit metadata and new fold audit | ran 15 official fold fits and patient-cluster evaluation | current evidence is mixed and not interchangeable with legacy |
| E17 fallback source mislabeled | RESOLVED | implementation inspection | renamed to `genotype_median_across_stages` | fallback still pools stages by design |
| E17 row bootstrap ignored repeated-patient dependence | RESOLVED | evaluation-code inspection | patient-cluster bootstrap when patient ID exists | point estimates unchanged |
| `run_all.py` implied all E1-E17 coverage | RESOLVED_WITH_CAVEAT | runner/function inventory | renamed/documented as current canonical core workflows | several legacy experiments still lack canonical runners |
| E12 favorable-result selection risk | RESOLVED_WITH_CAVEAT | `CLINICAL_RESULT_AUDIT.tsv` | retained all overall and subgroup directions | E12 remains legacy-only and associative |
| E16 route interpretation | RESOLVED | `TOPOLOGY_ROUTE_AUDIT.tsv` | recorded every edge and selection rule | representative route, not phylogeny |

## Cross-sectional preparation and event panels

| cohort | analysis units | unique patients | screening events | final model events | p15 SHA-256 |
|---|---:|---:|---:|---:|---|
| AACR_LUAD | 27,148 | 22,777 | 25 | 15 | `8aed973e43f6364836f3627862ef568edfc9cf0b5b8ab0a9175255f49ac4b956` |
| AACR_COAD | 12,693 | 12,141 | 25 | 15 | `d59059891ec79e44364e9d33d73c2c6dda221c3100fa34c53b050351737e8959` |
| AACR_IDC | 10,964 | 10,177 | 17 | 15 | `455f45f59a149072ad163c70fc860962a661ed57b6d407bdc461986927c88b7c` |

All regenerated p15 matrices are byte-identical to historical E1. The final event names are fixed in `configs/cross_sectional_preparation.yaml`.

## MHN and estimator boundary

Historical E3 metadata confirms independent official `mhn==1.2.3` cMHN fits, L1 penalty, five-fold CV, 1-SE selection, CPU, seed `20260624`, finite `15 x 15` theta, and optimizer convergence. Current E17 fold fits use official cMHN and patient-grouped training sets. Full details are in `MHN_FIT_AUDIT.tsv`.

`F_hat` is relative expected inflow mass from conditional next-event probabilities, not an absolute CTMC flux. Bootstrap uncertainty is conditional on fixed theta. See the dedicated theoretical and bootstrap audits.

## Biological and clinical boundaries

- E8 module-level q-values are all 1.0 in the historical table. Biological coherence may be described as plausibility, not statistically significant enrichment.
- E12 shows mixed subgroup directions and tiny incremental C-index gains; it is secondary association evidence, not dwell truth or causality.
- E16 paths trace maximum-inflow dominant predecessors among observed same-stage states and are not complete phylogenetic trees.

## Canonical workflow coverage

Current runners cover fixed-panel preparation, cross-sectional cMHN/R*, continuous-gradient simulation, E11/E15/E16-style secondary tables, and E17 preparation/evaluation. E2, legacy enhanced E6, E7, E8, E9, E10, E12, E13 and E14 do not all have complete canonical runner/config/result contracts. Their numerical status is maintained in `RESULT_PROVENANCE_MATRIX.tsv` rather than being inferred from reusable functions.

## Phase-0 decision

Core package execution and the two highest-risk provenance conflicts (E5 and E17) are resolved. Full manuscript-number regeneration is not yet complete because several evidence rows remain legacy-only and the scientifically stronger E17 result changes the conclusion from uniformly positive to mixed.
