# E5 Legacy Versus Current

## Provenance

Historical E5 was produced by `src/run_experiment_05.py` with `configs/experiment_05.yaml`, official cMHN theta inherited from E3, E4 `rule_a_one_step` counts/edges, seed `20260625 + 5000 + sum(ord(cohort))`, and 200 multinomial bootstrap replicates. Theta and edge probabilities were fixed.

The frozen v0.2.0 defaults instead specify 500 replicates, seed `20260630`, `minimum_inflow=1e-8`, and `epsilon=1e-12`. Therefore the difference is not solely 200 to 500 replicates.

## Exact reproduction check

The current core bootstrap was run against the historical E4 inputs with the historical thresholds and seed. Both reported stability metrics reproduced to floating-point precision:

| cohort | legacy top stability | recomputed 200 | legacy high-confidence stability | recomputed 200 |
|---|---:|---:|---:|---:|
| AACR_LUAD | 0.1390 | 0.1390 | 0.6235 | 0.6235 |
| AACR_COAD | 0.0625 | 0.0625 | 0.6745 | 0.6745 |
| AACR_IDC | 0.4770 | 0.4770 | 0.6755 | 0.6755 |

## 500-replicate refinement

| cohort | legacy-threshold top stability | legacy-threshold high-confidence stability | v0.2.0-threshold top stability | v0.2.0-threshold high-confidence stability |
|---|---:|---:|---:|---:|
| AACR_LUAD | 0.1400 | 0.6252 | 0.1198 | 0.6294 |
| AACR_COAD | 0.0640 | 0.6772 | 0.0676 | 0.6862 |
| AACR_IDC | 0.4850 | 0.6846 | 0.4862 | 0.6796 |

All detailed outputs are isolated under `outputs/phase0_reaudit_2026/E5_legacy_vs_current/`. The exact comparison is `e5_comparison.tsv`.

## Decision

The historical E5 is reproducible. Increasing bootstrap replicates does not change the qualitative conclusion. For a new manuscript table, use the 500-replicate conditional-bootstrap results and disclose that theta is fixed. Preserve the 200-replicate values as legacy provenance.

Status: `RESOLVED_WITH_CAVEAT`.
