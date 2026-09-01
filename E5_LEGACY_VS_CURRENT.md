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

## Final current full-pipeline result

The final manuscript pipeline was subsequently rerun from the exact p15 inputs
through a new official-cMHN fit and 500 conditional count bootstraps. Because
this run uses the final fitted theta and final eligibility set, it supersedes
the earlier isolated v0.2.0-threshold refinement above for manuscript use.

| cohort | eligible states | high-confidence states | observed top-10 high-confidence stability |
|---|---:|---:|---:|
| AACR_LUAD | 416 | 233 | 0.6704 |
| AACR_COAD | 282 | 147 | 0.6758 |
| AACR_IDC | 167 | 111 | 0.7192 |

The aggregate stability is the mean
`high_confidence_top_bootstrap_stability` among the ten highest observed `R*`
states that satisfy the high-confidence count threshold. Per-state values are
frozen under `reference_results/final_manuscript_evidence/cross_sectional/`.

## Decision

The historical 200-replicate E5 is exactly reproducible and is retained only
for provenance. The final manuscript uses the current full-pipeline,
500-replicate values immediately above and discloses that theta and edge
probabilities are fixed during bootstrap.

Status: `RESOLVED_CURRENT_WITH_CONDITIONAL_UNCERTAINTY`.
