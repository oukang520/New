# Phase-0 v3 Final Audit

## Binary decision

**PASS**

## Closure checklist

| v2 requirement | objective check | status |
|---|---|---|
| full p15 to official-cMHN to theta to `F_hat` to `R*` | completed for LUAD, COAD and IDC; finite theta; clean manifests | PASS |
| regenerate E4/E10/E11/E14/E15A/E15B/E16 | current outputs frozen for all three cohorts | PASS |
| canonical E6 continuous gradient | 60 repeats, five dwell levels, per-level coverage and paired metrics frozen | PASS |
| canonical E7 robustness | 36 conditions, 360 repeats, explicit oracle-backbone contract frozen | PASS |
| result-independent E17 selection | structural eligibility contract forbids outcome metrics | PASS |
| disclose weak/negative longitudinal evidence | BRCA-MSK, ALP-breast and strict official-cMHN sensitivity retained | PASS |
| manuscript-facing E5 uses 500 bootstrap | current 500-replicate outputs and aggregate values synchronized | PASS |
| unify manuscript-control files | seven controls use the same frozen evidence and terminology | PASS |
| final hash freeze | 99 evidence files indexed by SHA-256 | PASS |
| final tests | 17 passed | PASS |

## Scientific grading

- Core method definition and implementation: strong.
- Continuous simulated dwell-gradient recovery: successful.
- Real cross-sectional computability and component/ablation controls: successful.
- Topology robustness: supportive but modest and condition-dependent.
- External longitudinal consistency: directionally supportive in the selected
  analysis, but limited by fallback-backbone estimation, cohort heterogeneity,
  small changed classes, and mixed strict sensitivity results.

The evidence supports the claim that `R*` extracts a relative state-level
dwell/stasis signal beyond occupancy alone under the stated model. It does not
support absolute-time prediction or universal longitudinal discrimination.

