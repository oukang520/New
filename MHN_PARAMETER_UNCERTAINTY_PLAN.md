# MHN Parameter Uncertainty Plan

Recommended design: stratified patient subset refits rather than an immediate nested bootstrap.

1. Draw 20-30 independent 80% patient subsets per cohort.
2. Rebuild occupancy and independently fit official cMHN with the prespecified p15 panel and lambda grid.
3. Recompute `R*` with no full-cohort theta reuse.
4. Compare common eligible states with Spearman correlation, top-10 overlap/Jaccard, and sign agreement around `R*=1`.
5. Report optimizer failures and boundary-lambda selections without deletion.

This analysis is computationally feasible for LUAD/IDC but potentially expensive for COAD. It is currently classified `RECOMMENDED_FUTURE_ANALYSIS`; no result is claimed.
