# Experiment 17 transparent cohort supplement

This directory freezes longitudinal cohorts that were screened or analyzed but
are not part of the selected three-cohort external-consistency table. Cohort
disposition follows the structural rules in `configs/longitudinal.yaml` and
does not use AUC, persistence contrast, AP lift, or dwell-correlation direction.

`BRCA-MSK` satisfies the numerical rule and is therefore reported as an
eligible heterogeneous challenge cohort even though its results are weak.
`ALP-breast` is not eligible because active-treatment cfDNA sampling violates
the comparable observational dwell-target contract; its negative pilot remains
reported to make the screening history transparent.

The selected Experiment 17 result remains an **external longitudinal
consistency analysis** based on the frozen full-cohort fallback backbone. It is
not a calibrated calendar-time prediction test and not patient-grouped
out-of-fold official-cMHN validation.
