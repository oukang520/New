# Theoretical Quantity Audit

This audit describes the implemented quantity, without altering it.

## Implemented transition quantity

For an observed predecessor genotype `u`, the code computes the log hazard for each absent event `j` as

```text
theta[j,j] + sum(theta[j,k] for events k present in u)
```

and applies a softmax over absent events. Therefore `P(u -> v | theta)` is the conditional probability that a particular absent event is the next added event, conditional on an event addition occurring.

## Answers to the required questions

1. **Is `F_hat` absolute flux?** No.
2. **Does it contain the absolute exit rate?** No. Softmax removes the total rate scale.
3. **Is it conditional on the next event occurring?** Yes.
4. **Is normalization only over absent events?** Yes.
5. **Are only observed predecessor states used?** Yes, in the primary implementation.
6. **Are only same-stage one-event edges allowed?** Yes.
7. **Do stage transitions enter primary `F_hat`?** No.
8. **Permitted interpretation:** relative expected inflow mass under an observed-state, same-stage, one-addition MHN backbone; `R*` is relative over-occupancy or persistence tendency against that modeled inflow.
9. **Prohibited interpretation:** absolute CTMC flux, absolute calendar dwell time, a complete tumor phylogeny, causal transition frequency, or a stage-transition model.

The implemented estimator is

```text
L_v = N_v / N
F_hat_v = sum_u L_u * P(next added event maps u to v | theta)
R_raw_v = L_v / (F_hat_v + epsilon)
R*_v = R_raw_v / median(R_raw among eligible states)
```

The median normalization makes `R*` dimensionless and cohort-relative. Comparisons across cohorts require care because panels, state support, and predecessor coverage differ.

## Terminology decision

Use **MHN-derived conditional evolutionary inflow** or **relative expected inflow mass**. Do not call `F_hat` an absolute rate or absolute flux. Use **relative dwell/stasis proxy** for `R*`.

Status: `RESOLVED`.
