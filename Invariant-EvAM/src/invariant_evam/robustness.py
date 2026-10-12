from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from phase2.invariant_functionals import interval_decision


@dataclass(frozen=True)
class RobustnessCell:
    epsilon: float
    eta: float
    lower: float
    upper: float
    decision: str


def effective_error_interval(center: float, epsilon: float, eta: float,
                             operator_loading: float = 1.0) -> tuple[float, float]:
    """Sharp coordinate interval under an additive local effective-error contract.

    The contract is ``|delta_j| <= epsilon`` for biological deviation and
    ``|r_j| <= operator_loading * eta`` for the effective generator-scale error
    induced by operator misspecification.  It is deliberately reported in the
    parameter scale and is not a metric on arbitrary operators.
    """
    radius = float(epsilon) + abs(float(operator_loading)) * float(eta)
    return float(center - radius), float(center + radius)


def robustness_grid(center: float, epsilons, etas, operator_loading: float = 1.0) -> list[RobustnessCell]:
    out = []
    for epsilon in epsilons:
        for eta in etas:
            low, high = effective_error_interval(center, epsilon, eta, operator_loading)
            out.append(RobustnessCell(float(epsilon), float(eta), low, high,
                                      interval_decision((low, high))))
    return out


def linear_functional_interval(vector: np.ndarray, center: np.ndarray,
                               epsilon: float, eta: float) -> tuple[float, float]:
    """L-infinity robust interval for a linear functional a^T theta."""
    a = np.asarray(vector, float)
    value = float(a @ np.asarray(center, float))
    radius = float(np.linalg.norm(a, 1) * (epsilon + eta))
    return value - radius, value + radius
