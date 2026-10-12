from __future__ import annotations

import numpy as np

from phase2.jacobian_identifiability import analyze_environments
from phase2.profile_intervals import fit_joint, profile_coordinate
from phase2.sensitivity_delta import population_coordinate_interval


def diagnose_identifiability(model, environments, theta, phis=None, tol: float = 1e-8):
    """Return the profiled tangent certificate at a stated parameter/operator point."""
    if phis is None:
        phis = [np.zeros(env.nuisance_count) for env in environments]
    outputs = [env.probabilities_and_jacobians(model, theta, phi)
               for env, phi in zip(environments, phis)]
    return analyze_environments(outputs, tol=tol)


def fit(model, environments, counts, **kwargs):
    """Fit only after attaching the identification certificate at the estimate."""
    fitted = fit_joint(model, environments, counts, **kwargs)
    certificate = diagnose_identifiability(model, environments, fitted.theta, fitted.phis)
    return {"fit": fitted, "identifiability": certificate,
            "point_interpretable": certificate.rank == model.parameter_count}


def profile_functional(model, environments, counts, coordinate: int, grid, **kwargs):
    """Likelihood-ratio profile for a coordinate functional."""
    return profile_coordinate(model, environments, counts, coordinate, np.asarray(grid), **kwargs)


def sensitivity_analysis(effective_parameters, directions, coordinate: int, epsilons,
                         theta_bound: float = 3.0, phi_bound: float = 2.0):
    """Bounded-Delta identified intervals across a sensitivity grid."""
    rows = []
    for epsilon in epsilons:
        low, high, feasible = population_coordinate_interval(
            effective_parameters, directions, coordinate, float(epsilon), theta_bound, phi_bound
        )
        rows.append({"epsilon": float(epsilon), "lower": low, "upper": high,
                     "feasible": feasible,
                     "decision": "positive" if feasible and low > 0 else
                                 "negative" if feasible and high < 0 else "unknown"})
    return rows


def operator_audit(environments, model=None) -> list[dict[str, object]]:
    """Check the implementation contract of supplied panel/ascertainment operators."""
    rows = []
    for i, env in enumerate(environments):
        mask = getattr(env, "panel_mask", None)
        features = getattr(env, "features", None)
        directions = getattr(env, "directions", None)
        if mask is None and model is not None:
            covered = model.d
        elif mask is None:
            covered = None
        else:
            covered = int(np.asarray(mask, bool).sum())
        finite = True
        if features is not None:
            finite = bool(np.isfinite(np.asarray(features)).all())
        if directions is not None:
            finite = finite and bool(np.isfinite(np.asarray(directions)).all())
        rows.append({"environment": getattr(env, "name", f"environment_{i}"),
                     "nuisance_parameters": env.nuisance_count,
                     "covered_events": covered, "finite_operator": finite})
    return rows
