"""Bounded biological-deviation sensitivity sets in the local shift contract."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def population_coordinate_interval(effective_parameters: list[np.ndarray], directions: list[np.ndarray],
                                   coordinate: int, epsilon: float, theta_bound: float = 3.0,
                                   phi_bound: float = 2.0) -> tuple[float, float, bool]:
    """Profile theta_j under eta_e = theta + B_e phi_e + delta_e, |delta_e|_inf <= epsilon."""
    p = len(effective_parameters[0])
    qs = [np.asarray(b).reshape(p, -1).shape[1] for b in directions]
    total = p + sum(qs)
    rows, upper = [], []
    offset = p
    for eta, b, q in zip(effective_parameters, directions, qs):
        b = np.asarray(b).reshape(p, q)
        design = np.zeros((p, total)); design[:, :p] = np.eye(p)
        design[:, offset:offset + q] = b; offset += q
        rows.extend([design, -design]); upper.extend([eta + epsilon, -eta + epsilon])
    a_ub = np.vstack(rows); b_ub = np.concatenate(upper)
    bounds = [(-theta_bound, theta_bound)] * p + [(-phi_bound, phi_bound)] * sum(qs)
    objective = np.zeros(total); objective[coordinate] = 1.0
    low = linprog(objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    high = linprog(-objective, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not (low.success and high.success):
        return np.nan, np.nan, False
    return float(low.fun), float(-high.fun), True

