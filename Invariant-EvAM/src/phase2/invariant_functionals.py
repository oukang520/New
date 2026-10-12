"""Linear local functionals and interval decisions."""
from __future__ import annotations

import numpy as np


def identifiable_projection(coordinate: int, result) -> tuple[np.ndarray, float]:
    """Project a coordinate contrast onto the locally identifiable row space."""
    e = np.zeros(result.projected_jacobian.shape[1]); e[coordinate] = 1.0
    if result.nullspace.size:
        identified = e - result.nullspace @ (result.nullspace.T @ e)
    else:
        identified = e
    lost = float(np.linalg.norm(e - identified))
    norm = np.linalg.norm(identified)
    return (identified / norm if norm else identified), lost


def interval_decision(interval: tuple[float, float] | list[float], null_value: float = 0.0) -> str:
    low, high = interval
    if not np.isfinite(low + high):
        return "unknown"
    if low > null_value:
        return "positive"
    if high < null_value:
        return "negative"
    return "unknown"


def epsilon_breakdown(center: float, standard_error: float, loading: float, z: float = 1.96) -> float:
    """Largest bounded-deviation radius preserving a Wald sign conclusion."""
    margin = abs(center) - z * standard_error
    if margin <= 0:
        return 0.0
    return float("inf") if loading == 0 else float(margin / abs(loading))

