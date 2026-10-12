"""Weak-identification diagnostics based on projected Fisher geometry."""
from __future__ import annotations

import numpy as np


def projected_fisher(projected_jacobian: np.ndarray, sample_sizes: list[int] | None = None) -> np.ndarray:
    j = np.asarray(projected_jacobian)
    # Blocks are already probability-whitened. Equal block weighting is the
    # population design score; sample weighting is handled by row scaling upstream.
    return j.T @ j


def identification_strength(result) -> dict[str, float]:
    s = result.singular_values
    positive = s[s > (s[0] * 1e-8 if len(s) else 1e-8)]
    return {
        "rank": int(result.rank),
        "sigma_min_all": float(s[-1]) if len(s) else 0.0,
        "sigma_min_identified": float(positive[-1]) if len(positive) else 0.0,
        "sigma_max": float(s[0]) if len(s) else 0.0,
        "condition_identified": float(positive[0] / positive[-1]) if len(positive) else float("inf"),
    }

