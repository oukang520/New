"""Structured ascertainment and assay operators for Phase 2."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .interaction_generator import SparseInteractingCTMC


def aggregation_matrix(states: np.ndarray, panel_mask: np.ndarray | None) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    if panel_mask is None:
        panel_mask = np.ones(states.shape[1], dtype=bool)
    panel_mask = np.asarray(panel_mask, dtype=bool)
    observed = states[:, panel_mask]
    unique = sorted({tuple(row) for row in observed})
    index = {row: i for i, row in enumerate(unique)}
    matrix = np.zeros((len(unique), len(states)))
    for full_i, row in enumerate(observed):
        matrix[index[tuple(row)], full_i] = 1.0
    return matrix


@dataclass(frozen=True)
class TiltPanelEnvironment:
    """Positive state-dependent exponential tilt followed by known panel coarsening."""

    features: np.ndarray
    panel_mask: np.ndarray | None = None
    name: str = "environment"

    @property
    def nuisance_count(self) -> int:
        f = np.asarray(self.features)
        return 0 if f.size == 0 else (1 if f.ndim == 1 else f.shape[1])

    def probabilities_and_jacobians(
        self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, j_base = model.distribution_and_jacobian(theta)
        features = np.asarray(self.features, dtype=float)
        if features.size == 0:
            features = np.zeros((len(base), 0))
        elif features.ndim == 1:
            features = features[:, None]
        phi = np.asarray(phi, dtype=float)
        weights = np.exp(features @ phi) if features.shape[1] else np.ones(len(base))
        u = base * weights
        z = u.sum()
        selected = u / z
        # Quotient-rule derivative with respect to the latent generator.
        weighted_j = weights[:, None] * j_base
        j_theta_full = weighted_j / z - selected[:, None] * (weighted_j.sum(axis=0) / z)
        if features.shape[1]:
            centered = features - selected @ features
            j_phi_full = selected[:, None] * centered
        else:
            j_phi_full = np.zeros((len(base), 0))
        aggregate = aggregation_matrix(model.states, self.panel_mask)
        return aggregate @ selected, aggregate @ j_theta_full, aggregate @ j_phi_full

    def probabilities(self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return self.probabilities_and_jacobians(model, theta, phi)[0]


@dataclass(frozen=True)
class ShiftEnvironment:
    """Exact effective observation distortion along generator directions.

    This family is a controlled geometry benchmark: observed probabilities are
    those of theta + B phi, optionally coarsened by a panel.  It represents an
    observation mechanism whose local score is aligned with selected biological
    directions and must not be interpreted as a literal clinical sampling law.
    """

    directions: np.ndarray
    panel_mask: np.ndarray | None = None
    name: str = "shift_environment"

    @property
    def nuisance_count(self) -> int:
        b = np.asarray(self.directions)
        return 0 if b.size == 0 else (1 if b.ndim == 1 else b.shape[1])

    def probabilities_and_jacobians(
        self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = np.asarray(self.directions, dtype=float)
        if b.size == 0:
            b = np.zeros((model.parameter_count, 0))
        elif b.ndim == 1:
            b = b[:, None]
        effective = np.asarray(theta) + b @ np.asarray(phi)
        p, j = model.distribution_and_jacobian(effective)
        aggregate = aggregation_matrix(model.states, self.panel_mask)
        j_theta = aggregate @ j
        return aggregate @ p, j_theta, j_theta @ b

    def probabilities(self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return self.probabilities_and_jacobians(model, theta, phi)[0]


def genotype_features(states: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    """Return single/pair/higher-order binary features for specified terms."""
    x = np.asarray(states, dtype=float)
    return np.column_stack([np.prod(x[:, term], axis=1) for term in terms])

