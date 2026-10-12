"""Projected score geometry for shared evolutionary parameters."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def orthogonal_residual(j_theta: np.ndarray, j_nuisance: np.ndarray, probabilities: np.ndarray,
                        tol: float = 1e-10) -> np.ndarray:
    """Project Fisher-whitened evolutionary tangents off nuisance tangents."""
    scale = 1.0 / np.sqrt(np.maximum(np.asarray(probabilities), 1e-12))
    a = scale[:, None] * np.asarray(j_theta)
    n = scale[:, None] * np.asarray(j_nuisance)
    if n.shape[1] == 0:
        return a
    u, s, _ = np.linalg.svd(n, full_matrices=False)
    rank = int(np.sum(s > (s[0] * tol if len(s) else tol)))
    if rank == 0:
        return a
    basis = u[:, :rank]
    return a - basis @ (basis.T @ a)


@dataclass(frozen=True)
class IdentificationResult:
    projected_jacobian: np.ndarray
    singular_values: np.ndarray
    rank: int
    nullspace: np.ndarray
    condition_number: float
    min_singular_value: float


def analyze_environments(environment_outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                         tol: float = 1e-8) -> IdentificationResult:
    blocks = [orthogonal_residual(jt, jp, p) for p, jt, jp in environment_outputs]
    stacked = np.vstack(blocks)
    _, singular, vh = np.linalg.svd(stacked, full_matrices=True)
    threshold = (singular[0] * tol if len(singular) else tol)
    rank = int(np.sum(singular > threshold))
    null = vh[rank:].T.copy()
    minimum = float(singular[-1]) if len(singular) else 0.0
    condition = float(singular[0] / minimum) if minimum > threshold else float("inf")
    return IdentificationResult(stacked, singular, rank, null, condition, minimum)


def functional_status(vector: np.ndarray, result: IdentificationResult, tol: float = 1e-8) -> dict[str, float | bool]:
    a = np.asarray(vector, dtype=float)
    a = a / np.linalg.norm(a)
    leakage = float(np.linalg.norm(result.nullspace.T @ a)) if result.nullspace.size else 0.0
    information = float(np.linalg.norm(result.projected_jacobian @ a))
    return {"identified": leakage <= tol, "null_leakage": leakage, "projected_information": information}


def finite_difference_environment(environment, model, theta, phi, step=1e-6):
    p0 = environment.probabilities(model, theta, phi)
    jt = np.zeros((len(p0), len(theta)))
    jp = np.zeros((len(p0), len(phi)))
    for j in range(len(theta)):
        plus = np.asarray(theta).copy(); plus[j] += step
        minus = np.asarray(theta).copy(); minus[j] -= step
        jt[:, j] = (environment.probabilities(model, plus, phi) -
                    environment.probabilities(model, minus, phi)) / (2 * step)
    for j in range(len(phi)):
        plus = np.asarray(phi).copy(); plus[j] += step
        minus = np.asarray(phi).copy(); minus[j] -= step
        jp[:, j] = (environment.probabilities(model, theta, plus) -
                    environment.probabilities(model, theta, minus)) / (2 * step)
    return jt, jp

