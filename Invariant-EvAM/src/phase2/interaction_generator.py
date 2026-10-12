"""Small irreversible interacting CTMC with analytic snapshot derivatives."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.linalg import expm, expm_frechet


def binary_states(d: int) -> np.ndarray:
    return np.asarray(list(product([0, 1], repeat=d)), dtype=int)


@dataclass(frozen=True)
class SparseInteractingCTMC:
    """Irreversible CTMC with log-linear acquisition hazards.

    Parameter order is all baselines followed by the listed directed edges.
    Edge ``(target, source)`` contributes theta_edge * x_source to the log
    hazard for acquiring ``target``.
    """

    d: int
    edges: tuple[tuple[int, int], ...]
    horizon: float = 1.0

    def __post_init__(self) -> None:
        for target, source in self.edges:
            if target == source or not (0 <= target < self.d and 0 <= source < self.d):
                raise ValueError(f"invalid edge {(target, source)}")

    @property
    def parameter_count(self) -> int:
        return self.d + len(self.edges)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple([f"baseline_{i}" for i in range(self.d)] +
                     [f"interaction_{source}_to_{target}" for target, source in self.edges])

    @property
    def states(self) -> np.ndarray:
        return binary_states(self.d)

    def generator_and_derivatives(self, theta: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.parameter_count,):
            raise ValueError("theta has wrong shape")
        x = self.states
        n = len(x)
        q = np.zeros((n, n))
        dq = [np.zeros((n, n)) for _ in range(self.parameter_count)]
        edge_index = {(target, source): self.d + k for k, (target, source) in enumerate(self.edges)}
        state_index = {tuple(row): i for i, row in enumerate(x)}
        for s, state in enumerate(x):
            for target in range(self.d):
                if state[target]:
                    continue
                log_rate = theta[target]
                contributors: list[tuple[int, float]] = [(target, 1.0)]
                for source in range(self.d):
                    key = (target, source)
                    if key in edge_index and state[source]:
                        idx = edge_index[key]
                        log_rate += theta[idx]
                        contributors.append((idx, 1.0))
                rate = float(np.exp(log_rate))
                nxt = state.copy(); nxt[target] = 1
                t = state_index[tuple(nxt)]
                q[s, t] += rate; q[s, s] -= rate
                for idx, multiplier in contributors:
                    value = rate * multiplier
                    dq[idx][s, t] += value; dq[idx][s, s] -= value
        return q, dq

    def distribution(self, theta: np.ndarray) -> np.ndarray:
        q, _ = self.generator_and_derivatives(theta)
        initial = np.zeros(2 ** self.d); initial[0] = 1.0
        p = expm(q.T * self.horizon) @ initial
        return np.maximum(p, 0.0) / p.sum()

    def distribution_and_jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q, dq = self.generator_and_derivatives(theta)
        a = q.T * self.horizon
        initial = np.zeros(2 ** self.d); initial[0] = 1.0
        p = expm(a) @ initial
        columns = []
        for derivative in dq:
            frechet = expm_frechet(a, derivative.T * self.horizon, compute_expm=False)
            columns.append(frechet @ initial)
        return p, np.column_stack(columns)

    def finite_difference_jacobian(self, theta: np.ndarray, step: float = 1e-6) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        out = np.zeros((2 ** self.d, self.parameter_count))
        for j in range(self.parameter_count):
            plus = theta.copy(); plus[j] += step
            minus = theta.copy(); minus[j] -= step
            out[:, j] = (self.distribution(plus) - self.distribution(minus)) / (2 * step)
        return out

