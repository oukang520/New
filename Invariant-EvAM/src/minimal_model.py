"""Minimal exact model for Invariant-EvAM Phase 1.

Latent biology is an irreversible independent-acquisition CTMC observed at a
known horizon.  The snapshot state is a product Bernoulli distribution with
theta_i = logit(1-exp(-lambda_i T)).  Environment e exponentially tilts one
known genotype feature x_{b_e}; this shifts only its logit by gamma_e.

This is intentionally not a general MHN.  It is the smallest model that can
separate structural identification gain from sample-size gain.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit, logit
from scipy.stats import norm


def states(d: int) -> np.ndarray:
    return np.asarray(list(product([0, 1], repeat=d)), dtype=int)


def theta_to_rates(theta: np.ndarray, horizon: float = 1.0) -> np.ndarray:
    p = expit(np.asarray(theta, dtype=float))
    return -np.log1p(-p) / horizon


def rates_to_theta(rates: np.ndarray, horizon: float = 1.0) -> np.ndarray:
    p = -np.expm1(-np.asarray(rates, dtype=float) * horizon)
    return logit(p)


def product_distribution(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    x = states(len(theta))
    logits = x @ theta
    # Product Bernoulli normalizer is sum log(1+exp(theta_i)).
    logp = logits - np.logaddexp(0.0, theta).sum()
    return np.exp(logp)


def tilted_distribution(theta: np.ndarray, bias_coordinate: int, gamma: float,
                        delta: np.ndarray | None = None) -> np.ndarray:
    effective = np.asarray(theta, dtype=float).copy()
    if delta is not None:
        effective += np.asarray(delta, dtype=float)
    effective[bias_coordinate] += gamma
    return product_distribution(effective)


def weighted_distribution(base: np.ndarray, feature: np.ndarray, gamma: float) -> np.ndarray:
    q = np.asarray(base, dtype=float) * np.exp(gamma * np.asarray(feature, dtype=float))
    return q / q.sum()


def sample_counts(theta: np.ndarray, bias_coordinate: int, gamma: float, n: int,
                  rng: np.random.Generator, delta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    effective = np.asarray(theta, dtype=float).copy()
    if delta is not None:
        effective += np.asarray(delta, dtype=float)
    effective[bias_coordinate] += gamma
    probs = expit(effective)
    counts = rng.binomial(n, probs)
    return counts, np.full(len(theta), n, dtype=int)


@dataclass(frozen=True)
class Design:
    matrix: np.ndarray
    rows: list[tuple[int, int]]
    d: int
    environments: int
    bias_coordinates: tuple[int, ...]
    coverage: np.ndarray


def build_design(bias_coordinates: list[int] | tuple[int, ...], coverage: np.ndarray) -> Design:
    bias = tuple(int(v) for v in bias_coordinates)
    coverage = np.asarray(coverage, dtype=bool)
    e_count, d = coverage.shape
    if e_count != len(bias):
        raise ValueError("coverage and bias_coordinates disagree")
    rows: list[tuple[int, int]] = []
    matrix = []
    for e in range(e_count):
        for j in range(d):
            if not coverage[e, j]:
                continue
            row = np.zeros(d + e_count)
            row[j] = 1.0
            if j == bias[e]:
                row[d + e] = 1.0
            rows.append((e, j))
            matrix.append(row)
    return Design(np.asarray(matrix), rows, d, e_count, bias, coverage)


def nullspace(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    _, s, vh = np.linalg.svd(np.asarray(a, dtype=float), full_matrices=True)
    rank = int((s > (s[0] * rtol if len(s) else rtol)).sum())
    return vh[rank:].T.copy()


def theta_ambiguity_dimension(design: Design) -> int:
    ns = nullspace(design.matrix)
    if ns.size == 0:
        return 0
    theta_projection = ns[:design.d]
    return int(np.linalg.matrix_rank(theta_projection, tol=1e-9))


def identifiable_theta_dimension(design: Design) -> int:
    return design.d - theta_ambiguity_dimension(design)


def population_observed_logits(theta: np.ndarray, design: Design, gammas: np.ndarray,
                               deltas: np.ndarray | None = None) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if deltas is None:
        deltas = np.zeros((design.environments, design.d))
    return np.asarray([
        theta[j] + deltas[e, j] + (gammas[e] if j == design.bias_coordinates[e] else 0.0)
        for e, j in design.rows
    ])


def empirical_logits(counts: np.ndarray, totals: np.ndarray, design: Design) -> tuple[np.ndarray, np.ndarray]:
    values, variances = [], []
    for e, j in design.rows:
        k = float(counts[e, j])
        n = float(totals[e, j])
        # Jeffreys correction avoids infinite empirical logits.
        values.append(np.log((k + 0.5) / (n - k + 0.5)))
        variances.append(1.0 / (k + 0.5) + 1.0 / (n - k + 0.5))
    return np.asarray(values), np.asarray(variances)


def weighted_joint_estimate(y: np.ndarray, variances: np.ndarray, design: Design) -> tuple[np.ndarray, np.ndarray]:
    w = 1.0 / np.maximum(np.asarray(variances), 1e-12)
    aw = design.matrix * np.sqrt(w)[:, None]
    yw = np.asarray(y) * np.sqrt(w)
    beta = np.linalg.pinv(aw, rcond=1e-10) @ yw
    return beta[:design.d], beta[design.d:]


def direct_invariant_estimate(y: np.ndarray, variances: np.ndarray, design: Design) -> tuple[np.ndarray, np.ndarray]:
    """Estimate only theta coordinates observed outside their bias direction."""
    theta = np.full(design.d, np.nan)
    identifiable = np.zeros(design.d, dtype=bool)
    for j in range(design.d):
        ix = [r for r, (e, k) in enumerate(design.rows)
              if k == j and j != design.bias_coordinates[e]]
        if not ix:
            continue
        w = 1.0 / np.maximum(np.asarray(variances)[ix], 1e-12)
        theta[j] = np.average(np.asarray(y)[ix], weights=w)
        identifiable[j] = True
    return theta, identifiable


def naive_estimates(counts: np.ndarray, totals: np.ndarray, coverage: np.ndarray) -> dict[str, np.ndarray]:
    e_count, d = counts.shape
    per_env = np.full((e_count, d), np.nan)
    per_var = np.full((e_count, d), np.nan)
    for e in range(e_count):
        for j in range(d):
            if not coverage[e, j]:
                continue
            k, n = counts[e, j], totals[e, j]
            per_env[e, j] = np.log((k + 0.5) / (n - k + 0.5))
            per_var[e, j] = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)
    first = per_env[0].copy()
    pooled = np.full(d, np.nan)
    meta = np.full(d, np.nan)
    for j in range(d):
        observed = coverage[:, j]
        kk = counts[observed, j].sum()
        nn = totals[observed, j].sum()
        pooled[j] = np.log((kk + 0.5) / (nn - kk + 0.5))
        weights = 1.0 / per_var[observed, j]
        meta[j] = np.average(per_env[observed, j], weights=weights)
    return {"single_naive": first, "pooled_naive": pooled, "independent_meta": meta}


def bounded_profile_intervals(y: np.ndarray, variances: np.ndarray, design: Design,
                              gamma_bound: float, theta_bound: float,
                              alpha: float = 0.05, population: bool = False) -> tuple[np.ndarray, np.ndarray, bool]:
    m, p = design.matrix.shape
    if population:
        lower = np.asarray(y) - 1e-10
        upper = np.asarray(y) + 1e-10
    else:
        z = norm.ppf(1.0 - alpha / (2.0 * m))
        se = np.sqrt(np.asarray(variances))
        lower, upper = np.asarray(y) - z * se, np.asarray(y) + z * se
    a_ub = np.vstack([design.matrix, -design.matrix])
    b_ub = np.concatenate([upper, -lower])
    bounds = [(-theta_bound, theta_bound)] * design.d + [(-gamma_bound, gamma_bound)] * design.environments
    lows, highs = [], []
    for j in range(design.d):
        c = np.zeros(p); c[j] = 1.0
        lo = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        hi = linprog(-c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not (lo.success and hi.success):
            return np.full(design.d, np.nan), np.full(design.d, np.nan), False
        lows.append(lo.fun)
        highs.append(-hi.fun)
    return np.asarray(lows), np.asarray(highs), True


def state_kl(theta_true: np.ndarray, theta_est: np.ndarray) -> float:
    if np.isnan(theta_est).any():
        return np.nan
    p = product_distribution(theta_true)
    q = np.maximum(product_distribution(theta_est), 1e-300)
    return float(np.sum(p * np.log(p / q)))


def arbitrary_weight_alternative(observed: np.ndarray, alternative_theta: np.ndarray) -> np.ndarray:
    base = product_distribution(alternative_theta)
    weights = np.asarray(observed) / base
    # Multiplicative scale is irrelevant after normalization; set geometric mean to one.
    return weights / np.exp(np.mean(np.log(weights)))
