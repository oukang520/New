"""Joint likelihood and coordinate-profile intervals for small interaction models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


def multinomial_loglik(counts: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.sum(np.asarray(counts) * np.log(np.maximum(probabilities, 1e-300))))


@dataclass(frozen=True)
class FitResult:
    theta: np.ndarray
    phis: tuple[np.ndarray, ...]
    loglik: float
    success: bool


def _unpack(vector, p, nuisance_sizes):
    theta = np.asarray(vector[:p])
    phis = []
    start = p
    for q in nuisance_sizes:
        phis.append(np.asarray(vector[start:start + q])); start += q
    return theta, tuple(phis)


def fit_joint(model, environments, counts, theta_start=None, phi_starts=None,
              theta_bound=3.0, phi_bound=2.0, ridge=0.0) -> FitResult:
    p = model.parameter_count
    nuisance_sizes = [env.nuisance_count for env in environments]
    theta_start = np.zeros(p) if theta_start is None else np.asarray(theta_start)
    if phi_starts is None:
        phi_starts = [np.zeros(q) for q in nuisance_sizes]
    initial = np.concatenate([theta_start, *phi_starts])
    bounds = [(-theta_bound, theta_bound)] * p + [(-phi_bound, phi_bound)] * sum(nuisance_sizes)

    def objective(vector):
        theta, phis = _unpack(vector, p, nuisance_sizes)
        value = 0.0
        for env, observed, phi in zip(environments, counts, phis):
            value -= multinomial_loglik(observed, env.probabilities(model, theta, phi))
        value += 0.5 * ridge * float(vector @ vector)
        return value

    def gradient(vector):
        theta, phis = _unpack(vector, p, nuisance_sizes)
        g_theta = np.zeros(p); g_phi = []
        for env, observed, phi in zip(environments, counts, phis):
            probs, jt, jp = env.probabilities_and_jacobians(model, theta, phi)
            ratio = np.asarray(observed) / np.maximum(probs, 1e-300)
            g_theta -= jt.T @ ratio
            g_phi.append(-(jp.T @ ratio))
        return np.concatenate([g_theta, *g_phi]) + ridge * vector

    opt = minimize(objective, initial, jac=gradient, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-8})
    theta, phis = _unpack(opt.x, p, nuisance_sizes)
    return FitResult(theta, phis, -float(objective(opt.x)), bool(opt.success))


def profile_coordinate(model, environments, counts, coordinate: int, grid: np.ndarray,
                       global_fit: FitResult | None = None, theta_bound=3.0, phi_bound=2.0,
                       alpha=0.05) -> dict[str, object]:
    """Profile one raw coordinate; grid endpoints expose boundary-limited sets."""
    if global_fit is None:
        global_fit = fit_joint(model, environments, counts, theta_bound=theta_bound,
                               phi_bound=phi_bound)
    p = model.parameter_count
    nuisance_sizes = [env.nuisance_count for env in environments]
    other = [j for j in range(p) if j != coordinate]
    start = np.concatenate([global_fit.theta[other], *global_fit.phis])
    bounds = [(-theta_bound, theta_bound)] * len(other) + [(-phi_bound, phi_bound)] * sum(nuisance_sizes)
    values, success = [], []

    for fixed in np.asarray(grid, dtype=float):
        def objective(vector):
            theta = np.empty(p); theta[coordinate] = fixed; theta[other] = vector[:len(other)]
            _, phis = _unpack(np.r_[theta, vector[len(other):]], p, nuisance_sizes)
            return -sum(multinomial_loglik(obs, env.probabilities(model, theta, phi))
                        for env, obs, phi in zip(environments, counts, phis))

        def gradient(vector):
            theta = np.empty(p); theta[coordinate] = fixed; theta[other] = vector[:len(other)]
            _, phis = _unpack(np.r_[theta, vector[len(other):]], p, nuisance_sizes)
            g_theta = np.zeros(p); g_phi = []
            for env, obs, phi in zip(environments, counts, phis):
                probs, jt, jp = env.probabilities_and_jacobians(model, theta, phi)
                ratio = np.asarray(obs) / np.maximum(probs, 1e-300)
                g_theta -= jt.T @ ratio; g_phi.append(-(jp.T @ ratio))
            return np.concatenate([g_theta[other], *g_phi])

        opt = minimize(objective, start, jac=gradient, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 700, "ftol": 1e-10})
        values.append(-float(opt.fun)); success.append(bool(opt.success)); start = opt.x
    values = np.asarray(values)
    cutoff = global_fit.loglik - 0.5 * chi2.ppf(1 - alpha, df=1)
    accepted = np.asarray(grid)[values >= cutoff]
    interval = [float(accepted.min()), float(accepted.max())] if len(accepted) else [np.nan, np.nan]
    return {
        "grid": np.asarray(grid), "profile_loglik": values, "optimizer_success": np.asarray(success),
        "cutoff": float(cutoff), "interval": interval,
        "boundary_limited": bool(len(accepted) and (accepted.min() == np.min(grid) or accepted.max() == np.max(grid))),
        "global_fit": global_fit,
    }


def sample_environment_counts(model, environments, theta, phis, sample_sizes, rng):
    return [rng.multinomial(int(n), env.probabilities(model, theta, phi))
            for env, phi, n in zip(environments, phis, sample_sizes)]
