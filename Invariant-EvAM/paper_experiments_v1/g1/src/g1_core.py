from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import hashlib
import json

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares

from phase2.interaction_generator import SparseInteractingCTMC
from phase2.observation_operators import aggregation_matrix


@dataclass(frozen=True)
class Geometry:
    projected_jacobian: np.ndarray
    information_per_sample: np.ndarray
    singular_values: np.ndarray
    rank: int
    nullspace: np.ndarray
    nuisance_rank: int
    sigma_min_all: float
    sigma_min_nonzero: float
    rank_by_tolerance: Dict[str, int]


@dataclass(frozen=True)
class FrozenOperatorEnvironment:
    operator: Mapping[str, object]
    knowledge: str

    @property
    def nuisance_count(self) -> int:
        if self.knowledge != "K1_STRUCTURED":
            return 0
        feature = self.operator["feature"]
        if feature["kind"] == "none":
            return 0
        if feature["kind"] == "burden":
            return 1
        return len(feature["terms"])

    def probabilities_and_jacobians(
        self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base, j_base = model.distribution_and_jacobian(theta)
        features = feature_matrix(model.states, self.operator["feature"])
        if self.knowledge == "K1_STRUCTURED":
            active_phi = np.asarray(phi, dtype=float)
        else:
            active_phi = np.asarray(self.operator["phi_true"], dtype=float)
        weights = np.exp(features @ active_phi) if features.shape[1] else np.ones(len(base))
        u = base * weights
        z = float(u.sum())
        selected = u / z
        weighted_j = weights[:, None] * j_base
        jt_full = weighted_j / z - selected[:, None] * (weighted_j.sum(axis=0) / z)
        if self.knowledge == "K1_STRUCTURED" and features.shape[1]:
            centered = features - selected @ features
            jp_full = selected[:, None] * centered
        else:
            jp_full = np.zeros((len(base), 0))
        aggregate = aggregation_matrix(model.states, np.asarray(self.operator["panel_mask"], dtype=bool))
        return aggregate @ selected, aggregate @ jt_full, aggregate @ jp_full

    def probabilities(self, model: SparseInteractingCTMC, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return self.probabilities_and_jacobians(model, theta, phi)[0]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(master: int, group: int, purpose: int, *indices: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([master, group, purpose, *indices])


def all_directed_edges(d: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((target, source) for target in range(d) for source in range(d) if target != source)


def build_models(config: Mapping[str, object]) -> Dict[str, SparseInteractingCTMC]:
    models = {}
    for model_id, spec in config["models_core"].items():
        models[model_id] = SparseInteractingCTMC(
            int(spec["d"]), tuple(tuple(edge) for edge in spec["edges"]), horizon=1.0
        )
    return models


def parameter_names(model: SparseInteractingCTMC) -> List[str]:
    return [f"baseline_{i}" for i in range(model.d)] + [
        f"interaction_{source}_to_{target}" for target, source in model.edges
    ]


def primary_edge(model: SparseInteractingCTMC) -> Tuple[int, int]:
    preferred = (1, 0)
    if preferred in model.edges:
        return preferred
    return model.edges[0]


def primary_target_index(model: SparseInteractingCTMC) -> int:
    return model.d + list(model.edges).index(primary_edge(model))


def draw_parameters(
    model: SparseInteractingCTMC,
    master: int,
    model_index: int,
    purpose: int,
    count: int,
    baseline_range: Sequence[float],
    magnitudes: Sequence[float],
) -> np.ndarray:
    draws = []
    magnitudes = np.asarray(magnitudes, dtype=float)
    for draw_id in range(count):
        rng = np.random.default_rng(stable_seed(master, 1, purpose, model_index, draw_id))
        baseline = rng.uniform(float(baseline_range[0]), float(baseline_range[1]), model.d)
        mag = rng.choice(magnitudes, size=len(model.edges), replace=True)
        sign = rng.choice(np.array([-1.0, 1.0]), size=len(model.edges), replace=True)
        interaction = mag * sign
        # Deterministically retain mixed signs when all nonzero draws happen to agree.
        nz = np.flatnonzero(mag > 0)
        if len(nz) >= 2 and np.all(interaction[nz] >= 0):
            interaction[nz[0]] *= -1
        elif len(nz) >= 2 and np.all(interaction[nz] <= 0):
            interaction[nz[0]] *= -1
        draws.append(np.r_[baseline, interaction])
    return np.asarray(draws)


def _feature_spec(kind: str, d: int) -> Dict[str, object]:
    if kind.startswith("single_"):
        event = int(kind.split("_")[1])
        return {"kind": "terms", "terms": [[event]], "label": kind}
    if kind == "burden":
        return {"kind": "burden", "terms": [], "label": kind}
    if kind == "pair_vector":
        return {"kind": "terms", "terms": [[0], [d - 1]], "label": kind}
    if kind == "cooccur_01":
        return {"kind": "terms", "terms": [[0, 1]], "label": kind}
    if kind == "none":
        return {"kind": "none", "terms": [], "label": kind}
    raise KeyError(kind)


def feature_matrix(states: np.ndarray, feature: Mapping[str, object]) -> np.ndarray:
    kind = feature["kind"]
    if kind == "none":
        return np.zeros((len(states), 0), dtype=float)
    if kind == "burden":
        return states.mean(axis=1, keepdims=True).astype(float)
    terms = feature["terms"]
    return np.column_stack([np.prod(states[:, term], axis=1) for term in terms]).astype(float)


def _operator(
    operator_id: str,
    d: int,
    panel_mask: Sequence[int],
    feature_kind: str,
    phi: Sequence[float],
) -> Dict[str, object]:
    feature = _feature_spec(feature_kind, d)
    if len(phi) != (0 if feature["kind"] == "none" else (1 if feature["kind"] == "burden" else len(feature["terms"]))):
        raise ValueError((operator_id, feature, phi))
    return {
        "operator_id": operator_id,
        "d": d,
        "panel_mask": [int(x) for x in panel_mask],
        "panel_width": int(sum(panel_mask)),
        "feature": feature,
        "phi_true": [float(x) for x in phi],
        "selection_stage": "before_measurement",
        "measurement_missing_rule": "marginalize_not_zero_fill",
    }


def build_design_library(model_id: str, d: int) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    full = [1] * d
    drop_last = [1] * d; drop_last[d - 1] = 0
    drop_source = [1] * d; drop_source[0] = 0
    drop_target = [1] * d; drop_target[1] = 0
    pair01 = [1 if i in (0, 1) else 0 for i in range(d)]
    source_side = [1 if i != 1 else 0 for i in range(d)]
    target_side = [1 if i != 0 else 0 for i in range(d)]
    feature_defs = [
        ("single_0", [0.5]),
        ("single_1", [1.0]),
        ("burden", [0.5]),
        ("pair_vector", [0.5, -0.5]),
        ("cooccur_01", [1.0]),
        (f"single_{d - 1}", [1.5]),
    ]
    operators: Dict[str, Dict[str, object]] = {}

    def op(mask_name: str, mask: Sequence[int], feature_index: int) -> str:
        feature, phi = feature_defs[feature_index]
        operator_id = f"{model_id}__{mask_name}__F{feature_index}"
        if operator_id not in operators:
            operators[operator_id] = _operator(operator_id, d, mask, feature, phi)
        return operator_id

    designs: List[Dict[str, object]] = []
    # Six exact duplicate designs.
    for j in range(6):
        oid = op("full", full, j)
        designs.append({
            "design_id": f"{model_id}__D{len(designs):02d}", "model_id": model_id,
            "family": "REDUNDANT", "operator_ids": [oid, oid],
            "nuisance_groups": ["env0", "env1"], "predeclared_failure": False,
        })
    assay_pairs = [
        ("full", full, "drop_last", drop_last),
        ("full", full, "drop_source", drop_source),
        ("full", full, "drop_target", drop_target),
        ("drop_last", drop_last, "drop_source", drop_source),
        ("pair01", pair01, "drop_source", drop_source),
        ("source_side", source_side, "target_side", target_side),
    ]
    # Same selection parameter, only measurement changes; nuisance is shared.
    for j, (n0, m0, n1, m1) in enumerate(assay_pairs):
        designs.append({
            "design_id": f"{model_id}__D{len(designs):02d}", "model_id": model_id,
            "family": "ASSAY_ONLY", "operator_ids": [op(n0, m0, j), op(n1, m1, j)],
            "nuisance_groups": ["shared_selection", "shared_selection"],
            "predeclared_failure": j == 5,
        })
    # Full measurement, different predeclared selection families.
    for j in range(6):
        designs.append({
            "design_id": f"{model_id}__D{len(designs):02d}", "model_id": model_id,
            "family": "SELECTION_ONLY", "operator_ids": [op("full", full, j), op("full", full, (j + 1) % 6)],
            "nuisance_groups": ["env0", "env1"], "predeclared_failure": False,
        })
    joint_specs = [
        ("full", full, 0, "drop_last", drop_last, 1, False),
        ("drop_last", drop_last, 2, "drop_source", drop_source, 3, False),
        ("pair01", pair01, 4, "drop_target", drop_target, 5, False),
        ("source_side", source_side, 0, "target_side", target_side, 1, True),
        ("source_side", source_side, 2, "target_side", target_side, 4, True),
        ("drop_source", drop_source, 3, "drop_target", drop_target, 5, True),
    ]
    for n0, m0, f0, n1, m1, f1, fail in joint_specs:
        designs.append({
            "design_id": f"{model_id}__D{len(designs):02d}", "model_id": model_id,
            "family": "ASSAY_AND_SELECTION", "operator_ids": [op(n0, m0, f0), op(n1, m1, f1)],
            "nuisance_groups": ["env0", "env1"], "predeclared_failure": fail,
        })
    assert len(designs) == 24
    for design in designs:
        masks = [operators[x]["panel_mask"] for x in design["operator_ids"]]
        design["target_joint_observed"] = any(mask[0] and mask[1] for mask in masks)
        intersection = sum(int(a and b) for a, b in zip(*masks))
        union = sum(int(a or b) for a, b in zip(*masks))
        design["panel_jaccard"] = float(intersection / union) if union else 0.0
        design["measurement_cost_per_pair"] = int(sum(sum(mask) for mask in masks))
        design["eligibility"] = "ELIGIBLE"
    return operators, designs


def observable_output(
    model: SparseInteractingCTMC,
    base: np.ndarray,
    j_base: np.ndarray,
    operator: Mapping[str, object],
    knowledge: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features = feature_matrix(model.states, operator["feature"])
    phi = np.asarray(operator["phi_true"], dtype=float)
    weights = np.exp(features @ phi) if features.shape[1] else np.ones(len(base))
    selected_unnorm = base * weights
    z = float(selected_unnorm.sum())
    selected = selected_unnorm / z
    weighted_j = weights[:, None] * j_base
    jt_full = weighted_j / z - selected[:, None] * (weighted_j.sum(axis=0) / z)
    if knowledge == "K1_STRUCTURED" and features.shape[1]:
        centered = features - selected @ features
        jp_full = selected[:, None] * centered
    else:
        jp_full = np.zeros((len(base), 0))
    aggregate = aggregation_matrix(model.states, np.asarray(operator["panel_mask"], dtype=bool))
    # Rejection-sampling scale c=1/max(w); acceptance is a reporting diagnostic.
    acceptance = float((base * (weights / weights.max())).sum())
    return aggregate @ selected, aggregate @ jt_full, aggregate @ jp_full, acceptance


def joint_profiled_geometry(
    outputs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sample_weights: Sequence[float],
    nuisance_groups: Sequence[str],
    rtol: float = 1e-8,
    sensitivity_tolerances: Sequence[float] = (1e-6, 1e-7, 1e-9, 1e-10),
) -> Geometry:
    if not (len(outputs) == len(sample_weights) == len(nuisance_groups)):
        raise ValueError("environment metadata length mismatch")
    rho = np.asarray(sample_weights, dtype=float)
    rho = rho / rho.sum()
    a_blocks = []
    nuisance_dims: Dict[str, int] = {}
    n_blocks = []
    for (p, jt, jp), weight, group in zip(outputs, rho, nuisance_groups):
        scale = np.sqrt(weight) / np.sqrt(np.maximum(np.asarray(p), 1e-12))
        a_blocks.append(scale[:, None] * jt)
        n_blocks.append(scale[:, None] * jp)
        if jp.shape[1]:
            if group in nuisance_dims and nuisance_dims[group] != jp.shape[1]:
                raise ValueError("shared nuisance group has inconsistent dimension")
            nuisance_dims[group] = jp.shape[1]
    a = np.vstack(a_blocks)
    offsets = {}
    start = 0
    for group in sorted(nuisance_dims):
        offsets[group] = (start, start + nuisance_dims[group])
        start += nuisance_dims[group]
    n = np.zeros((a.shape[0], start))
    row_start = 0
    for block, group in zip(n_blocks, nuisance_groups):
        rows = slice(row_start, row_start + block.shape[0])
        if block.shape[1]:
            lo, hi = offsets[group]
            n[rows, lo:hi] = block
        row_start += block.shape[0]
    nuisance_rank = 0
    if n.shape[1]:
        u, s_n, _ = np.linalg.svd(n, full_matrices=False)
        threshold_n = (s_n[0] * 1e-10) if len(s_n) else 1e-10
        nuisance_rank = int(np.sum(s_n > threshold_n))
        if nuisance_rank:
            basis = u[:, :nuisance_rank]
            h = a - basis @ (basis.T @ a)
        else:
            h = a
    else:
        h = a
    _, singular, vh = np.linalg.svd(h, full_matrices=True)
    threshold = singular[0] * rtol if len(singular) else rtol
    rank = int(np.sum(singular > threshold))
    null = vh[rank:].T.copy()
    nonzero = singular[singular > threshold]
    by_tol = {}
    for tol in [rtol, *sensitivity_tolerances]:
        key = f"{tol:.0e}"
        by_tol[key] = int(np.sum(singular > (singular[0] * tol if len(singular) else tol)))
    return Geometry(
        projected_jacobian=h,
        information_per_sample=h.T @ h,
        singular_values=singular,
        rank=rank,
        nullspace=null,
        nuisance_rank=nuisance_rank,
        sigma_min_all=float(singular[-1]) if len(singular) else 0.0,
        sigma_min_nonzero=float(nonzero[-1]) if len(nonzero) else 0.0,
        rank_by_tolerance=by_tol,
    )


def target_metrics(vector: np.ndarray, geometry: Geometry, tol: float = 1e-8) -> Dict[str, object]:
    a = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(a))
    if norm == 0:
        raise ValueError("zero target")
    unit = a / norm
    leakage = float(np.linalg.norm(geometry.nullspace.T @ unit)) if geometry.nullspace.size else 0.0
    identified = leakage <= tol
    directional_norm = float(np.linalg.norm(geometry.projected_jacobian @ unit))
    if identified:
        inverse = np.linalg.pinv(geometry.information_per_sample, rcond=tol)
        variance_per_sample_inverse_n = float(unit @ inverse @ unit)
        target_information = (1.0 / variance_per_sample_inverse_n) if variance_per_sample_inverse_n > 0 else float("inf")
    else:
        variance_per_sample_inverse_n = float("inf")
        target_information = 0.0
    return {
        "identified": bool(identified),
        "null_leakage": leakage,
        "directional_score_norm": directional_norm,
        "target_information_per_sample": float(target_information),
        "asymptotic_variance_times_n": variance_per_sample_inverse_n,
    }


def expand_design(
    design: Mapping[str, object], environments: int
) -> Tuple[List[str], List[str]]:
    ids = list(design["operator_ids"])
    groups = list(design["nuisance_groups"])
    if environments == 1:
        return ids[:1], groups[:1]
    return [ids[i % len(ids)] for i in range(environments)], [f"{groups[i % len(groups)]}_{i // len(groups)}" if groups[i % len(groups)].startswith("env") else groups[i % len(groups)] for i in range(environments)]


def latent_geometry(model: SparseInteractingCTMC, theta: np.ndarray, rtol: float = 1e-8) -> Geometry:
    p, j = model.distribution_and_jacobian(theta)
    return joint_profiled_geometry([(p, j, np.zeros((len(p), 0)))], [1.0], ["none"], rtol=rtol)


def recover_source_from_known_selection(selected: np.ndarray, features: np.ndarray, phi: np.ndarray) -> np.ndarray:
    weights = np.exp(features @ phi) if features.shape[1] else np.ones(len(selected))
    recovered = np.asarray(selected) / weights
    return recovered / recovered.sum()


def trace_source_alias(
    model: SparseInteractingCTMC,
    theta: np.ndarray,
    target_index: int,
    target_delta: float = 0.4,
    bounds: Tuple[float, float] = (-3.0, 3.0),
) -> Dict[str, object]:
    target_probability = model.distribution(theta)
    other = [j for j in range(model.parameter_count) if j != target_index]
    fixed = float(theta[target_index] + target_delta)
    fixed = min(max(fixed, bounds[0] + 1e-8), bounds[1] - 1e-8)

    def unpack(x: np.ndarray) -> np.ndarray:
        candidate = np.empty(model.parameter_count)
        candidate[target_index] = fixed
        candidate[other] = x
        return candidate

    result = least_squares(
        lambda x: model.distribution(unpack(x))[:-1] - target_probability[:-1],
        theta[other], bounds=(bounds[0], bounds[1]),
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=4000,
    )
    alias = unpack(result.x)
    probability = model.distribution(alias)
    kl = float(np.sum(target_probability * np.log(np.maximum(target_probability, 1e-300) / np.maximum(probability, 1e-300))))
    return {
        "theta_alias": alias,
        "success": bool(result.success),
        "max_abs_probability_error": float(np.max(np.abs(probability - target_probability))),
        "state_kl": kl,
        "parameter_distance": float(np.linalg.norm(alias - theta)),
        "target_change": float(alias[target_index] - theta[target_index]),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }
