"""Frozen, patient-disjoint TRACERx acquisition replay.

The CTMC is a population snapshot surrogate for recorded four-gene reference
states.  It does not model spatial clone growth, ancestry, or recurrence risk.
A separately fitted regional measurement kernel and ordinary ridge action-
utility regression form the explicitly new patient-level prediction bridge.

Policy entry points receive VisibleCase, public costs/capacities, and a frozen
development model only.  Branch labels, hidden variants and future labels are
never passed to a policy.  Confirmation actions are written and hashed before
the branch/driver scorer tables are opened.  No plotting/publication occurs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
GENES = ("TP53", "KRAS", "STK11", "KEAP1")
STATES = np.array([[s >> (3 - g) & 1 for g in range(4)] for s in range(16)])
NONSYNONYMOUS = frozenset(("nonsynonymous SNV", "stopgain", "stoploss", "stopgain SNV", "stoploss SNV",
    "frameshift deletion", "frameshift insertion", "nonframeshift deletion",
    "nonframeshift insertion", "nonframeshift substitution", "frameshift substitution"))
METHODS = ("BEST_FIXED", "RANDOM", "CTMC_BRIDGE", "EMPIRICAL_BRIDGE", "DIRECT_RIDGE")
RIDGE_ALPHA = 10.0
PRIOR_PSEUDOCOUNT = 0.5
KERNEL_PSEUDOCOUNT = 0.5  # Total row mass, distributed over allowed sub-states.


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seeded_integer(*values: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()[:16], 16)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def state_number(bits: np.ndarray) -> int:
    return int(np.asarray(bits, dtype=int) @ np.array([8, 4, 2, 1]))


@dataclass(frozen=True)
class VisibleCase:
    """The complete selection-time interface, with no scorer fields."""
    patient_id: str
    tumour_id: str
    histology: str
    repeat: int
    initial_region: str
    region_order: tuple[str, ...]
    core_state: int
    positive_core_counts: tuple[int, ...]
    mean_vaf: tuple[float, ...]
    max_vaf: tuple[float, ...]


@dataclass
class RegionObservation:
    positive: frozenset[str]
    core_positive: frozenset[str]
    core_state: int
    core_counts: tuple[int, ...]
    mean_vaf: tuple[float, ...]
    max_vaf: tuple[float, ...]


@dataclass
class PatientObservation:
    patient_id: str
    tumour_id: str
    histology: str
    regions: dict[str, RegionObservation]

    def visible(self, repeat: int, seed: int) -> VisibleCase:
        order = sorted(self.regions, key=lambda r: (seeded_integer(seed,
            "REGION_ORDER", self.patient_id, r), r))
        # Five prescribed perturbations rotate one frozen random ordering.
        # Small region counts legitimately cause repeated starting regions.
        shift = repeat % len(order)
        order = order[shift:] + order[:shift]
        first = self.regions[order[0]]
        return VisibleCase(self.patient_id, self.tumour_id, self.histology,
            repeat, order[0], tuple(order), first.core_state, first.core_counts,
            first.mean_vaf, first.max_vaf)

    def reference_core_state(self) -> int:
        return int(np.bitwise_or.reduce([r.core_state for r in self.regions.values()]))


@dataclass(frozen=True)
class Plan:
    expand: int
    new_full: int
    new_narrow: int
    cost: int

    @property
    def name(self) -> str:
        return f"E{self.expand}_F{self.new_full}_N{self.new_narrow}"

    def to_dict(self) -> dict:
        return dict(expand=self.expand, new_full=self.new_full,
            new_narrow=self.new_narrow, cost=self.cost, plan=self.name)


def plan_templates(budget: int, overhead: int, narrow: int = 2,
                   full: int = 8) -> list[Plan]:
    out = []
    for expand in (0, 1):
        for new_full in range(budget // (full + overhead) + 1):
            for new_narrow in range(budget // (narrow + overhead) + 1):
                cost = (full - narrow) * expand + (full + overhead) * new_full + (narrow + overhead) * new_narrow
                if cost <= budget:
                    out.append(Plan(expand, new_full, new_narrow, cost))
    return sorted(out, key=lambda p: p.name)


def instantiate(template: Plan, capacity: int, overhead: int,
                narrow: int = 2, full: int = 8) -> Plan:
    """Public capacity clipping, WES regions first, then narrow regions."""
    nf = min(template.new_full, capacity)
    nn = min(template.new_narrow, capacity - nf)
    cost = (full - narrow) * template.expand + (full + overhead) * nf + (narrow + overhead) * nn
    return Plan(template.expand, nf, nn, cost)


def feasible_plans(case: VisibleCase, budget: int, overhead: int,
                   narrow: int = 2, full: int = 8) -> list[Plan]:
    unique = {q.name: q for p in plan_templates(budget, overhead, narrow, full)
        for q in [instantiate(p, len(case.region_order) - 1, overhead, narrow, full)]}
    return sorted(unique.values(), key=lambda p: p.name)


def dominates(a: Plan, b: Plan) -> bool:
    """Strict observation inclusion under the same frozen region sequence."""
    if a.expand < b.expand or a.new_full < b.new_full:
        return False
    if a.new_full + a.new_narrow < b.new_full + b.new_narrow:
        return False
    return (a.expand, a.new_full, a.new_narrow) != (b.expand, b.new_full, b.new_narrow)


def nondominated(plans: list[Plan]) -> list[Plan]:
    return [p for p in plans if not any(dominates(q, p) for q in plans)]


def load_observations(audit_dir: Path) -> dict[str, PatientObservation]:
    """Load replay measurements.  No branch/scorer tables are opened here."""
    roster = pd.read_csv(audit_dir / "patient_primary_tumour.tsv", sep="\t", dtype=str)
    roster = roster[roster.primary_regions.astype(int).ge(3)]
    keep = dict(zip(roster.tumour_id, roster.patient_id))
    regions = pd.read_csv(audit_dir / "regions.tsv", sep="\t", dtype=str)
    regions = regions[regions.tumour_id.isin(keep) & regions.region_class.eq("SURGICAL_PRIMARY")]
    grouped = defaultdict(dict)
    for row in regions.itertuples(index=False):
        grouped[row.patient_id][row.RegionID] = dict(positive=set(), core=set(),
            bits=np.zeros(4, dtype=int), counts=np.zeros(4, dtype=int), vaf=[[] for _ in GENES])
    # CCF, cluster IDs, driver annotation and negative-locus identities are not
    # used in the policy features. VAF uses actual counts in exposed positives.
    cols = ["patient_id", "tumour_id", "RegionID", "mutation_id", "gene",
            "var_count", "depth", "positive", "exonic.func", "func"]
    for chunk in pd.read_csv(audit_dir / "region_variants.tsv.gz", sep="\t",
                             usecols=cols, chunksize=200000, low_memory=False):
        chunk = chunk[chunk.tumour_id.isin(keep) & as_bool(chunk.positive)]
        chunk = chunk[chunk.RegionID.isin(set(regions.RegionID))][cols]
        for row in chunk.itertuples(index=False, name=None):
            item = dict(zip(cols, row))
            r = grouped[item["patient_id"]][item["RegionID"]]
            mutation = str(item["mutation_id"])
            r["positive"].add(mutation)
            if item["gene"] in GENES:
                r["core"].add(mutation)
                g = GENES.index(item["gene"])
                r["counts"][g] += 1
                r["vaf"][g].append(float(item["var_count"]) / max(float(item["depth"]), 1.))
                label = str(item["exonic.func"]).lower()
                if any(t in label for t in ("nonsynonymous", "stopgain", "stoploss", "frameshift", "startloss")) or item["func"] == "splicing":
                    r["bits"][g] = 1
    histology = dict(zip(roster.patient_id, roster.histology_3))
    tumour = dict(zip(roster.patient_id, roster.tumour_id))
    out = {}
    for pid, values in grouped.items():
        rr = {rid: RegionObservation(frozenset(z["positive"]), frozenset(z["core"]),
            state_number(z["bits"]), tuple(map(int, z["counts"])),
            tuple(float(np.mean(v)) if v else 0. for v in z["vaf"]),
            tuple(float(max(v)) if v else 0. for v in z["vaf"])) for rid, z in values.items()}
        out[pid] = PatientObservation(pid, tumour[pid], histology[pid], rr)
    if set(out) != set(roster.patient_id):
        raise RuntimeError("Eligible patient/region roster mismatch")
    return out


@dataclass
class Scorer:
    branches: dict[str, dict[str, frozenset[str]]]
    driver_variants: frozenset[str]
    minimum_markers: int
    score_cache: dict = field(default_factory=dict)


def load_scorer(audit_dir: Path, protocol: dict) -> Scorer:
    """Private evaluation data, loaded only after confirmation action freeze."""
    b = pd.read_csv(audit_dir / "branch_reference.tsv", sep="\t", dtype=str)
    b = b[as_bool(b.eligible_multimarker_reference)]
    support = pd.read_csv(audit_dir / "cluster_region_support.tsv.gz", sep="\t", dtype=str)
    support = support[support.region_class.eq("SURGICAL_PRIMARY") & as_bool(support.detected)]
    supported = set(zip(support.tumour_id, support.cluster))
    wanted = set((t, c) for t, c in zip(b.tumour_id, b.cluster) if (t, c) in supported)
    markers = pd.read_csv(audit_dir / "branch_owned_markers.tsv.gz", sep="\t", dtype=str)
    groups = defaultdict(lambda: defaultdict(set))
    for row in markers.itertuples(index=False):
        if (row.tumour_id, row.cluster) in wanted:
            groups[row.tumour_id][row.cluster].add(row.mutation_id)
    native = Path(protocol["native_source"])
    mutation = pd.read_csv(native / "mutations.tsv.gz", sep="\t",
        usecols=["mutation_id", "DriverMut"], dtype=str)
    drivers = frozenset(mutation.loc[as_bool(mutation.DriverMut), "mutation_id"])
    return Scorer({t: {c: frozenset(v) for c, v in cs.items()} for t, cs in groups.items()},
        drivers, int(protocol["reference_rules"]["minimum_branch_owned_markers"]))


def exposed_regions(case: VisibleCase, plan: Plan) -> dict[str, str]:
    modes = {case.initial_region: "FULL" if plan.expand else "NARROW"}
    for j, rid in enumerate(case.region_order[1:1 + plan.new_full + plan.new_narrow]):
        modes[rid] = "FULL" if j < plan.new_full else "NARROW"
    return modes


def expose(patient: PatientObservation, case: VisibleCase, plan: Plan) -> dict[str, frozenset[str]]:
    return {rid: (patient.regions[rid].positive if mode == "FULL" else patient.regions[rid].core_positive)
        for rid, mode in exposed_regions(case, plan).items()}


def detected_branches(visible: dict[str, frozenset[str]],
                      marker_sets: dict[str, frozenset[str]], threshold: int) -> set[str]:
    # Require co-detection of at least three branch-owned variants in one
    # actually exposed region. Inherited trunk markers never enter marker_sets.
    return {c for c, markers in marker_sets.items()
            if any(len(v & markers) >= threshold for v in visible.values())}


def score_plan(patient: PatientObservation, case: VisibleCase, plan: Plan,
               scorer: Scorer) -> dict:
    cache_key = (patient.patient_id, case.repeat, plan.name)
    if cache_key in scorer.score_cache:
        return scorer.score_cache[cache_key]
    marker_sets = scorer.branches.get(patient.tumour_id, {})
    initial = {case.initial_region: patient.regions[case.initial_region].core_positive}
    visible = expose(patient, case, plan)
    init_branch = detected_branches(initial, marker_sets, scorer.minimum_markers)
    after_branch = detected_branches(visible, marker_sets, scorer.minimum_markers)
    all_positive = frozenset().union(*(r.positive for r in patient.regions.values()))
    init_positive = patient.regions[case.initial_region].core_positive
    after_positive = frozenset().union(*visible.values())
    drivers = scorer.driver_variants & all_positive
    init_drivers = drivers & init_positive
    after_drivers = drivers & after_positive
    branch_n = len(marker_sets)
    acquired_branch = len(after_branch - init_branch)
    hidden_branch_n = branch_n - len(init_branch)
    hidden_driver_n = len(drivers - init_drivers)
    acquired_drivers = len(after_drivers - init_drivers)
    observed_core = int(np.bitwise_or.reduce([patient.regions[r].core_state for r in visible]))
    result = dict(initial_branch=len(init_branch), recovered_branch=acquired_branch,
        all_branch=branch_n, hidden_branch=hidden_branch_n, after_branch=len(after_branch),
        branch_recall=acquired_branch / branch_n if branch_n else 0.,
        hidden_branch_recall=acquired_branch / hidden_branch_n if hidden_branch_n else 0.,
        all_branch_recall=len(after_branch) / branch_n if branch_n else 0.,
        denominator_free_branch=branch_n == 0,
        initialdrivers=len(init_drivers), hidden_driver=acquired_drivers,
        hidden_driver_denominator=hidden_driver_n, totaldrivers=len(drivers),
        afterdrivers=len(after_drivers),
        hidden_driver_recall=acquired_drivers / hidden_driver_n if hidden_driver_n else 0.,
        all_driver_recall=len(after_drivers) / len(drivers) if drivers else 0.,
        denominator_free_driver=hidden_driver_n == 0,
        reference_core_state=patient.reference_core_state(),
        after_core_state=observed_core, observed_regions=len(visible),
        recovered_branch_ids=";".join(sorted(after_branch - init_branch)),
        retrieved_driver_ids=";".join(sorted(after_drivers - init_drivers)))
    scorer.score_cache[cache_key] = result
    return result


def fit_snapshot(reference_counts: np.ndarray) -> dict:
    """Use the project's unchanged four-event snapshot implementation."""
    import sys
    model_dir = PROJECT / "paper_experiments_v1" / "g5" / "src"
    sys.path.insert(0, str(model_dir))
    from g5_model import Model, fit  # noqa: deferred to development fit only
    counts = np.asarray(reference_counts, dtype=float) + PRIOR_PSEUDOCOUNT
    spec = dict(topology="SPARSE", features=[np.zeros((16, 0)).tolist()],
                known_phi=[[]], bound=3.)
    likelihood = Model([dict(env=0, A=np.eye(16).tolist(), counts=counts.tolist())], spec)
    result = fit(likelihood)
    if not np.isfinite(result["nll"]):
        raise RuntimeError("Non-finite CTMC surrogate fit; no silent substitution")
    p, _ = likelihood.source(np.asarray(result["x"]))
    p = np.maximum(np.asarray(p, dtype=float), 1e-14)
    p /= p.sum()
    return dict(probabilities=p.tolist(), fit=result, spec=spec,
        source_sha256=digest(model_dir / "g5_model.py"),
        semantics="REGULARIZED_DEVELOPMENT_SNAPSHOT_REFERENCE_SURROGATE_NOT_NATURAL_HAZARD_TRUTH")


def fit_prior_and_kernel(patients: list[PatientObservation]) -> dict:
    counts = np.bincount([p.reference_core_state() for p in patients], minlength=16).astype(float)
    empirical = (counts + PRIOR_PSEUDOCOUNT) / (counts.sum() + 16 * PRIOR_PSEUDOCOUNT)
    kernels = {}
    max_n = max(len(p.regions) for p in patients)
    # Each patient contributes weight one at each subset size. Every within-
    # patient region combination is a perturbation, not an independent case.
    for m in range(1, max_n):
        k = np.zeros((16, 16))
        for p in patients:
            values = [r.core_state for r in p.regions.values()]
            if m >= len(values):
                continue
            subsets = list(itertools.combinations(values, m))
            for subset in subsets:
                z = int(np.bitwise_or.reduce(subset))
                k[p.reference_core_state(), z] += 1 / len(subsets)
        for s in range(16):
            allowed = np.array([z & s == z for z in range(16)])
            k[s, allowed] += KERNEL_PSEUDOCOUNT / allowed.sum()
            k[s, ~allowed] = 0.
            k[s] /= k[s].sum()
        kernels[str(m)] = k.tolist()
    snapshot = fit_snapshot(counts)
    return dict(reference_counts=counts.tolist(), empirical_prior=empirical.tolist(),
        ctmc=snapshot, regional_kernels=kernels,
        patient_ids=[p.patient_id for p in patients],
        prior_pseudocount=PRIOR_PSEUDOCOUNT, kernel_total_row_pseudocount=KERNEL_PSEUDOCOUNT,
        state_semantics="FOUR_GENE_REPORTED_NONSYNONYMOUS_PRIMARY_REGION_UNION_PROXY",
        negative_semantics="NOT_RECORDED_IS_NOT_VERIFIED_GENE_CALLABLE_ABSENCE")


def posterior(prior_model: dict, kind: str, observed_state: int,
              measured_regions: int, available_regions: int) -> np.ndarray:
    if measured_regions >= available_regions:
        return np.eye(16)[observed_state]
    prior = np.asarray(prior_model["ctmc"]["probabilities"] if kind == "CTMC_BRIDGE"
                       else prior_model["empirical_prior"], dtype=float)
    key = str(measured_regions)
    if key not in prior_model["regional_kernels"]:
        key = str(max(map(int, prior_model["regional_kernels"])))
    kernel = np.asarray(prior_model["regional_kernels"][key])
    q = prior * kernel[:, observed_state]
    if q.sum() <= 0 or not np.isfinite(q).all():
        raise RuntimeError("Invalid frozen posterior")
    return q / q.sum()


def visible_features(case: VisibleCase, prior_model: dict, method: str) -> np.ndarray:
    bits = STATES[case.core_state].astype(float)
    # Saturated observed-state contrasts give DIRECT_RIDGE a transparent strong
    # categorical baseline, rather than forcing a weak linear four-bit model.
    x = np.r_[bits, np.eye(16)[case.core_state], np.log1p(case.positive_core_counts),
        case.mean_vaf, case.max_vaf, math.log1p(len(case.region_order))]
    if method != "DIRECT_RIDGE":
        q = posterior(prior_model, method, case.core_state, 1, len(case.region_order))
        missing = q @ STATES - bits
        entropy = -float(q @ np.log(np.maximum(q, 1e-14)))
        x = np.r_[x, q, missing, entropy, 1. - float(q[case.core_state])]
    return x


def action_features(plan: Plan, budget: int, region_count: int) -> np.ndarray:
    return np.array([plan.expand, plan.new_full, plan.new_narrow,
        plan.expand + plan.new_full, plan.new_full + plan.new_narrow,
        plan.cost / max(budget, 1), (budget - plan.cost) / max(budget, 1),
        (1 + plan.new_full + plan.new_narrow) / region_count], dtype=float)


def utility_features(case: VisibleCase, plan: Plan, prior_model: dict,
                     method: str, budget: int) -> np.ndarray:
    v = visible_features(case, prior_model, method)
    a = action_features(plan, budget, len(case.region_order))
    return np.r_[v, a, np.outer(v, a[:5]).ravel()]


def ridge_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=float)
    normalized = w / w.sum()
    mean = normalized @ x
    scale = np.sqrt(normalized @ ((x - mean) ** 2))
    scale[scale < 1e-12] = 1.
    z = (x - mean) / scale
    ym = float(normalized @ y)
    # The sum of weights is normalized to the number of independent patients,
    # not to region rotations or to the number of feasible action templates.
    gram = z.T @ (w[:, None] * z)
    rhs = z.T @ (w * (y - ym))
    coef = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(x.shape[1]), rhs)
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficients=coef.tolist(),
        intercept=ym, alpha=RIDGE_ALPHA, objective="PATIENT_BALANCED_SUM_SQUARED_ERROR_PLUS_10_L2",
        feature_count=int(x.shape[1]), row_count=int(len(y)))


def ridge_predict(model: dict, x: np.ndarray) -> float:
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return float(model["intercept"] + z @ np.asarray(model["coefficients"]))


def choose_frozen(case: VisibleCase, budget: int, overhead: int, method: str,
                  model: dict, seed: int) -> tuple[Plan, float]:
    """No PatientObservation, Scorer, hidden state, or future outcome input."""
    cell = model["cells"][f"B{budget}_O{overhead}"]
    narrow, full = model["costs"]["narrow"], model["costs"]["full"]
    plans = feasible_plans(case, budget, overhead, narrow, full)
    frontier = nondominated(plans)
    if method == "BEST_FIXED":
        t = cell["best_fixed_template"]
        p = instantiate(Plan(t["expand"], t["new_full"], t["new_narrow"], t["cost"]),
            len(case.region_order) - 1, overhead, narrow, full)
        return p, float(cell["best_fixed_development_mean"])
    if method == "RANDOM":
        index = seeded_integer(seed, "RANDOM_PLAN", case.patient_id, case.repeat,
            budget, overhead) % len(frontier)
        return frontier[index], float("nan")
    regression = cell["regression"][method]
    scores = [(ridge_predict(regression, utility_features(case, p, model["prior"],
        method, budget)), p) for p in frontier]
    value, p = min(scores, key=lambda z: (-z[0], z[1].cost, z[1].name))
    return p, value


def cell_settings(protocol: dict) -> list[tuple[int, int]]:
    acq = protocol["acquisition"]
    return [(int(b), int(o)) for b in acq["incremental_budgets"] for o in acq["new_region_overheads"]]


def score_predictions(prior: dict, method: str, scored: dict, n_regions: int) -> dict:
    kind = method if method == "CTMC_BRIDGE" else "EMPIRICAL_BRIDGE"
    q = posterior(prior, kind, scored["after_core_state"], scored["observed_regions"], n_regions)
    truth = scored["reference_core_state"]
    marginal = q @ STATES
    return dict(four_gene_log_loss=-math.log(max(float(q[truth]), 1e-14)),
        four_gene_brier=float(np.sum((q - np.eye(16)[truth]) ** 2)),
        four_gene_marginal_brier=float(np.mean((marginal - STATES[truth]) ** 2)),
        four_gene_probability_true=float(q[truth]))


def base_row(case: VisibleCase, population: str, stage: str, budget: int,
             overhead: int, method: str, plan: Plan) -> dict:
    modes = exposed_regions(case, plan)
    return dict(population=population, patient_id=case.patient_id, tumour_id=case.tumour_id,
        split=stage, histology=case.histology, repeat=case.repeat, budget=budget,
        overhead=overhead, method=method, plan=plan.name, cost=plan.cost,
        initial_cost=2, total_cost=2 + plan.cost, unused_budget=budget - plan.cost,
        initial_region=case.initial_region, region_order=";".join(case.region_order),
        initial_core_state=case.core_state,
        full_regions=";".join(r for r, mode in modes.items() if mode == "FULL"),
        narrow_regions=";".join(r for r, mode in modes.items() if mode == "NARROW"))


def summarize_patients(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["population", "patient_id", "tumour_id", "split", "histology", "budget", "overhead", "method"]
    metrics = ["cost", "unused_budget", "initial_branch", "recovered_branch", "all_branch",
        "hidden_branch", "branch_recall", "hidden_branch_recall", "all_branch_recall",
        "initialdrivers", "hidden_driver", "hidden_driver_denominator", "totaldrivers",
        "hidden_driver_recall", "all_driver_recall", "four_gene_log_loss", "four_gene_brier",
        "four_gene_marginal_brier"]
    out = frame.groupby(keys, as_index=False)[metrics].mean()
    repeats = frame.groupby(keys).size().rename("perturbation_rows").reset_index()
    return out.merge(repeats, on=keys, validate="one_to_one")


def fit_population(patients: list[PatientObservation], population: str,
                   protocol: dict, scorer: Scorer) -> tuple[dict, list[dict]]:
    if not patients:
        raise RuntimeError(f"No development patients for {population}")
    prior = fit_prior_and_kernel(patients)
    acq = protocol["acquisition"]
    narrow, full = int(acq["narrow_cost"]), int(acq["full_cost"])
    repeats, seed = int(acq["initial_region_repeats"]), int(protocol["master_seed"])
    model = dict(population=population, prior=prior, costs=dict(narrow=narrow, full=full), cells={})
    fixed_rows = []
    for budget, overhead in cell_settings(protocol):
        templates = plan_templates(budget, overhead, narrow, full)
        examples = []
        template_y = defaultdict(list)
        for patient in patients:
            for repeat in range(repeats):
                case = patient.visible(repeat, seed)
                unique = feasible_plans(case, budget, overhead, narrow, full)
                scores = {p.name: score_plan(patient, case, p, scorer) for p in unique}
                for template in templates:
                    p = instantiate(template, len(case.region_order) - 1, overhead, narrow, full)
                    s = scores[p.name]
                    template_y[template.name].append(s["branch_recall"])
                    row = base_row(case, population, "development", budget, overhead,
                        f"FIXED_{template.name}", p)
                    row.update(s)
                    row["template"] = template.name
                    row.update(score_predictions(prior, "EMPIRICAL_BRIDGE", s, len(case.region_order)))
                    fixed_rows.append(row)
                # Equal weight patients/rotations, then equal weight each
                # feasible action within a rotation; capacity cannot add N.
                for p in unique:
                    examples.append((case, p, scores[p.name]["branch_recall"],
                        1. / (repeats * len(unique))))
        best = min(templates, key=lambda p: (-float(np.mean(template_y[p.name])), p.cost, p.name))
        regressions = {}
        for method in ("CTMC_BRIDGE", "EMPIRICAL_BRIDGE", "DIRECT_RIDGE"):
            x = np.vstack([utility_features(c, p, prior, method, budget) for c, p, y, w in examples])
            y = np.array([y for c, p, y, w in examples])
            weights = np.array([w for c, p, y, w in examples])
            regressions[method] = ridge_fit(x, y, weights)
            regressions[method]["independent_training_patients"] = len(patients)
        model["cells"][f"B{budget}_O{overhead}"] = dict(best_fixed_template=best.to_dict(),
            best_fixed_development_mean=float(np.mean(template_y[best.name])),
            all_fixed_development_mean={k: float(np.mean(v)) for k, v in template_y.items()},
            regression=regressions)
    return model, fixed_rows


def population_patients(observations: dict[str, PatientObservation], split: dict[str, str],
                        population: str, stage: str) -> list[PatientObservation]:
    return sorted((p for pid, p in observations.items() if split.get(pid) == stage
        and (population == "ALL_NSCLC" or p.histology == "LUAD")), key=lambda p: p.patient_id)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def freeze_actions(observations: dict[str, PatientObservation], split: dict[str, str],
                   models: dict, protocol: dict, stage: str) -> list[dict]:
    rows = []
    repeats = int(protocol["acquisition"]["initial_region_repeats"])
    seed = int(protocol["master_seed"])
    for population, model in models.items():
        patients = population_patients(observations, split, population, stage)
        for patient in patients:
            for repeat in range(repeats):
                case = patient.visible(repeat, seed)
                for budget, overhead in cell_settings(protocol):
                    for method in METHODS:
                        plan, estimate = choose_frozen(case, budget, overhead, method, model, seed)
                        row = base_row(case, population, stage, budget, overhead, method, plan)
                        row.update(plan.to_dict())
                        row["predicted_utility"] = estimate if math.isfinite(estimate) else None
                        rows.append(row)
    return rows


def evaluate_actions(actions: pd.DataFrame, observations: dict[str, PatientObservation],
                     models: dict, protocol: dict, scorer: Scorer) -> pd.DataFrame:
    rows = []
    seed = int(protocol["master_seed"])
    for raw in actions.to_dict(orient="records"):
        patient = observations[raw["patient_id"]]
        case = patient.visible(int(raw["repeat"]), seed)
        plan = Plan(int(raw["expand"]), int(raw["new_full"]), int(raw["new_narrow"]), int(raw["cost"]))
        scored = score_plan(patient, case, plan, scorer)
        raw.update(scored)
        raw.update(score_predictions(models[raw["population"]]["prior"], raw["method"], scored, len(case.region_order)))
        rows.append(raw)
    return pd.DataFrame(rows)


def evaluate_fixed_confirmation(observations: dict[str, PatientObservation], split: dict[str, str],
                                models: dict, protocol: dict, scorer: Scorer) -> pd.DataFrame:
    rows = []
    seed = int(protocol["master_seed"])
    repeats = int(protocol["acquisition"]["initial_region_repeats"])
    for population, model in models.items():
        patients = population_patients(observations, split, population, "confirmation")
        for patient in patients:
            for repeat in range(repeats):
                case = patient.visible(repeat, seed)
                for budget, overhead in cell_settings(protocol):
                    for template in plan_templates(budget, overhead, model["costs"]["narrow"], model["costs"]["full"]):
                        plan = instantiate(template, len(case.region_order) - 1, overhead,
                            model["costs"]["narrow"], model["costs"]["full"])
                        row = base_row(case, population, "confirmation", budget, overhead,
                            f"FIXED_{template.name}", plan)
                        row["template"] = template.name
                        scored = score_plan(patient, case, plan, scorer)
                        row.update(scored)
                        row.update(score_predictions(model["prior"], "EMPIRICAL_BRIDGE", scored, len(case.region_order)))
                        rows.append(row)
    return pd.DataFrame(rows)


def source_manifest(audit_dir: Path, protocol_path: Path, split_path: Path) -> dict:
    files = [audit_dir / x for x in ("patient_primary_tumour.tsv", "regions.tsv",
        "region_variants.tsv.gz", "branch_reference.tsv", "branch_owned_markers.tsv.gz",
        "cluster_region_support.tsv.gz")]
    files += [protocol_path, split_path, Path(__file__)]
    return {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}


def run_development(audit_dir: Path, protocol_path: Path, split_path: Path, out: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    split_frame = pd.read_csv(split_path, sep="\t", dtype=str)
    if split_frame.patient_id.duplicated().any():
        raise RuntimeError("Duplicate split patient")
    if digest(split_path) != protocol["split_hash"]:
        raise RuntimeError("Split hash differs from frozen protocol")
    split = dict(zip(split_frame.patient_id, split_frame.split))
    observations = load_observations(audit_dir)
    scorer = load_scorer(audit_dir, protocol)
    models, fixed = {}, []
    for population in ("LUAD", "ALL_NSCLC"):
        patients = population_patients(observations, split, population, "development")
        model, raw = fit_population(patients, population, protocol, scorer)
        models[population] = model
        fixed.extend(raw)
        print(f"development fit {population}: patients={len(patients)}, CTMC_success={model['prior']['ctmc']['fit']['success']}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "FROZEN_MODELS.json"
    write_json(model_path, dict(models=models, protocol_sha256=digest(protocol_path),
        split_sha256=digest(split_path), code_sha256=digest(Path(__file__)),
        fit_settings=dict(ridge_alpha=RIDGE_ALPHA,
            prior_pseudocount=PRIOR_PSEUDOCOUNT, kernel_pseudocount=KERNEL_PSEUDOCOUNT,
            feature_policy="ONLY_EXPOSED_CORE_CALLS_POSITIVE_VAF_PUBLIC_REGION_CAPACITY",
            random_policy="UNIFORM_DISTINCT_NONDOMINATED_FEASIBLE_PLAN",
            fixed_policy="DEVELOPMENT_BEST_TEMPLATE_WITH_PUBLIC_CAPACITY_CLIPPING",
            kernel_policy="PATIENT_EQUAL_WEIGHT_EMPIRICAL_OR_REGION_SUBSET_KERNEL")))
    write_json(out / "MODEL_FREEZE.json", dict(status="FROZEN_BEFORE_CONFIRMATION_ACTIONS",
        model_sha256=digest(model_path), protocol_sha256=digest(protocol_path),
        split_sha256=digest(split_path), code_sha256=digest(Path(__file__)),
        source_files=source_manifest(audit_dir, protocol_path, split_path),
        independent_development_patients={p: len(m["prior"]["patient_ids"]) for p, m in models.items()},
        confirmation_patient_ids_used_for_fit=[]))
    pd.DataFrame(fixed).to_csv(out / "development_all_fixed_plans.tsv.gz", sep="\t", index=False)
    actions = pd.DataFrame(freeze_actions(observations, split, models, protocol, "development"))
    actions.to_csv(out / "development_actions.tsv", sep="\t", index=False)
    evaluated = evaluate_actions(actions, observations, models, protocol, scorer)
    evaluated.to_csv(out / "development_raw.tsv.gz", sep="\t", index=False)
    summarize_patients(evaluated).to_csv(out / "development_patient_summary.tsv", sep="\t", index=False)


def run_confirmation(audit_dir: Path, protocol_path: Path, split_path: Path,
                     model_path: Path, out: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    frozen = json.loads(model_path.read_text())
    freeze = json.loads(model_path.with_name("MODEL_FREEZE.json").read_text())
    if digest(model_path) != freeze["model_sha256"]:
        raise RuntimeError("Frozen model hash mismatch")
    for key, path in (("protocol_sha256", protocol_path), ("split_sha256", split_path),
                      ("code_sha256", Path(__file__))):
        if digest(path) != frozen[key]:
            raise RuntimeError(f"Frozen {key} mismatch")
    models = frozen["models"]
    sf = pd.read_csv(split_path, sep="\t", dtype=str)
    split = dict(zip(sf.patient_id, sf.split))
    for model in models.values():
        if any(split[p] != "development" for p in model["prior"]["patient_ids"]):
            raise RuntimeError("Non-development fit patient")
    observations = load_observations(audit_dir)
    actions = pd.DataFrame(freeze_actions(observations, split, models, protocol, "confirmation"))
    out.mkdir(parents=True, exist_ok=True)
    action_path = out / "confirmation_actions_frozen.tsv"
    actions.to_csv(action_path, sep="\t", index=False)
    write_json(out / "ACTION_FREEZE.json", dict(status="ACTIONS_WRITTEN_BEFORE_OPENING_SCORER",
        actions_sha256=digest(action_path), model_sha256=digest(model_path),
        protocol_sha256=digest(protocol_path), split_sha256=digest(split_path),
        code_sha256=digest(Path(__file__)), rows=len(actions),
        selection_interface_fields=list(VisibleCase.__dataclass_fields__),
        scorer_fields_exposed_to_policy=[], future_fields_exposed_to_policy=[]))
    # No confirmation branch/driver outcomes were available to choose_frozen.
    scorer = load_scorer(audit_dir, protocol)
    evaluated = evaluate_actions(actions, observations, models, protocol, scorer)
    evaluated.to_csv(out / "confirmation_raw.tsv.gz", sep="\t", index=False)
    summary = summarize_patients(evaluated)
    summary.to_csv(out / "confirmation_patient_summary.tsv", sep="\t", index=False)
    fixed = evaluate_fixed_confirmation(observations, split, models, protocol, scorer)
    fixed.to_csv(out / "confirmation_all_fixed_plans.tsv.gz", sep="\t", index=False)
    primary = summary[(summary.budget == protocol["acquisition"]["primary_budget"]) &
                      (summary.overhead == protocol["acquisition"]["primary_overhead"])]
    overview = primary.groupby(["population", "method"]).agg(
        patients=("patient_id", "nunique"), branch_recall=("branch_recall", "mean"),
        hidden_driver_recall=("hidden_driver_recall", "mean"),
        four_gene_log_loss=("four_gene_log_loss", "mean"), cost=("cost", "mean")).reset_index()
    overview.to_csv(out / "PRIMARY_OVERVIEW.tsv", sep="\t", index=False)
    write_json(out / "EXECUTION_MANIFEST.json", dict(status="COMPLETE", planned_cost_cells=9,
        planned_methods=list(METHODS), planned_perturbations_per_patient=5,
        independent_confirmation_patients=primary.groupby("population").patient_id.nunique().to_dict(),
        confirmation_rows=len(evaluated), fixed_plan_rows=len(fixed),
        patients_retained_including_zero_branch_denominators=True,
        post_confirmation_selection=False,
        policy_fit_or_parameter_change_in_confirmation=False,
        source_files=source_manifest(audit_dir, protocol_path, split_path)))
    print(overview.to_string(index=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["development", "confirmation"])
    p.add_argument("--audit-dir", type=Path, default=BASE / "data" / "native_audit")
    p.add_argument("--protocol", type=Path, default=BASE / "configs" / "PROTOCOL.json")
    p.add_argument("--split", type=Path, default=BASE / "configs" / "PATIENT_SPLIT.tsv")
    p.add_argument("--output-dir", type=Path, default=BASE / "results" / "acquisition")
    p.add_argument("--model", type=Path)
    args = p.parse_args()
    if args.phase == "development":
        run_development(args.audit_dir, args.protocol, args.split, args.output_dir)
    else:
        model = args.model or args.output_dir / "FROZEN_MODELS.json"
        run_confirmation(args.audit_dir, args.protocol, args.split, model, args.output_dir)


if __name__ == "__main__":
    main()
