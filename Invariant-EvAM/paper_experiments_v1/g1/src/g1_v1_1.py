from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
G1 = ROOT / "paper_experiments_v1/g1"
SHARED = ROOT / "paper_experiments_v1/shared"
sys.path[:0] = [str(ROOT / "src"), str(G1 / "src")]

from g1_core import (
    FrozenOperatorEnvironment, build_models, draw_parameters, expand_design, file_sha256,
    joint_profiled_geometry, latent_geometry, observable_output,
    primary_target_index, stable_seed, target_metrics,
)
from g1_pipeline import _geometry_for, load_config
from phase2.profile_intervals import fit_joint, sample_environment_counts


def load_v11():
    cfg = load_config()
    models = build_models(cfg)
    operators = json.loads((SHARED / "operator_library.json").read_text())["operators"]
    operators.update(json.loads((SHARED / "operator_library_v1_1.json").read_text())["operators"])
    old = pd.read_csv(SHARED / "design_registry.tsv", sep="\t")
    new = pd.read_csv(SHARED / "design_registry_v1_1.tsv", sep="\t")
    designs = []
    for row in pd.concat([old, new], ignore_index=True).to_dict("records"):
        row["operator_ids"] = json.loads(row["operator_ids"])
        row["nuisance_groups"] = json.loads(row["nuisance_groups"])
        designs.append(row)
    params = pd.read_csv(SHARED / "parameter_library_v1_1.tsv", sep="\t")
    params["theta"] = params.theta_json.map(lambda x: np.asarray(json.loads(x), dtype=float))
    return cfg, models, operators, designs, params


def sentinels(model_id, designs):
    by_id = {d["design_id"]: d for d in designs if d["model_id"] == model_id}
    return {
        "single": (by_id[f"{model_id}__D12"], [1]),
        "duplicated": (by_id[f"{model_id}__D00"], [2, 4]),
        "candidate_heterogeneous": (by_id[f"{model_id}__D12"], [2, 4]),
        "pure_cross_view_failure": (by_id[f"{model_id}__PURE_CROSS_VIEW_FAILURE"], [2, 4]),
    }


def freeze_stress():
    cfg = load_config(); models = build_models(cfg); rows = []
    for mi, (model_id, model) in enumerate(models.items()):
        for label, purpose, interval in [("rare", 14, [-4.0, -3.0]), ("high_frequency", 15, [0.2, 0.8])]:
            draws = draw_parameters(model, cfg["seed_master"], mi, purpose, 2, interval,
                                    cfg["parameter_design"]["interaction_magnitudes"])
            for draw_id, theta in enumerate(draws):
                payload = json.dumps(theta.tolist(), separators=(",", ":"))
                rows.append({"protocol_version":"1.2", "model_id":model_id, "purpose":"stress",
                    "stress_class":label, "draw_id":draw_id, "rng_purpose_id":purpose,
                    "baseline_interval":json.dumps(interval), "theta_json":payload,
                    "theta_sha256":__import__("hashlib").sha256(payload.encode()).hexdigest(),
                    "screened_on_performance":False})
    path = SHARED / "parameter_library_stress_v1_2.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    payload = {"protocol_version":"1.2", "status":"FROZEN",
        "frozen_before_stress_outcomes":True,
        "rule":"two rare and two high-frequency draws per core model; no performance screening",
        "file":str(path.relative_to(ROOT)), "sha256":file_sha256(path), "bytes":path.stat().st_size}
    (SHARED / "G1_STRESS_ASSET_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("stress parameters frozen", len(rows))


def run_stress():
    cfg, models, operators, designs, _ = load_v11()
    params = pd.read_csv(SHARED / "parameter_library_stress_v1_2.tsv", sep="\t")
    params["theta"] = params.theta_json.map(lambda x: np.asarray(json.loads(x), dtype=float))
    rtol=cfg["numerical"]["rank_rtol"]; sensitivity=cfg["numerical"]["rank_rtol_sensitivity"]
    rows=[]
    for model_id, model in models.items():
        primary=np.eye(model.parameter_count)[primary_target_index(model)]
        model_designs=[d for d in designs if d["model_id"]==model_id]
        for _,pr in params[params.model_id==model_id].iterrows():
            source=latent_geometry(model,pr.theta,rtol)
            for design in model_designs:
                ids,groups=expand_design(design,2)
                for knowledge in cfg["operator_design"]["knowledge"]:
                    g,_,_= _geometry_for(model,pr.theta,ids,groups,operators,knowledge,rtol,sensitivity)
                    tm=target_metrics(primary,g,rtol)
                    rows.append({"protocol_version":"1.2","evidence_type":"NEW_RUN_STRESS",
                        "model_id":model_id,"stress_class":pr.stress_class,"parameter_draw_id":int(pr.draw_id),
                        "design_id":design["design_id"],"design_family":design["family"],
                        "operator_knowledge":knowledge,"rank":g.rank,"source_rank_cap":source.rank,
                        "primary_target_identified":tm["identified"],"primary_null_leakage":tm["null_leakage"],
                        "target_information_per_sample":tm["target_information_per_sample"],
                        "rank_tolerance_stable":len(set(g.rank_by_tolerance.values()))==1})
    pd.DataFrame(rows).to_csv(G1/"results/G1_stress_geometry.tsv",sep="\t",index=False)
    print("stress rows",len(rows))


def run_population():
    cfg, models, operators, designs, params = load_v11()
    rtol = cfg["numerical"]["rank_rtol"]
    sensitivity = cfg["numerical"]["rank_rtol_sensitivity"]
    rows = []
    started = time.time()
    for model_id, model in models.items():
        primary = np.eye(model.parameter_count)[primary_target_index(model)]
        for _, pr in params[params.model_id == model_id].iterrows():
            theta = pr.theta
            source = latent_geometry(model, theta, rtol)
            for scenario, (design, e_values) in sentinels(model_id, designs).items():
                for environments in e_values:
                    ids, groups = expand_design(design, environments)
                    for knowledge in cfg["operator_design"]["knowledge"]:
                        geometry, _, acceptance = _geometry_for(
                            model, theta, ids, groups, operators, knowledge, rtol, sensitivity
                        )
                        tm = target_metrics(primary, geometry, rtol)
                        widths = [operators[x]["panel_width"] for x in ids]
                        common = {
                            "protocol_version": "1.1", "evidence_type": "NEW_RUN_CONFIRMATION",
                            "model_id": model_id, "parameter_draw_id": int(pr.draw_id),
                            "parameter_stream": "confirmation_v1_1_seed_purpose_13",
                            "scenario": scenario, "design_id": design["design_id"],
                            "operator_knowledge": knowledge, "environments": environments,
                            "rank": geometry.rank, "source_rank_cap": source.rank,
                            "rank_gain_over_first_environment": geometry.rank - _geometry_for(
                                model, theta, ids[:1], groups[:1], operators, knowledge, rtol, sensitivity
                            )[0].rank,
                            "primary_target_identified": tm["identified"],
                            "primary_null_leakage": tm["null_leakage"],
                            "target_information_per_sample": tm["target_information_per_sample"],
                            "sigma_min_per_sample": geometry.sigma_min_all,
                            "rank_tolerance_stable": len(set(geometry.rank_by_tolerance.values())) == 1,
                            "target_joint_observed": bool(design.get("target_joint_observed", False)),
                            "target_related_selection_path": bool(design.get("target_related_selection_path", scenario != "pure_cross_view_failure")),
                            "mean_acceptance_probability": float(np.mean(acceptance)),
                            "panel_width_total": int(sum(widths)),
                        }
                        for n_total in cfg["population_geometry"]["n_total"]:
                            rows.append({**common, "budget_basis": "fixed_enrolled_patients",
                                "budget_value": n_total, "n_enrolled": float(n_total),
                                "measurement_proxy": float(n_total * np.mean(widths)),
                                "primary_target_information_total": float(n_total * tm["target_information_per_sample"]),
                                "expected_screened": float(n_total * np.mean([1 / x for x in acceptance]))})
                        proxy = 1000 * model.d
                        n_each = proxy / sum(widths)
                        n_enrolled = n_each * environments
                        rows.append({**common, "budget_basis": "fixed_measurement_proxy",
                            "budget_value": proxy, "n_enrolled": float(n_enrolled),
                            "measurement_proxy": float(proxy),
                            "primary_target_information_total": float(n_enrolled * tm["target_information_per_sample"]),
                            "expected_screened": float(n_enrolled * np.mean([1 / x for x in acceptance]))})
        print("v1.1 population", model_id, "elapsed", round(time.time() - started, 1), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(G1 / "results/G1A_population_geometry_v1_1.tsv", sep="\t", index=False)
    out[out.budget_basis == "fixed_measurement_proxy"].to_csv(
        G1 / "results/G1A_budget_matched_v1_1.tsv", sep="\t", index=False)
    print("v1.1 sentinel rows", len(out))


def run_finite():
    cfg, models, operators, designs, params = load_v11()
    model_id = cfg["finite_sample_illustrations"]["model"]
    model = models[model_id]
    theta = params[(params.model_id == model_id) & (params.draw_id == 0)].iloc[0].theta
    primary_idx = primary_target_index(model)
    rows = []
    started = time.time()
    for scenario_index, (scenario, (design, e_values)) in enumerate(sentinels(model_id, designs).items()):
        environments = e_values[0]
        ids, groups = expand_design(design, environments)
        envs = [FrozenOperatorEnvironment(operators[x], "K1_STRUCTURED") for x in ids]
        phis = [np.asarray(operators[x]["phi_true"], dtype=float) for x in ids]
        base, jbase = model.distribution_and_jacobian(theta)
        outputs = [observable_output(model, base, jbase, operators[x], "K1_STRUCTURED")[:3] for x in ids]
        geometry = joint_profiled_geometry(outputs, np.ones(environments), groups,
            cfg["numerical"]["rank_rtol"], cfg["numerical"]["rank_rtol_sensitivity"])
        tm = target_metrics(np.eye(model.parameter_count)[primary_idx], geometry, cfg["numerical"]["rank_rtol"])
        for n_index, n_total in enumerate(cfg["finite_sample_illustrations"]["n_total"]):
            sizes = [n_total // environments] * environments
            sizes[-1] += n_total - sum(sizes)
            for replicate in range(cfg["finite_sample_illustrations"]["replicates"]):
                seedseq = stable_seed(cfg["seed_master"], 1, 41, scenario_index, n_index, replicate)
                rng = np.random.default_rng(seedseq)
                counts = sample_environment_counts(model, envs, theta, phis, sizes, rng)
                t0 = time.time()
                fit = fit_joint(model, envs, counts, theta_start=np.zeros(model.parameter_count),
                                theta_bound=3.0, phi_bound=2.0)
                estimate = float(fit.theta[primary_idx])
                rows.append({
                    "protocol_version": "1.1", "evidence_type": "NEW_RUN_ILLUSTRATIVE",
                    "model_id": model_id, "parameter_draw_id": 0,
                    "parameter_stream": "confirmation_v1_1_seed_purpose_13",
                    "scenario": scenario, "design_id": design["design_id"],
                    "operator_knowledge": "K1_STRUCTURED", "n_unique_patients": n_total,
                    "n_observed": n_total, "replicate_id": replicate,
                    "seed_entropy": json.dumps(seedseq.entropy), "truth": float(theta[primary_idx]),
                    "estimate": estimate, "error": estimate - float(theta[primary_idx]),
                    "absolute_error": abs(estimate - float(theta[primary_idx])),
                    "rank": geometry.rank, "sigma_min_per_sample": geometry.sigma_min_all,
                    "structural_status": "IDENTIFIED" if tm["identified"] else "NONIDENTIFIED",
                    "computational_status": "COMPLETE" if fit.success else "OPTIMIZER_FAILED",
                    "runtime_seconds": time.time() - t0, "coverage_claim_allowed": False,
                })
            print("finite v1.1", scenario, n_total, "elapsed", round(time.time() - started, 1), flush=True)
    pd.DataFrame(rows).to_csv(G1 / "results/G1_finite_sample_illustrations_v1_1.tsv", sep="\t", index=False)
    print("finite rows", len(rows), "seconds", round(time.time() - started, 1))


def validate_numerics():
    cfg, models, _, _, params = load_v11()
    rows = []
    for model_id, model in models.items():
        for _, pr in params[params.model_id == model_id].head(3).iterrows():
            theta = pr.theta
            p, jac = model.distribution_and_jacobian(theta)
            for step in cfg["numerical"]["finite_difference_steps"]:
                fd = np.column_stack([(model.distribution(theta + np.eye(model.parameter_count)[j] * step) -
                                       model.distribution(theta - np.eye(model.parameter_count)[j] * step)) / (2 * step)
                                      for j in range(model.parameter_count)])
                abs_err = float(np.max(np.abs(fd - jac)))
                rel_err = float(np.max(np.abs(fd - jac) / np.maximum(np.abs(jac), 1e-10)))
                informative = np.abs(jac) >= 1e-6
                rel_informative = float(np.max(np.abs(fd - jac)[informative] / np.abs(jac)[informative])) if informative.any() else math.nan
                frobenius_relative = float(np.linalg.norm(fd - jac) / max(np.linalg.norm(jac), 1e-15))
                rows.append({"model_id": model_id, "parameter_draw_id": int(pr.draw_id),
                    "finite_difference_step": step, "max_absolute_error": abs_err,
                    "max_relative_error_floor_1e-10": rel_err,
                    "max_relative_error_where_abs_jac_ge_1e-6": rel_informative,
                    "frobenius_relative_error": frobenius_relative,
                    "probability_sum_error": abs(float(p.sum()) - 1),
                    "minimum_probability": float(p.min())})
    pd.DataFrame(rows).to_csv(G1 / "results/G1_numerical_validation.tsv", sep="\t", index=False)


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (math.nan, math.nan)
    p = k / n; den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return centre-half, centre+half


def summarize():
    geometry = pd.read_csv(G1 / "results/G1_population_geometry.tsv", sep="\t")
    keys = ["model_id", "operator_knowledge", "design_family", "target_joint_observed", "predeclared_failure"]
    rows = []
    for key, f in geometry.groupby(keys, dropna=False):
        n = len(f); kg = int((f.rank_gain > 0).sum()); kr = int(f.primary_target_rescued.sum())
        gl, gu = wilson(kg, n); rl, ru = wilson(kr, n)
        rows.append({**dict(zip(keys, key)), "eligible_evaluations": n,
            "numerically_completed": int((f.numerical_status == "COMPLETE").sum()),
            "rank_gain_count": kg, "rank_gain_fraction": kg/n,
            "rank_gain_wilson95_low": gl, "rank_gain_wilson95_high": gu,
            "target_rescue_count": kr, "target_rescue_fraction": kr/n,
            "target_rescue_wilson95_low": rl, "target_rescue_wilson95_high": ru,
            "target_identified_fraction": float(f.primary_target_identified.mean()),
            "rank_tolerance_unstable_count": int((~f.rank_tolerance_stable).sum()),
            "median_target_information_per_sample": float(f.primary_target_information_per_sample.median())})
    pd.DataFrame(rows).to_csv(G1 / "results/G1_design_yield_with_intervals.tsv", sep="\t", index=False)

    operators = json.loads((SHARED / "operator_library.json").read_text())["operators"]
    designs = pd.read_csv(SHARED / "design_registry.tsv", sep="\t")
    audit = []
    for _, d in designs.iterrows():
        ids = json.loads(d.operator_ids)
        paths = []
        no_selection = []
        for oid in ids:
            feat = operators[oid]["feature"]
            terms = feat.get("terms", [])
            related = feat["kind"] == "burden" or any(0 in term or 1 in term for term in terms)
            paths.append(bool(related))
            no_selection.append(feat["kind"] == "none")
        audit.append({"design_id": d.design_id, "model_id": d.model_id, "design_family": d.family,
            "predeclared_failure_v1_0": bool(d.predeclared_failure),
            "target_joint_observed": bool(d.target_joint_observed),
            "environment_target_related_selection": json.dumps(paths),
            "any_target_related_selection_path": any(paths),
            "all_environments_no_selection": all(no_selection),
            "valid_pure_cross_view_negative_control": (not bool(d.target_joint_observed)) and all(no_selection),
            "interpretation": "selection_can_carry_latent_target_information" if any(paths) else
                              ("pure_assay_only" if all(no_selection) else "selection_on_other_latent_states_remains")})
    pd.DataFrame(audit).to_csv(G1 / "results/G1_design_path_audit.tsv", sep="\t", index=False)


def manifest():
    manifest_path = G1 / "runs/G1_OUTPUT_MANIFEST.json"
    files = sorted(p for parent in [G1, SHARED]
                   for p in parent.rglob("*") if p.is_file() and p != manifest_path)
    payload = {"group": "G1", "scope": "01_INFORMATION_AND_COMPLEMENTARITY",
        "status": "executed", "generated_at": "2026-09-23",
        "files": {str(p.relative_to(ROOT)): {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in files}}
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["freeze-stress", "stress", "population", "finite", "validate", "summarize", "manifest"])
    cmd = p.parse_args().command
    {"freeze-stress": freeze_stress, "stress": run_stress,
     "population": run_population, "finite": run_finite, "validate": validate_numerics,
     "summarize": summarize, "manifest": manifest}[cmd]()


if __name__ == "__main__":
    main()
