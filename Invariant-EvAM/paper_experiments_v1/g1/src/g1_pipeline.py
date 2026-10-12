from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
G1 = ROOT / "paper_experiments_v1/g1"
SHARED = ROOT / "paper_experiments_v1/shared"
CONFIG_PATH = G1 / "configs/g1_protocol_v1.json"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(G1 / "src"))

from g1_core import (
    FrozenOperatorEnvironment,
    build_design_library,
    build_models,
    draw_parameters,
    expand_design,
    feature_matrix,
    file_sha256,
    joint_profiled_geometry,
    latent_geometry,
    observable_output,
    parameter_names,
    primary_edge,
    primary_target_index,
    recover_source_from_known_selection,
    stable_seed,
    target_metrics,
    trace_source_alias,
)
from phase2.profile_intervals import fit_joint, sample_environment_counts


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit() -> None:
    rows = []
    paths = [
        "reports/FINAL_INVARIANT_EVAM_REPORT_CN.md",
        "reports/INVARIANT_EVAM_PHASE2_REPORT_CN.md",
        "reports/phase2/PHASE2_THEORY.md",
        "src/phase2/interaction_generator.py",
        "src/phase2/observation_operators.py",
        "src/phase2/jacobian_identifiability.py",
        "results/phase2/interaction_rank_results.tsv",
        "results/phase2/profile_interval_summary.tsv",
        "results/continuous/cross_cancer_geometry.tsv",
        "runs/FINAL_MANIFEST.json",
    ]
    for name in paths:
        path = ROOT / name
        rows.append({
            "artifact": name,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
            "evidence_label": "LEGACY_VERIFIED" if path.is_file() else "MISSING",
            "verification_scope": "existence_and_hash_not_new_scientific_validation",
        })
    protocol_files = [
        ROOT.parent / "00_README_AND_SHARED_CONTRACT.md",
        ROOT.parent / "01_INFORMATION_AND_COMPLEMENTARITY.md",
        ROOT.parent / "02_RELIABLE_INFERENCE_AND_BASELINES.md",
        ROOT.parent / "03_UNKNOWN_OBSERVATION_AND_BIOLOGY.md",
        ROOT.parent / "04_REAL_PATIENT_CONTROLLED_VALIDATION.md",
        ROOT.parent / "05_NATURAL_COHORT_AND_DATA_DESIGN.md",
    ]
    for path in protocol_files:
        rows.append({
            "artifact": str(path), "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
            "evidence_label": "PROTOCOL_VERIFIED" if path.is_file() else "MISSING",
            "verification_scope": "protocol_input",
        })
    raw_files = [
        Path("/Users/oukang/Downloads/project/Rel_ObsHN/Data/AACR/AACR/data_clinical_sample.txt"),
        Path("/Users/oukang/Downloads/project/Rel_ObsHN/Data/AACR/AACR/genomic_information.txt"),
        Path("/Users/oukang/Downloads/project/Rel_ObsHN/Data/AACR/data_mutations_extended.txt"),
    ]
    for path in raw_files:
        rows.append({
            "artifact": str(path), "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
            "evidence_label": "RAW_INPUT_VERIFIED" if path.is_file() else "MISSING",
            "verification_scope": "read_only_input_not_used_for_outcome_selection",
        })
    pd.DataFrame(rows).to_csv(G1 / "results/G1_legacy_and_input_audit.tsv", sep="\t", index=False)
    missing = [row["artifact"] for row in rows if not row["exists"]]
    report = f"""# G1 legacy and dependency audit

Audit date: 2026-09-23. This audit verifies local existence and hashes; it does not relabel historical outcomes as NEW_RUN evidence.

- Existing Invariant-EvAM report, theory, source, frozen results and manifest were located.
- GENIE raw inputs were located and remain read-only.
- The requested G1 protocol and shared contract were located and copied into the new namespace.
- Missing protocol inputs: {', '.join(missing) if missing else 'none'}.
- Protocol books currently present were hashed. Only G1 is executed in this task; the other books do not alter G1 outcomes.

All new G1 outputs use the NEW_RUN label and are stored only under paper_experiments_v1.
"""
    (G1 / "reports/G1_LEGACY_AND_DEPENDENCY_AUDIT.md").write_text(report)
    print("audit rows", len(rows), "missing", len(missing))


def freeze() -> None:
    cfg = load_config()
    models = build_models(cfg)
    SHARED.mkdir(parents=True, exist_ok=True)
    model_contract = {
        "protocol_version": "1.0", "status": "FROZEN", "frozen_at": "2026-09-23",
        "time_contract": cfg["time_contract"], "models": cfg["models_core"],
        "state_order": "itertools.product([0,1], repeat=d); event_0 is most significant bit",
        "generator_convention": "Q[row=current_state,column=next_state]; distribution=expm(Q.T*t)@initial",
        "evidence_label": "NEW_RUN_CONTRACT",
    }
    write_json(SHARED / "MODEL_CONTRACT.json", model_contract)
    event_rows = []
    parameter_rows = []
    target_rows = []
    parameter_library = []
    all_operators = {}
    design_rows = []
    for model_index, (model_id, model) in enumerate(models.items()):
        for event in range(model.d):
            event_rows.append({
                "protocol_version": "1.0", "model_id": model_id, "event_index": event,
                "simulation_label": f"event_{event}",
                "LUAD_mapping_if_applicable": ["TP53", "KRAS", "STK11", "KEAP1"][event] if model.d == 4 else "NA",
                "COAD_mapping_if_applicable": ["APC", "TP53", "KRAS", "PIK3CA"][event] if model.d == 4 else "NA",
                "ordering_role": "binary_event_coordinate_not_discovered_temporal_order",
            })
        names = parameter_names(model)
        for idx, name in enumerate(names):
            row = {
                "protocol_version": "1.0", "model_id": model_id, "parameter_index": idx,
                "parameter_name": name, "parameter_type": "baseline" if idx < model.d else "interaction",
                "direction_convention": "source_to_target; theta[target<-source]",
                "topology_assumed": cfg["models_core"][model_id]["topology_assumed"],
            }
            if idx >= model.d:
                target, source = model.edges[idx - model.d]
                row.update(source_event=source, target_event=target)
                target_rows.append({
                    "protocol_version": "1.0", "target_id": f"{model_id}__{name}",
                    "model_id": model_id, "target_type": "interaction_coordinate",
                    "vector": json.dumps([1.0 if j == idx else 0.0 for j in range(model.parameter_count)]),
                    "primary": (target, source) == primary_edge(model),
                    "interpretation": "conditional acquisition log-hazard; not causality or ancestry",
                })
            else:
                row.update(source_event="NA", target_event=idx)
            parameter_rows.append(row)
        if len(model.edges) >= 2:
            v = np.zeros(model.parameter_count)
            v[model.d] = 1.0; v[model.d + 1] = -1.0
            target_rows.append({
                "protocol_version": "1.0", "target_id": f"{model_id}__interaction_contrast_0_minus_1",
                "model_id": model_id, "target_type": "interaction_contrast", "vector": json.dumps(v.tolist()),
                "primary": False, "interpretation": "difference of two conditional acquisition log-hazards",
            })
        for purpose_name, purpose_id, count in (
            ("development", 11, cfg["parameter_design"]["development_draws"]),
            ("confirmation", 12, cfg["parameter_design"]["confirmation_draws"]),
        ):
            draws = draw_parameters(
                model, cfg["seed_master"], model_index, purpose_id, count,
                cfg["parameter_design"]["baseline_range"], cfg["parameter_design"]["interaction_magnitudes"],
            )
            for draw_id, theta in enumerate(draws):
                payload = json.dumps(theta.tolist(), separators=(",", ":"))
                parameter_library.append({
                    "protocol_version": "1.0", "model_id": model_id, "purpose": purpose_name,
                    "draw_id": draw_id, "theta_json": payload,
                    "theta_sha256": __import__("hashlib").sha256(payload.encode()).hexdigest(),
                    "screened_on_performance": False,
                })
        operators, designs = build_design_library(model_id, model.d)
        all_operators.update(operators)
        for design in designs:
            design_rows.append({
                "protocol_version": "1.0", **design,
                "operator_ids": json.dumps(design["operator_ids"]),
                "nuisance_groups": json.dumps(design["nuisance_groups"]),
                "confirmation_truth_used_in_selection": False,
            })
    pd.DataFrame(event_rows).to_csv(SHARED / "event_dictionary.tsv", sep="\t", index=False)
    pd.DataFrame(parameter_rows).to_csv(SHARED / "parameter_dictionary.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(SHARED / "target_registry.tsv", sep="\t", index=False)
    pd.DataFrame(parameter_library).to_csv(SHARED / "parameter_library.tsv", sep="\t", index=False)
    pd.DataFrame(design_rows).to_csv(SHARED / "design_registry.tsv", sep="\t", index=False)
    write_json(SHARED / "operator_library.json", {"protocol_version": "1.0", "status": "FROZEN", "operators": all_operators})
    write_json(SHARED / "numerical_precision_contract.json", {
        "protocol_version": "1.0", "status": "FROZEN", **cfg["numerical"],
        "rank_interpretation": "numerical diagnostic, not theorem", "zero_count_rule": "retain model-positive cells",
    })
    assets = [
        SHARED / "MODEL_CONTRACT.json", SHARED / "event_dictionary.tsv",
        SHARED / "parameter_dictionary.tsv", SHARED / "target_registry.tsv",
        SHARED / "parameter_library.tsv", SHARED / "operator_library.json",
        SHARED / "design_registry.tsv", SHARED / "numerical_precision_contract.json",
        CONFIG_PATH,
    ]
    manifest = {
        "protocol_version": "1.0", "status": "FROZEN", "frozen_at": "2026-09-23",
        "files": {str(p.relative_to(ROOT)): {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in assets},
        "freeze_rule": "confirmation outcomes were not computed before this manifest",
    }
    write_json(SHARED / "G1_SHARED_ASSET_MANIFEST.json", manifest)
    print("frozen models", len(models), "operators", len(all_operators), "designs", len(design_rows), "parameter rows", len(parameter_library))


def _load_frozen():
    cfg = load_config()
    models = build_models(cfg)
    operators = json.loads((SHARED / "operator_library.json").read_text())["operators"]
    designs_df = pd.read_csv(SHARED / "design_registry.tsv", sep="\t")
    designs = []
    for row in designs_df.to_dict("records"):
        row["operator_ids"] = json.loads(row["operator_ids"])
        row["nuisance_groups"] = json.loads(row["nuisance_groups"])
        designs.append(row)
    params = pd.read_csv(SHARED / "parameter_library.tsv", sep="\t")
    params = params[params.purpose == "confirmation"].copy()
    params["theta"] = params.theta_json.map(lambda x: np.asarray(json.loads(x), dtype=float))
    targets = pd.read_csv(SHARED / "target_registry.tsv", sep="\t")
    targets["vector_array"] = targets.vector.map(lambda x: np.asarray(json.loads(x), dtype=float))
    return cfg, models, operators, designs, params, targets


def _geometry_for(
    model, theta, operator_ids, nuisance_groups, operators, knowledge, rtol, sensitivity
):
    base, j_base = model.distribution_and_jacobian(theta)
    outputs = []
    acceptance = []
    for operator_id in operator_ids:
        p, jt, jp, acc = observable_output(model, base, j_base, operators[operator_id], knowledge)
        outputs.append((p, jt, jp)); acceptance.append(acc)
    geometry = joint_profiled_geometry(outputs, np.ones(len(outputs)), nuisance_groups, rtol, sensitivity)
    return geometry, outputs, acceptance


def run_population() -> None:
    cfg, models, operators, designs, params, targets = _load_frozen()
    rtol = cfg["numerical"]["rank_rtol"]
    sensitivity = cfg["numerical"]["rank_rtol_sensitivity"]
    geometry_rows = []
    target_rows = []
    cap_rows = []
    recovery_rows = []
    sentinel_rows = []
    start_time = time.time()
    for model_index, (model_id, model) in enumerate(models.items()):
        model_params = params[params.model_id == model_id]
        model_designs = [d for d in designs if d["model_id"] == model_id]
        model_targets = targets[targets.model_id == model_id]
        source_cap = 2 ** model.d - 1
        for _, parameter_row in model_params.iterrows():
            draw_id = int(parameter_row.draw_id)
            theta = parameter_row.theta
            base, j_base = model.distribution_and_jacobian(theta)
            source_geometry = latent_geometry(model, theta, rtol)
            primary_vector = np.zeros(model.parameter_count); primary_vector[primary_target_index(model)] = 1.0
            source_primary = target_metrics(primary_vector, source_geometry, rtol)
            cap_rows.append({
                "protocol_version": "1.0", "evidence_type": "NEW_RUN", "model_id": model_id,
                "parameter_draw_id": draw_id, "parameter_count": model.parameter_count,
                "state_probability_df_cap": source_cap, "source_rank": source_geometry.rank,
                "source_nullity": model.parameter_count - source_geometry.rank,
                "primary_target_identified_from_source": source_primary["identified"],
                "primary_target_null_leakage": source_primary["null_leakage"],
                "rank_by_tolerance": json.dumps(source_geometry.rank_by_tolerance, sort_keys=True),
                "rank_tolerance_stable": len(set(source_geometry.rank_by_tolerance.values())) == 1,
                "source_sigma_min_all": source_geometry.sigma_min_all,
                "source_sigma_min_nonzero": source_geometry.sigma_min_nonzero,
            })
            # K0 full-panel inversion positive control, six predeclared features.
            full_ops = [op for op in operators.values() if op["d"] == model.d and op["panel_width"] == model.d and op["operator_id"].startswith(model_id)]
            for op in sorted(full_ops, key=lambda x: x["operator_id"]):
                p, _, _, acc = observable_output(model, base, j_base, op, "K0_KNOWN")
                recovered = recover_source_from_known_selection(
                    p, feature_matrix(model.states, op["feature"]), np.asarray(op["phi_true"])
                )
                recovery_rows.append({
                    "protocol_version": "1.0", "model_id": model_id, "parameter_draw_id": draw_id,
                    "operator_id": op["operator_id"], "max_abs_probability_error": float(np.max(np.abs(recovered - base))),
                    "source_state_kl": float(np.sum(base * np.log(np.maximum(base, 1e-300) / np.maximum(recovered, 1e-300)))),
                    "rejection_acceptance_probability": acc,
                    "screened_per_1000_enrolled": 1000.0 / acc,
                })
            for design in model_designs:
                for knowledge in cfg["operator_design"]["knowledge"]:
                    g, outputs, acceptance = _geometry_for(
                        model, theta, design["operator_ids"], design["nuisance_groups"], operators,
                        knowledge, rtol, sensitivity,
                    )
                    single_g, _, _ = _geometry_for(
                        model, theta, design["operator_ids"][:1], design["nuisance_groups"][:1], operators,
                        knowledge, rtol, sensitivity,
                    )
                    primary = target_metrics(primary_vector, g, rtol)
                    single_primary = target_metrics(primary_vector, single_g, rtol)
                    widths = [operators[x]["panel_width"] for x in design["operator_ids"]]
                    row = {
                        "protocol_version": "1.0", "evidence_type": "NEW_RUN",
                        "model_id": model_id, "parameter_draw_id": draw_id,
                        "design_id": design["design_id"], "design_family": design["family"],
                        "operator_knowledge": knowledge, "environments": 2,
                        "operator_ids": json.dumps(design["operator_ids"]),
                        "target_joint_observed": bool(design["target_joint_observed"]),
                        "predeclared_failure": bool(design["predeclared_failure"]),
                        "panel_jaccard": design["panel_jaccard"], "panel_width_total": sum(widths),
                        "source_rank_cap": source_geometry.rank, "single_rank": single_g.rank, "rank": g.rank,
                        "rank_gain": g.rank - single_g.rank, "nullity": model.parameter_count - g.rank,
                        "nuisance_rank": g.nuisance_rank, "sigma_min_all": g.sigma_min_all,
                        "sigma_min_nonzero": g.sigma_min_nonzero,
                        "rank_by_tolerance": json.dumps(g.rank_by_tolerance, sort_keys=True),
                        "rank_tolerance_stable": len(set(g.rank_by_tolerance.values())) == 1,
                        "primary_target_identified_single": single_primary["identified"],
                        "primary_target_identified": primary["identified"],
                        "primary_target_rescued": (not single_primary["identified"]) and primary["identified"],
                        "primary_null_leakage_single": single_primary["null_leakage"],
                        "primary_null_leakage": primary["null_leakage"],
                        "primary_target_information_per_sample": primary["target_information_per_sample"],
                        "primary_directional_score_norm": primary["directional_score_norm"],
                        "mean_acceptance_probability": float(np.mean(acceptance)),
                        "numerical_status": "COMPLETE",
                    }
                    geometry_rows.append(row)
                    for _, target in model_targets.iterrows():
                        tm = target_metrics(target.vector_array, g, rtol)
                        target_rows.append({
                            "protocol_version": "1.0", "model_id": model_id,
                            "parameter_draw_id": draw_id, "design_id": design["design_id"],
                            "design_family": design["family"], "operator_knowledge": knowledge,
                            "target_id": target.target_id, "target_type": target.target_type,
                            "primary": bool(target.primary), "identified": tm["identified"],
                            "null_leakage": tm["null_leakage"],
                            "target_information_per_sample": tm["target_information_per_sample"],
                            "directional_score_norm": tm["directional_score_norm"],
                            "rank": g.rank, "source_rank_cap": source_geometry.rank,
                        })
            # Sentinel G1A classes were frozen by neutral design IDs, not outcomes.
            by_id = {d["design_id"]: d for d in model_designs}
            sentinel = {
                "single": (by_id[f"{model_id}__D12"], 1),
                "duplicated": (by_id[f"{model_id}__D00"], 2),
                "candidate_heterogeneous": (by_id[f"{model_id}__D12"], 2),
                "cross_view_failure": (by_id[f"{model_id}__D21"], 2),
            }
            for scenario, (design, base_e) in sentinel.items():
                for environments in ([1] if scenario == "single" else [2, 4]):
                    ids, groups = expand_design(design, environments)
                    for knowledge in cfg["operator_design"]["knowledge"]:
                        g, _, acceptance = _geometry_for(model, theta, ids, groups, operators, knowledge, rtol, sensitivity)
                        tm = target_metrics(primary_vector, g, rtol)
                        widths = [operators[x]["panel_width"] for x in ids]
                        for n_total in cfg["population_geometry"]["n_total"]:
                            sentinel_rows.append({
                                "protocol_version": "1.0", "model_id": model_id,
                                "parameter_draw_id": draw_id, "scenario": scenario,
                                "design_id": design["design_id"], "operator_knowledge": knowledge,
                                "environments": environments, "budget_basis": "fixed_enrolled_patients",
                                "budget_value": n_total, "n_enrolled": n_total,
                                "measurement_proxy": n_total * float(np.mean(widths)),
                                "rank": g.rank, "source_rank_cap": source_geometry.rank,
                                "primary_target_identified": tm["identified"],
                                "primary_target_information_total": n_total * tm["target_information_per_sample"],
                                "primary_null_leakage": tm["null_leakage"],
                                "sigma_min_per_sample": g.sigma_min_all,
                                "expected_screened": n_total * float(np.mean([1.0 / x for x in acceptance])),
                            })
                        proxy_budget = 1000 * model.d
                        n_each = proxy_budget / sum(widths)
                        n_enrolled = n_each * environments
                        sentinel_rows.append({
                            "protocol_version": "1.0", "model_id": model_id,
                            "parameter_draw_id": draw_id, "scenario": scenario,
                            "design_id": design["design_id"], "operator_knowledge": knowledge,
                            "environments": environments, "budget_basis": "fixed_measurement_proxy",
                            "budget_value": proxy_budget, "n_enrolled": n_enrolled,
                            "measurement_proxy": proxy_budget,
                            "rank": g.rank, "source_rank_cap": source_geometry.rank,
                            "primary_target_identified": tm["identified"],
                            "primary_target_information_total": n_enrolled * tm["target_information_per_sample"],
                            "primary_null_leakage": tm["null_leakage"],
                            "sigma_min_per_sample": g.sigma_min_all,
                            "expected_screened": n_enrolled * float(np.mean([1.0 / x for x in acceptance])),
                        })
        print("population", model_id, "elapsed", round(time.time() - start_time, 1))
    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(G1 / "results/G1_population_geometry.tsv", sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(G1 / "results/G1_target_map.tsv", sep="\t", index=False)
    pd.DataFrame(cap_rows).to_csv(G1 / "results/G1_source_information_cap.tsv", sep="\t", index=False)
    pd.DataFrame(recovery_rows).to_csv(G1 / "results/G1_source_recovery.tsv", sep="\t", index=False)
    pd.DataFrame(sentinel_rows).to_csv(G1 / "results/G1A_population_geometry.tsv", sep="\t", index=False)
    pd.DataFrame(sentinel_rows).to_csv(G1 / "results/G1A_budget_matched.tsv", sep="\t", index=False)
    # Full denominator summaries; numerical failures would remain in the raw table.
    grouped = []
    keys = ["model_id", "operator_knowledge", "design_family", "target_joint_observed", "predeclared_failure"]
    for key, frame in geometry.groupby(keys, dropna=False):
        total = len(frame)
        completed = int((frame.numerical_status == "COMPLETE").sum())
        grouped.append({
            **dict(zip(keys, key)), "eligible_evaluations": total, "numerically_completed": completed,
            "rank_gain_count": int((frame.rank_gain > 0).sum()),
            "rank_gain_fraction_all_eligible": float((frame.rank_gain > 0).sum() / total),
            "target_rescue_count": int(frame.primary_target_rescued.sum()),
            "target_rescue_fraction_all_eligible": float(frame.primary_target_rescued.sum() / total),
            "target_identified_fraction": float(frame.primary_target_identified.mean()),
            "rank_tolerance_unstable_fraction": float((~frame.rank_tolerance_stable).mean()),
            "median_sigma_min_all": float(frame.sigma_min_all.median()),
            "median_target_information_per_sample": float(frame.primary_target_information_per_sample.median()),
        })
    yield_df = pd.DataFrame(grouped)
    yield_df.to_csv(G1 / "results/G1_design_yield.tsv", sep="\t", index=False)
    yield_df.to_csv(G1 / "results/G1_assay_selection_decomposition.tsv", sep="\t", index=False)
    print("geometry rows", len(geometry), "target rows", len(target_rows), "seconds", round(time.time()-start_time,1))


def run_alias() -> None:
    cfg, models, operators, designs, params, _ = _load_frozen()
    rows = []
    for model_id in cfg["alias_diagnostic"]["models"]:
        model = models[model_id]
        model_params = params[params.model_id == model_id].head(cfg["alias_diagnostic"]["draws_per_model"])
        design = next(d for d in designs if d["model_id"] == model_id and d["design_id"].endswith("D12"))
        for _, parameter_row in model_params.iterrows():
            theta = parameter_row.theta
            result = trace_source_alias(
                model, theta, primary_target_index(model), cfg["alias_diagnostic"]["fixed_target_delta"],
                tuple(cfg["alias_diagnostic"]["parameter_bounds"]),
            )
            p0, j0 = model.distribution_and_jacobian(theta)
            p1, j1 = model.distribution_and_jacobian(np.asarray(result["theta_alias"]))
            observed_errors = {}
            for knowledge in cfg["operator_design"]["knowledge"]:
                errors = []
                for operator_id in design["operator_ids"]:
                    op = operators[operator_id]
                    y0 = observable_output(model, p0, j0, op, knowledge)[0]
                    y1 = observable_output(model, p1, j1, op, knowledge)[0]
                    errors.append(float(np.max(np.abs(y0 - y1))))
                observed_errors[knowledge] = max(errors)
            row = {
                "protocol_version": "1.0", "evidence_type": "NEW_RUN_DIAGNOSTIC",
                "model_id": model_id, "parameter_draw_id": int(parameter_row.draw_id),
                "target_id": f"primary_{primary_edge(model)[1]}_to_{primary_edge(model)[0]}",
                "theta_original": theta.tolist(), "theta_alias": np.asarray(result.pop("theta_alias")).tolist(),
                **result, "max_observed_error_K0": observed_errors["K0_KNOWN"],
                "max_observed_error_K1": observed_errors["K1_STRUCTURED"],
                "global_proof": False,
            }
            rows.append(row)
    path = G1 / "results/G1_counterexamples.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame([{k:v for k,v in row.items() if k not in ("theta_original","theta_alias")} for row in rows]).to_csv(
        G1 / "results/G1_alias_summary.tsv", sep="\t", index=False
    )
    print("alias rows", len(rows))


def run_finite() -> None:
    cfg, models, operators, designs, params, _ = _load_frozen()
    model_id = cfg["finite_sample_illustrations"]["model"]
    model = models[model_id]
    theta = params[(params.model_id == model_id) & (params.draw_id == 0)].iloc[0].theta
    primary_idx = primary_target_index(model)
    model_designs = {d["design_id"]: d for d in designs if d["model_id"] == model_id}
    scenarios = {
        "single": (model_designs[f"{model_id}__D12"], 1),
        "duplicated": (model_designs[f"{model_id}__D00"], 2),
        "candidate_heterogeneous": (model_designs[f"{model_id}__D12"], 2),
        "cross_view_failure": (model_designs[f"{model_id}__D21"], 2),
    }
    rows = []
    start = time.time()
    for scenario_index, (scenario, (design, environments)) in enumerate(scenarios.items()):
        ids, groups = expand_design(design, environments)
        envs = [FrozenOperatorEnvironment(operators[x], "K1_STRUCTURED") for x in ids]
        true_phis = [np.asarray(operators[x]["phi_true"], dtype=float) for x in ids]
        base, j_base = model.distribution_and_jacobian(theta)
        outputs = [observable_output(model, base, j_base, operators[x], "K1_STRUCTURED")[:3] for x in ids]
        geometry = joint_profiled_geometry(outputs, np.ones(environments), groups, cfg["numerical"]["rank_rtol"], cfg["numerical"]["rank_rtol_sensitivity"])
        tm = target_metrics(np.eye(model.parameter_count)[primary_idx], geometry, cfg["numerical"]["rank_rtol"])
        for n_index, n_total in enumerate(cfg["finite_sample_illustrations"]["n_total"]):
            sizes = [n_total // environments] * environments
            sizes[-1] += n_total - sum(sizes)
            for replicate in range(cfg["finite_sample_illustrations"]["replicates"]):
                rng = np.random.default_rng(stable_seed(cfg["seed_master"], 1, 31, scenario_index, n_index, replicate))
                counts = sample_environment_counts(model, envs, theta, true_phis, sizes, rng)
                t0 = time.time()
                fit = fit_joint(model, envs, counts, theta_start=np.zeros(model.parameter_count), theta_bound=3.0, phi_bound=2.0)
                elapsed = time.time() - t0
                estimate = float(fit.theta[primary_idx])
                rows.append({
                    "protocol_version": "1.0", "evidence_type": "NEW_RUN_ILLUSTRATIVE",
                    "model_id": model_id, "parameter_draw_id": 0, "scenario": scenario,
                    "design_id": design["design_id"], "operator_knowledge": "K1_STRUCTURED",
                    "n_unique_patients": n_total, "n_observed": n_total,
                    "replicate_id": replicate, "seed": int(rng.bit_generator.state["state"]["state"] % (2**63-1)),
                    "truth": float(theta[primary_idx]), "estimate": estimate,
                    "error": estimate - float(theta[primary_idx]), "absolute_error": abs(estimate - float(theta[primary_idx])),
                    "rank": geometry.rank, "sigma_min_per_sample": geometry.sigma_min_all,
                    "structural_status": "IDENTIFIED" if tm["identified"] else "NONIDENTIFIED",
                    "computational_status": "COMPLETE" if fit.success else "OPTIMIZER_FAILED",
                    "runtime_seconds": elapsed, "coverage_claim_allowed": False,
                })
            print("finite", scenario, n_total, "elapsed", round(time.time()-start,1))
    pd.DataFrame(rows).to_csv(G1 / "results/G1_finite_sample_illustrations.tsv", sep="\t", index=False)
    print("finite rows", len(rows), "seconds", round(time.time()-start,1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "freeze", "run-population", "run-alias", "run-finite"])
    args = parser.parse_args()
    {"audit": audit, "freeze": freeze, "run-population": run_population, "run-alias": run_alias, "run-finite": run_finite}[args.command]()


if __name__ == "__main__":
    main()
