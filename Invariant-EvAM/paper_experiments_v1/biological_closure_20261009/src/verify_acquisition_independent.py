"""Independent read-only engineering/science review of the frozen acquisition run.

No fit, action change, result regeneration, protocol mutation or publication.
The only writes are this review's JSON/TSV/text artifacts in review/acquisition.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import acquisition as acq

BASE = Path(__file__).resolve().parents[1]
EXPECTED = {
    "code": "6a243d444f34fe4cafd63b60cf6911d22aa9039f47398145cf4d470fb8f514ec",
    "model": "4e9f25c36de3b642769db9b6018a0e8c3077dcd6cef6af3afeae565cfa2c31e8",
    "actions": "e554e166a07699aa3e34345b88dc6483273da87840e522e8b8d7309a70ecd84f",
    "protocol": "93bbb65ae6a09dfdd242b33d3b0fab07124c196f25804aa247c768f38fc1bbca",
    "split": "7342aa0dc54aea63614003d1ccace2f5c1e9842a79e0506f8e6293b8be803f22",
}


def check(condition: bool, name: str) -> None:
    if not bool(condition):
        raise AssertionError(name)


def effective_information(plan: acq.Plan, region_count: int) -> set[tuple[int, str]]:
    """Abstract exposed information: core plus additional WES per region."""
    info = {(0, "CORE")}
    if plan.expand:
        info.add((0, "NONCORE_WES"))
    for index in range(1, 1 + plan.new_full + plan.new_narrow):
        check(index < region_count, "plan exceeds public region capacity")
        info.add((index, "CORE"))
        if index <= plan.new_full:
            info.add((index, "NONCORE_WES"))
    return info


def cost_and_dominance(protocol: dict) -> dict:
    n, f = protocol["acquisition"]["narrow_cost"], protocol["acquisition"]["full_cost"]
    cases, comparisons = 0, 0
    for regions in range(3, 12):
        case = acq.VisibleCase("fixture", "tumour", "LUAD", 0, "r0",
            tuple(f"r{i}" for i in range(regions)), 0, (0,) * 4, (0.,) * 4, (0.,) * 4)
        for budget, overhead in acq.cell_settings(protocol):
            plans = acq.feasible_plans(case, budget, overhead, n, f)
            check(len(plans) == len(set(p.name for p in plans)), "duplicate actual plan")
            for plan in plans:
                expected_cost = (f - n) * plan.expand + (f + overhead) * plan.new_full + (n + overhead) * plan.new_narrow
                check(expected_cost == plan.cost <= budget, "incorrect acquisition cost")
                check(plan.new_full + plan.new_narrow <= regions - 1, "capacity failure")
            for a, b in itertools.product(plans, repeat=2):
                expected = effective_information(a, regions) > effective_information(b, regions)
                check(acq.dominates(a, b) == expected, "dominance differs from actual observation inclusion")
                comparisons += 1
            frontier = acq.nondominated(plans)
            check(frontier, "empty feasible frontier")
            check(all(not any(acq.dominates(q, p) for q in plans) for p in frontier), "dominated random plan")
            cases += 1
    return dict(cost_capacity_cases=cases, information_inclusion_pairs=comparisons,
        acquisition_costs_match_protocol=True, random_uses_nondominated_distinct_actual_plans=True,
        total_cost_semantics="INITIAL_NARROW_COST_PLUS_INCREMENTAL_COST; UNUSED_BUDGET_REPORTED")


def source_control_flow() -> dict:
    tree = ast.parse(Path(acq.__file__).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    queue, visited = ["choose_frozen"], set()
    forbidden = {"open", "read_csv", "read_text", "load_scorer", "load_observations",
        "score_plan", "fit_population", "fit_snapshot", "fit_prior_and_kernel"}
    calls = set()
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call):
                called = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                calls.add(called)
                check(called not in forbidden, f"private/IO/fit call in selection: {called}")
                if called in functions:
                    queue.append(called)
    body = inspect.getsource(acq.run_confirmation)
    check(body.index('"ACTION_FREEZE.json"') < body.index("scorer = load_scorer"), "scorer opened before action freeze")
    allowed = {"patient_id", "tumour_id", "histology", "repeat", "initial_region",
        "region_order", "core_state", "positive_core_counts", "mean_vaf", "max_vaf"}
    check(set(acq.VisibleCase.__dataclass_fields__) == allowed, "unexpected policy interface field")
    return dict(selection_call_graph=sorted(visited), observed_call_names=sorted(calls),
        private_or_future_reader_in_call_graph=False,
        confirmation_action_freeze_precedes_scorer_open_in_source=True,
        policy_signature=str(inspect.signature(acq.choose_frozen)), visible_fields=sorted(allowed))


def hostile_hidden_changes(patient: acq.PatientObservation, case: acq.VisibleCase,
                           pattern: str) -> acq.PatientObservation:
    regions = dict(patient.regions)
    for rid in regions:
        if rid == case.initial_region:
            continue
        if pattern == "EMPTY_HIDDEN":
            regions[rid] = acq.RegionObservation(frozenset(), frozenset(), 0,
                (0,) * 4, (0.,) * 4, (0.,) * 4)
        else:
            variants = frozenset(f"arbitrary-hidden-{rid}-{j}" for j in range(31))
            regions[rid] = acq.RegionObservation(variants, variants, 15,
                (101, 203, 307, 409), (.999, .123, .456, .789), (1.,) * 4)
    return acq.PatientObservation(patient.patient_id, patient.tumour_id, patient.histology, regions)


def interface_noninterference(observations: dict, models: dict, protocol: dict,
                             actions: pd.DataFrame) -> dict:
    seed = protocol["master_seed"]
    # A representative real case for every observed core state/region-count
    # combination in each population, plus all five initial-region rotations.
    representatives = {}
    for row in actions.itertuples(index=False):
        patient = observations[row.patient_id]
        case = patient.visible(int(row.repeat), seed)
        key = (row.population, case.core_state, len(case.region_order), case.repeat)
        representatives.setdefault(key, (row.population, patient, case))
    compared = 0
    reference_variations = [dict(branches={}, reference_core_state=0, future_label="NO_RELAPSE"),
        dict(branches={"invented": ["x", "y", "z"]}, reference_core_state=15,
            future_label="RELAPSE_WITH_ARBITRARY_BRANCH")]
    # Block accidentally introduced private-reader calls during policy tests.
    blocked_names = ("load_scorer", "load_observations", "score_plan", "fit_snapshot", "fit_prior_and_kernel")
    saved = {n: getattr(acq, n) for n in blocked_names}
    def blocked(*args, **kwargs):
        raise AssertionError("private reader or fit invoked by policy")
    for name in blocked_names:
        setattr(acq, name, blocked)
    try:
        for population, patient, case in representatives.values():
            model = models[population]
            for budget, overhead in acq.cell_settings(protocol):
                for method in acq.METHODS:
                    original, value = acq.choose_frozen(case, budget, overhead, method, model, seed)
                    for pattern in ("EMPTY_HIDDEN", "ALL_HIDDEN_MUTATED"):
                        changed = hostile_hidden_changes(patient, case, pattern)
                        # Changing all hidden-region genotype/read fields
                        # preserves public capacity and the exposed baseline.
                        changed_case = changed.visible(case.repeat, seed)
                        check(changed_case == case, "hidden changes altered visible interface")
                        for arbitrary_scorer in reference_variations:
                            # No branch/reference/future field is an argument to
                            # choose_frozen. Keep these poisoned fields attached
                            # only to a private holder to exercise the boundary.
                            private_holder = {"patient": changed, "scorer": arbitrary_scorer}
                            result, altered_value = acq.choose_frozen(changed_case,
                                budget, overhead, method, model, seed)
                            check(result == original, "hidden/scorer changes changed frozen action")
                            if np.isfinite(value):
                                check(altered_value == value, "hidden/scorer changes changed utility estimate")
                            check(private_holder["scorer"] is arbitrary_scorer, "fixture mutated scorer")
                            compared += 1
    finally:
        for name, fn in saved.items():
            setattr(acq, name, fn)
    return dict(real_visible_interface_strata=len(representatives), noninterference_comparisons=compared,
        hidden_patterns=["ALL_HIDDEN_REMOVED", "ALL_HIDDEN_CORE_STATES_AND_READS_ARBITRARILY_CHANGED"],
        private_reference_patterns=reference_variations, same_frozen_action=True,
        reader_guards=sorted(blocked_names), proof_scope="CODE_CALL_GRAPH_AND_INTERFACE_NONINTERFERENCE_TESTS; NOT_AN_OS_SECURITY_SANDBOX")


def core_registry_check(audit_dir: Path, protocol: dict) -> dict:
    original = json.loads((acq.PROJECT / "paper_experiments_v1/g5/configs/DATA_FREEZE.json").read_text())
    g4 = json.loads((acq.PROJECT / "paper_experiments_v1/g4/configs/DATA_FREEZE.json").read_text())
    check(tuple(original["modules"]) == acq.GENES == tuple(protocol["core_events"]), "core event registry order mismatch")
    check(tuple(g4["modules"]["LUAD"]) == acq.GENES, "original LUAD registry mismatch")
    check(acq.state_number(np.array([1, 0, 0, 0])) == 8, "MSB event-state encoding mismatch")
    x = pd.read_csv(audit_dir / "region_core_variants.tsv.gz", sep="\t", low_memory=False)
    expected_positive = x["Is.present.region"].astype(str).str.lower().eq("true") & x.depth.ge(20) & x.var_count.ge(3)
    check(np.array_equal(expected_positive, acq.as_bool(x.positive)), "positive does not match actual call/depth/alt rule")
    x = x[expected_positive & acq.as_bool(x.primary)]
    label = x["exonic.func"].fillna("").str.lower()
    common = label.str.contains("nonsynonymous|stopgain|stoploss|frameshift|startloss", regex=True)
    actual = common | x.func.eq("splicing")
    broad = common | x.func.fillna("").str.contains("splic", case=False)
    differences = int((actual != broad).sum())
    check(differences == 0, "current eligible core splice rows lost by exact annotation match")
    return dict(core_events=list(acq.GENES), encoding="MSB_FIRST_TP53_BIT8_KRAS_BIT4_STK11_BIT2_KEAP1_BIT1",
        original_event_definition=g4["event_definition"], current_event_definition=protocol["core_state"],
        annotation_translation_matches_current_positive_core_records=True,
        splice_annotation_current_data_mismatches=differences,
        positive_call_rule="AUTHOR_PRESENT_AND_DEPTH_AT_LEAST20_AND_ALT_AT_LEAST3",
        observed_auxiliary_features="ALL_ACTUALLY_REPORTED_CORE_GENE_VARIANT_COUNTS_AND_VAF; NOT_ALL_AUXILIARY_VARIANTS_ARE_NONSYNONYMOUS_EVENTS",
        intentional_contract_extension="ONE_REPORTED_SAMPLE_EVENT_PATTERN_TO_FOUR_GENE_PRIMARY_REGION_UNION_PROXY_WITH_LEARNED_DETECTION_KERNEL",
        no_CNA_or_SV_core_events=True, verified_negative_claim=False)


def verify_saved_rows(results: Path, observations: dict, models: dict, protocol: dict) -> dict:
    raw = pd.read_csv(results / "confirmation_raw.tsv.gz", sep="\t")
    patient = pd.read_csv(results / "confirmation_patient_summary.tsv", sep="\t")
    actions = pd.read_csv(results / "confirmation_actions_frozen.tsv", sep="\t")
    keys = ["population", "patient_id", "tumour_id", "split", "histology", "budget", "overhead", "method"]
    metrics = ["branch_recall", "hidden_driver_recall", "cost", "recovered_branch", "all_branch",
               "four_gene_log_loss", "four_gene_brier"]
    averaged = raw.groupby(keys, as_index=False)[metrics].mean()
    joined = averaged.merge(patient, on=keys, suffixes=("_computed", "_saved"), validate="one_to_one")
    for metric in metrics:
        check(np.allclose(joined[f"{metric}_computed"], joined[f"{metric}_saved"], rtol=0, atol=1e-12), f"patient-average mismatch {metric}")
    repeats = raw.groupby(keys).size()
    check(repeats.eq(5).all(), "patient rotations not retained/averaged")
    check(raw.cost.le(raw.budget).all(), "saved budget violation")
    check((raw.total_cost == raw.cost + protocol["acquisition"]["narrow_cost"]).all(), "total cost missing base")
    check((raw.unused_budget == raw.budget - raw.cost).all(), "unused budget mismatch")
    check(not actions.duplicated(["population", "patient_id", "repeat", "budget", "overhead", "method"]).any(), "duplicate action")
    reproduced = 0
    cases = {}
    for row in actions.itertuples(index=False):
        k = (row.patient_id, int(row.repeat))
        cases.setdefault(k, observations[row.patient_id].visible(int(row.repeat), protocol["master_seed"]))
        plan, _ = acq.choose_frozen(cases[k], int(row.budget), int(row.overhead),
            row.method, models[row.population], protocol["master_seed"])
        check(plan.name == row.plan and plan.cost == row.cost, "saved action does not reproduce")
        reproduced += 1
    zero = raw.groupby(["population", "patient_id"]).all_branch.first().eq(0).groupby(level=0).sum().to_dict()
    raw_costs = raw[(raw.budget == 10) & (raw.overhead == 2)].groupby(["population", "method"]).cost.mean().to_dict()
    return dict(confirmation_actions_exactly_reproduced=reproduced,
        patient_averages_exactly_reproduced=len(joined), perturbations_per_patient=5,
        independent_patient_n=patient.groupby("population").patient_id.nunique().to_dict(),
        zero_reference_branch_patients_retained=zero,
        primary_actual_incremental_mean_cost={f"{p}:{m}": float(v) for (p, m), v in raw_costs.items()},
        repeated_regions_not_independent_patients=True,
        populations_overlap="LUAD_IS_A_SUBSET_OF_ALL_NSCLC; DO_NOT_ADD_N")


def future_capacity_check(base: Path, results: Path) -> dict:
    path = base / "data/metastasis_audit/future_patient_reference_capacity.tsv"
    if not path.exists():
        return dict(status="NOT_AVAILABLE_AT_REVIEW")
    capacity = pd.read_csv(path, sep="\t", dtype=str)
    primary = pd.read_csv(results / "confirmation_patient_summary.tsv", sep="\t",
        usecols=["population", "patient_id"])
    out = dict(status="ACTUAL_FUTURE_MAPPING_METADATA_AUDITED_NOT_A_NEW_OUTCOME_RUN",
        path=str(path), sha256=acq.digest(path), future_reference_patients=int(capacity.patient_id.nunique()),
        confirmation_patients=int(capacity.loc[capacity.split.eq("confirmation"), "patient_id"].nunique()))
    for population in ("LUAD", "ALL_NSCLC"):
        ids = set(primary.loc[primary.population.eq(population), "patient_id"])
        q = capacity[capacity.patient_id.isin(ids)]
        out[population] = dict(confirmation_acquisition_patients_with_future_records=int(q.patient_id.nunique()),
            supported_nontrunk_reference_patients=int(q.loc[acq.as_bool(q.nontrunk_multimarker_reference_available), "patient_id"].nunique()),
            capacity_status=q.reference_capacity_status.value_counts().to_dict())
    out["interpretation"] = "SMALL_PATIENT_LEVEL_CASE_BRIDGE; MISSING_FUTURE_REFERENCE_IS_UNKNOWN_NOT_NEGATIVE; NO_ADDED_INDEPENDENT_COHORT_OR_CLINICAL_BENEFIT_CLAIM"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output-dir", type=Path, default=BASE / "review/acquisition")
    args = parser.parse_args()
    results, audit_dir = args.base / "results/acquisition", args.base / "data/native_audit"
    protocol_path, split_path = args.base / "configs/PROTOCOL.json", args.base / "configs/PATIENT_SPLIT.tsv"
    protocol = json.loads(protocol_path.read_text())
    model_path, action_path = results / "FROZEN_MODELS.json", results / "confirmation_actions_frozen.tsv"
    frozen = json.loads(model_path.read_text())
    mf = json.loads((results / "MODEL_FREEZE.json").read_text())
    af = json.loads((results / "ACTION_FREEZE.json").read_text())
    implementation = json.loads((results / "IMPLEMENTATION_FREEZE.json").read_text())
    for kind, path in (("code", Path(acq.__file__)), ("model", model_path),
            ("actions", action_path), ("protocol", protocol_path), ("split", split_path)):
        check(acq.digest(path) == EXPECTED[kind], f"historically observed frozen {kind} hash changed")
    check(mf["model_sha256"] == af["model_sha256"] == EXPECTED["model"], "model/action freeze disagreement")
    check(mf["code_sha256"] == af["code_sha256"] == implementation["code_sha256"] == EXPECTED["code"], "implementation overwritten")
    check(af["actions_sha256"] == EXPECTED["actions"], "actions overwritten")
    for path, meta in mf["source_files"].items():
        check(acq.digest(Path(path)) == meta["sha256"], f"source changed since model freeze: {path}")
    models = frozen["models"]
    split = pd.read_csv(split_path, sep="\t", dtype=str).set_index("patient_id")["split"].to_dict()
    for model in models.values():
        check(all(split[p] == "development" for p in model["prior"]["patient_ids"]), "fit used confirmation patients")
    registry = core_registry_check(audit_dir, protocol)
    costs = cost_and_dominance(protocol)
    flow = source_control_flow()
    observations = acq.load_observations(audit_dir)
    actions = pd.read_csv(action_path, sep="\t")
    interfaces = interface_noninterference(observations, models, protocol, actions)
    rows = verify_saved_rows(results, observations, models, protocol)
    future = future_capacity_check(args.base, results)
    assessment = json.loads((results / "ASSESSMENT.json").read_text())
    check(assessment["g5_diagnostic_value_supported"] is False, "tiny positive point estimate relabeled G5 success")
    check(assessment["CTMC_specific_improvement_supported"] is False, "CTMC-specific success falsely asserted")
    issues = [
        dict(id="PRIMARY_DIAGNOSTIC_GAIN_UNCONFIRMED", severity="MATERIAL_SCIENTIFIC_LIMITATION",
             finding="LUAD CTMC mean gain0.003245 vsBEST_FIXED; confidence interval includes0 and Holm3p=1. Tiny positive mean is not supported diagnostic superiority.", disposition="NO_G5_DIAGNOSTIC_SUCCESS_CLAIM"),
        dict(id="CTMC_EMPIRICAL_ACTIONS_EQUAL", severity="MATERIAL_SCOPE_LIMITATION",
             finding="The primary CTMC and empirical-prior bridges produce the same actions/recovery. No CTMC-specific acquisition improvement.", disposition="CLASSICAL_TOOLS_ALLOWED; DO_NOT_CLAIM_UNIQUE_CTMCDIAGNOSTIC_ADVANTAGE"),
        dict(id="REFERENCE_TARGET_EXTENSION", severity="MATERIAL_SCOPE_LIMITATION",
             finding="Original population snapshot target is unchanged. Patient OR-state, regional kernel and ridge branch utility are a separately defined biological bridge, not theta/ancestry truth.", disposition="EXPLICIT_NEW_BRIDGE_ENDPOINT_ONLY"),
        dict(id="ALLNSCLC_PRIOR_BOUNDARY", severity="MATERIAL_PREDICTIVE_MODEL_LIMITATION",
             finding="The pooled ALL_NSCLC surrogate optimizer succeeds but interaction index4 touches bound3. Primary LUAD surrogate has no active bounds.", disposition="PREDICTED_STATE_SURROGATE_ONLY; NO_PARAMETER_SIGN_OR_HAZARD_INTERPRETATION"),
        dict(id="HYPOTHETICAL_COSTS_AND_RANDOM_UNSPENT", severity="MATERIAL_INTERPRETATION_LIMITATION",
             finding="Costs2/8 and tissue overhead are hypothetical. Primary RANDOM spends9.33 LUADunits on average vs10 for fixed/bridges; it shares a budget ceiling but not exact realized spending.", disposition="BEST_FIXED_IS_THE_MATCHED_SPENDING_STRONG_CONTROL; NO_MONETARY_OR_RANDOM_ONLY_SUCCESS_CLAIM"),
        dict(id="REFERENCE_BRANCHES_NOT_NEW_TREE_RECONSTRUCTION", severity="MATERIAL_SCOPE_LIMITATION",
             finding="Scoring matches at least3 measured branch-owned variants to author reconstructed nontrunk clusters. It does not independently reconstruct or verify cancer ancestry.", disposition="REFERENCE_CLUSTER_MARKER_RECOVERY_ONLY"),
        dict(id="GENE_NEGATIVE_PROXY", severity="MATERIAL_SCOPE_LIMITATION",
             finding="No-report union states and finite-depth variant negatives are reference detection proxies, not independently verified gene-callable wildtype or never-acquired events.", disposition="NO_NATURAL_THETA_COVERAGE_CLAIM"),
        dict(id="SPLICE_LABEL_PORTABILITY", severity="NONMATERIAL_CURRENT_DATA_ENGINEERING_NOTE",
             finding="acquisition.py accepts exact func=splicing; future compound splicing annotations would require an explicit adapter amendment. Current qualifying core data have0 discrepancies.", disposition="NO_CURRENT_RESULT_CHANGE; NO_REFIT"),
        dict(id="MC_STATISTICS_SEED_DIFFERENCE", severity="NONMATERIAL_NUMERICAL_NOTE",
             finding="Independent quick review and root PAIRED_CONTRASTS use different bootstrap/sign-flip MonteCarlo draws. Primary mean and unsupported conclusion agree; primary quick signflip uses exact enumeration for18 nonzero patients.", disposition="ROOT_ASSESSMENT_IS_CANONICAL; DO_NOT_RESELECT_SEEDS"),
    ]
    issues.append(dict(id="FUTURE_DANGER_BRANCH_SMALL_N", severity="MATERIAL_EXTERNAL_MEANING_LIMITATION",
        finding=f"Actual future mapping available in primary LUAD acquisition confirmation for{future.get('LUAD', {}).get('confirmation_acquisition_patients_with_future_records', 'UNKNOWN')}patients; only{future.get('LUAD', {}).get('supported_nontrunk_reference_patients', 'UNKNOWN')}have supported nontrunk reference. Five regional starts or many markers cannot increase this patient N.",
        disposition="CASE_LEVEL_FUTURE_BRANCH_BRIDGE_ONLY; RETAIN_UNKNOWN_REFERENCE_STATUS; NO_STRONG_POPULATION_CLINICAL_CONCLUSION"))
    report = dict(status="PASS_ENGINEERING_WITH_SCIENTIFIC_LIMITATIONS", no_new_fit=True,
        protocol_or_results_modified=False, frozen_hashes_unchanged=EXPECTED,
        core_registry=registry, costs=costs, policy_source=flow, interface_tests=interfaces,
        saved_row_tests=rows, future_reference_capacity=future, g5_diagnostic_value_supported=False,
        CTMC_specific_improvement_supported=False, material_issues=issues,
        limitations_of_review="Hash anchors/control flow/interface tests verify this executed implementation. They are not cryptographic proof of arbitrary historical ordering or an OS security isolation guarantee.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "INDEPENDENT_REVIEW.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(issues).to_csv(args.output_dir / "ISSUES.tsv", sep="\t", index=False)
    (args.output_dir / "REVIEW_CN.txt").write_text(
        "独立采集审查通过工程检查：核心基因及MSB编码一致；费用/区域容量/信息支配关系一致；"
        "患者内五次扰动确实先平均；冻结动作可全部精确复算；任意改变隐藏区域基因型、读数、分支参考与未来标签不会改变策略输出。\n"
        "科学结果未证明G5诊断价值：主LUAD增益仅0.32个百分点，区间跨零，Holm校正p=1；"
        "CTMC与经验桥接主行动相同，不能包装成CTMC独有优势。\n"
        "原人群CTMC目标没有被真实树验证。新增区域检测核与患者效用层属于明确的预测桥接；"
        "分支是作者簇参考的多标记找回，费用是假设单位，阴性是记录代理。\n"
        "ALL_NSCLC先验有一个边界参数；随机策略平均花费略低于固定和诊断策略，因此主要说服力必须来自最佳固定对照。\n"
        "该审查不重新拟合，不改变方法、患者、参数、随机种子或正式发布文件。\n")
    print(json.dumps(dict(status=report["status"], interface_comparisons=interfaces["noninterference_comparisons"],
        reproduced_actions=rows["confirmation_actions_exactly_reproduced"], independent_patient_n=rows["independent_patient_n"],
        g5_diagnostic_value_supported=False, report=str(args.output_dir / "INDEPENDENT_REVIEW.json")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
