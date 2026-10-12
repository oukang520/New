#!/usr/bin/env python3
"""Frozen surgical marker selection and retrospective public ctDNA count replay."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ctdna" / "processed"
NATIVE = ROOT / "data" / "native_audit"
OUT = ROOT / "results" / "ctdna"
CONFIG = ROOT / "configs" / "CTDNA_ASSAY_FREEZE.json"
POLICIES = ["CLONAL_ONLY", "BRANCH_DIVERSE", "RANDOM_MARKERS", "INFORMATION_ADAPTIVE"]
BUDGETS = [20, 50, 100]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def seed(tag):
    return int(hashlib.sha256(tag.encode()).hexdigest()[:16], 16) ^ 20261009


def read(name):
    return pd.read_csv(DATA / (name + ".tsv.gz"), sep="\t", low_memory=False)


def keys(frame):
    chrom = frame.chromosome.astype(str).str.replace("chr", "", regex=False).str.replace(r"\.0$", "", regex=True)
    return (frame.PublicationID.astype(str) + ":" + chrom + ":" + frame.position.astype(int).astype(str)
            + ":" + frame.reference.astype(str) + ":" + frame.alternate.astype(str))


def intrinsic_pass(values, full_qc=False):
    def check(value):
        if pd.isna(value) or str(value).strip() == "":
            return True
        names = {x.strip() for x in str(value).split(",") if x.strip()}
        return not names if full_qc else names <= {"dao_imbalance"}
    return values.map(check)


def surgery_rank(value):
    observations = []
    positive = False
    if pd.isna(value):
        return np.nan, False
    for segment in str(value).split(";"):
        try:
            region, counts = segment.split(":", 1)
            if not region.startswith("SU_T"):
                continue
            alt, ref = map(float, counts.split("/"))
            total = alt + ref
            if total >= 20:
                observations.append(alt / total)
                positive |= alt >= 3
        except (ValueError, ZeroDivisionError):
            continue
    return (float(np.mean(observations)) if observations else np.nan), bool(positive)


def build_library(plasma):
    cols = ["PublicationID", "tumour_id", "chromosome", "position", "reference", "alternate",
            "var_ref_counts", "PyCloneClonal_SC", "PyCloneCluster_SC", "is_tree_clone"]
    library = plasma[cols].drop_duplicates(["PublicationID", "tumour_id", "chromosome", "position", "reference", "alternate"]).copy()
    library["marker_key"] = keys(library)
    library["variant_key"] = library["marker_key"].str.replace(r"^[^:]+:", "", regex=True)
    library["variant_key"] = library.tumour_id.astype(str) + ":" + library.variant_key
    # Same coordinate in two synchronous tumours is one assayed marker. Keep only
    # the root protocol's selected primary tumour when an ownership choice exists.
    chosen = pd.read_csv(NATIVE / "patient_primary_tumour.tsv", sep="\t")
    selected = chosen.set_index("patient_id").tumour_id.to_dict()
    library["selected_primary"] = [selected.get(p, t) == t for p, t in zip(library.PublicationID, library.tumour_id)]
    library = library.sort_values(["marker_key", "selected_primary", "tumour_id"], ascending=[True, False, True]).drop_duplicates("marker_key")
    ranks = library.var_ref_counts.map(surgery_rank)
    library["surgery_vaf"] = ranks.map(lambda x: x[0])
    library["valid_surgery"] = ranks.map(lambda x: x[1]) & library.selected_primary
    owners = pd.read_csv(NATIVE / "branch_owned_markers.tsv.gz", sep="\t", usecols=["tumour_id", "variant_key", "cluster"])
    owners = owners.drop_duplicates()
    assert not owners.duplicated(["tumour_id", "variant_key"]).any(), "ambiguous primary-only marker ownership"
    refs = pd.read_csv(NATIVE / "branch_reference.tsv", sep="\t")
    native_known = set(refs.patient_id)
    refs = refs[["tumour_id", "cluster", "is_trunk", "eligible_multimarker_reference", "reference_markers"]]
    owners = owners.merge(refs, on=["tumour_id", "cluster"], how="left", validate="many_to_one")
    library = library.merge(owners, on=["tumour_id", "variant_key"], how="left", validate="many_to_one")
    library["target_group"] = ""
    library["branch_group"] = ""
    library["reference_scope"] = "UNKNOWN_TREE"
    for p, group in library.groupby("PublicationID", sort=False):
        if p in native_known:
            library.loc[group.index, "reference_scope"] = "NATIVE_PRIMARY_TREE"
            for i, row in group.iterrows():
                if (row.valid_surgery and pd.notna(row.cluster) and row.reference_markers >= 3
                        and (row.is_trunk == True or row.eligible_multimarker_reference == True)):
                    label = str(row.tumour_id) + ":" + str(int(row.cluster))
                    library.at[i, "target_group"] = label
                    if row.is_trunk != True:
                        library.at[i, "branch_group"] = label
        elif p not in selected:
            # Exactly the nine ctDNA participants outside the native 421 cohort.
            library.loc[group.index, "reference_scope"] = "AUTHOR_PRIMARY_PROXY_OUTSIDE_NATIVE"
            valid = group[(group.is_tree_clone == True) & group.valid_surgery & group.PyCloneCluster_SC.notna()]
            for cluster, nodes in valid.groupby("PyCloneCluster_SC"):
                if len(nodes) < 3:
                    continue
                label = "PROXY:" + p + ":" + str(cluster)
                library.loc[nodes.index, "target_group"] = label
                if not (nodes.PyCloneClonal_SC == "C").all():
                    library.loc[nodes.index, "branch_group"] = label
    # Reference membership is entirely established before plasma outcome scoring.
    return library.sort_values("marker_key").reset_index(drop=True), chosen


def normal_parameters(normal):
    normal = normal.copy()
    normal["source_group"] = np.where(normal.tracerx_id.astype(str) == "GIAB", "GIAB", normal.PublicationID)
    normal["split"] = normal.source_group.map(lambda x: "development" if int(hashlib.sha256(("Invariant-EvAM-bio-assay-v1|" + str(x)).encode()).hexdigest()[:16], 16) / 2**64 < 0.6 else "confirmation")
    usable = normal[(normal.split == "development") & intrinsic_pass(normal["failed filters"]) & (normal.ddp > 0)]
    assert len(usable), "no development normal rows"
    error = usable[["d_group_error_rate", "tnc_error_rate_ci_upper_bound", "mnp_error_rate"]].max(axis=1)
    parameters = {"normal_development_source_groups": sorted(usable.source_group.unique()),
                  "normal_confirmation_source_groups": sorted(normal.loc[normal.split == "confirmation", "source_group"].unique()),
                  "depth": float(usable.ddp.median()), "error_rate": max(1e-9, float(error.median())),
                  "ctdna_fraction_grid": [0.0001, 0.0005, 0.001],
                  "parameters_use": "development normal depth/background only; marker P thresholds fixed0.01/K, not clinical-calibrated"}
    return normal, parameters


def branch_diverse(frame, k):
    rank = frame.sort_values(["surgery_vaf", "marker_key"], ascending=[False, True])
    clonal = rank[rank.PyCloneClonal_SC == "C"]
    selected = list(clonal.marker_key.head(k // 2))
    used_clonal = set(selected)
    branch_pool = rank[(rank.branch_group != "") & ~rank.marker_key.isin(used_clonal)]
    groups = {label: list(rows.marker_key) for label, rows in branch_pool.groupby("branch_group", sort=True)}
    branch_quota = k - k // 2
    branch = []
    # Give each clone two markers before subsequent single-marker rounds.
    for label in sorted(groups):
        for _ in range(2):
            if groups[label] and len(branch) < branch_quota:
                branch.append(groups[label].pop(0))
    while len(branch) < branch_quota and any(groups.values()):
        for label in sorted(groups):
            if groups[label] and len(branch) < branch_quota:
                branch.append(groups[label].pop(0))
    selected += branch
    used = set(selected)
    selected += [x for x in rank.marker_key if x not in used][:max(0, k - len(selected))]
    return selected[:k]


def information_adaptive(frame, k, parameters):
    ranked = frame.sort_values(["surgery_vaf", "marker_key"], ascending=[False, True])
    groups = [(name, rows) for name, rows in ranked[ranked.target_group != ""].groupby("target_group", sort=True)]
    nominal_k = k
    k = min(k, len(frame))
    if not k:
        return []
    depth, error = parameters["depth"], parameters["error_rate"]
    threshold = max(2, int(stats.poisson.isf(0.01 / max(nominal_k, 1), error * depth)) + 1)
    dp = np.full(k + 1, -np.inf); dp[0] = 0.0
    histories = []
    group_lists = []
    for name, rows in groups:
        nodes = list(rows.marker_key)
        vaf = rows.surgery_vaf.to_numpy()
        cap = min(k, len(nodes))
        utility = np.zeros(cap + 1)
        for fraction in parameters["ctdna_fraction_grid"]:
            probability = stats.poisson.sf(threshold - 1, depth * (error + fraction * np.minimum(1, 2 * vaf)))
            p0, p1 = 1.0, 0.0
            for count in range(1, cap + 1):
                q = probability[count - 1]
                p1, p0 = p1 * (1 - q) + p0 * q, p0 * (1 - q)
                utility[count] += (1 - p0 - p1) / 3.0
        new = np.full(k + 1, -np.inf)
        take = np.zeros(k + 1, dtype=int)
        for count in range(cap + 1):
            candidate = dp[:k + 1 - count] + utility[count]
            improve = candidate > new[count:] + 1e-14
            new[count:][improve] = candidate[improve]
            take[count:][improve] = count
        histories.append(take); group_lists.append(nodes); dp = new
    if not groups:
        return list(ranked.marker_key.head(k))
    # Spending fewer markers on known groups can leave ungrouped capacity unused.
    total = int(np.nanargmax(dp))
    selected = []
    for nodes, take in zip(group_lists[::-1], histories[::-1]):
        count = int(take[total]); selected.extend(nodes[:count]); total -= count
    used = set(selected)
    selected += [x for x in ranked.marker_key if x not in used][:max(0, k - len(selected))]
    return selected[:k]


def select_all(library, parameters, clinical):
    choices = {}
    rows = []
    for p in clinical.PublicationID:
        group = library[(library.PublicationID == p) & library.valid_surgery]
        ranked = group.sort_values(["surgery_vaf", "marker_key"], ascending=[False, True])
        rng = np.random.default_rng(seed("Invariant-EvAM-ctdna-random-v1|" + p))
        random = rng.permutation(group.sort_values("marker_key").marker_key.to_numpy()).tolist()
        for k in BUDGETS:
            plans = {"CLONAL_ONLY": list(ranked.loc[ranked.PyCloneClonal_SC == "C", "marker_key"].head(k)),
                     "BRANCH_DIVERSE": branch_diverse(group, k), "RANDOM_MARKERS": random[:k],
                     "INFORMATION_ADAPTIVE": information_adaptive(group, k, parameters)}
            for method, plan in plans.items():
                assert len(plan) == len(set(plan)) and len(plan) <= k
                choices[(p, method, k)] = frozenset(plan)
                for rank, marker in enumerate(plan, 1):
                    rows.append({"patient_id": p, "method": method, "nominal_K": k, "rank": rank, "marker_key": marker})
        choices[(p, "WHOLE_PSP_REFERENCE", 0)] = frozenset(library.loc[library.PublicationID == p, "marker_key"])
    return choices, pd.DataFrame(rows)


def call(rows, selected, k, full_qc=False, normal=False):
    use = rows[rows.marker_key.isin(selected)]
    quality = intrinsic_pass(use["failed filters" if normal else "failed.filters"], full_qc)
    pos = use[(use.variant_poisson_test <= 0.01 / max(1, k)) & (use.dao >= 2) & (use.ddp > 0) & quality]
    branch = collections.Counter(pos.loc[pos.branch_group != "", "branch_group"])
    detected = {group for group, count in branch.items() if count >= 2}
    return len(pos) >= 2, detected, len(pos), len(use)


def paired_inference(values, tag, replicates=2000):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "difference": None, "ci_lower": None, "ci_upper": None, "p_value": None}
    rng = np.random.default_rng(seed("ctdna-inference|" + tag))
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(axis=1)
    nonzero = values[np.abs(values) > 1e-14]
    if not len(nonzero):
        p = 1.0
    elif len(nonzero) <= 18:
        bits = np.arange(2 ** len(nonzero), dtype=np.uint64)[:, None]
        signs = 2 * ((bits >> np.arange(len(nonzero), dtype=np.uint64)) & 1).astype(float) - 1
        p = float(np.mean(np.abs(signs @ nonzero) >= abs(nonzero.sum()) - 1e-12))
    else:
        signs = rng.choice([-1.0, 1.0], size=(20000, len(nonzero)))
        p = float((1 + np.sum(np.abs(signs @ nonzero) >= abs(nonzero.sum()) - 1e-12)) / 20001)
    return {"n": len(values), "difference": float(values.mean()), "ci_lower": float(np.quantile(means, .025)),
            "ci_upper": float(np.quantile(means, .975)), "p_value": p}


def replay(plasma, normal, library, choices, clinical, splits, chosen):
    annotation = library[["marker_key", "branch_group", "target_group", "reference_scope"]]
    assert not annotation.marker_key.duplicated().any()
    plasma = plasma.copy(); plasma["marker_key"] = keys(plasma)
    plasma = plasma.merge(annotation, on="marker_key", how="left", validate="many_to_one")
    plasma = plasma[plasma.days_post_surgery > 0]
    normal = normal.copy(); normal["marker_key"] = keys(normal)
    normal = normal.merge(annotation, on="marker_key", how="left", validate="many_to_one")
    normal[["branch_group", "target_group"]] = normal[["branch_group", "target_group"]].fillna("")
    sample_rows, clone_rows, patient_rows, normal_rows = [], [], [], []
    scopes = library.groupby("PublicationID").reference_scope.first().to_dict()
    native_eligible = chosen.set_index("patient_id").to_dict("index")
    patient_groups = dict(tuple(plasma.groupby("PublicationID")))
    for p in clinical.PublicationID:
        cl = clinical[clinical.PublicationID == p].iloc[0]
        samples = patient_groups.get(p, plasma.iloc[:0])
        scope = scopes.get(p, "UNKNOWN_TREE")
        primary_pop = native_eligible.get(p, {}).get("histology_3") == "LUAD" and native_eligible.get(p, {}).get("primary_regions", 0) >= 3
        recurrence = float(cl.Time_DFS_from_surgery) if cl.Event_status in {"Intrathoracic", "Extrathoracic", "Intra & Extra"} and pd.notna(cl.Time_DFS_from_surgery) else np.nan
        by_sample = list(samples.groupby("tracerx_id"))
        for scenario, full in [("INTRINSIC_QC", False), ("FULL_QC_SENSITIVITY", True)]:
            reference_union = set(); ref_per_sample = {}; ref_mrd_days = []
            whole = choices[(p, "WHOLE_PSP_REFERENCE", 0)]
            for sid, records in by_sample:
                mrd, detected, npos, nused = call(records, whole, len(whole), full)
                ref_per_sample[sid] = detected; reference_union.update(detected)
                if mrd: ref_mrd_days.append(float(records.days_post_surgery.iloc[0]))
                sample_rows.append({"patient_id": p, "sample_id": sid, "days_post_surgery": records.days_post_surgery.iloc[0], "split": splits[p], "scenario": scenario, "method": "WHOLE_PSP_REFERENCE", "nominal_K": 0, "selected_markers": len(whole), "recorded_markers": nused, "positive_markers": npos, "panel_MRD": mrd, "nontrunk_detected": len(detected), "reference_detected": len(detected), "sample_reference_recall": 1.0 if detected else 0.0})
                for branch in detected:
                    clone_rows.append({"patient_id": p, "sample_id": sid, "scenario": scenario, "method": "WHOLE_PSP_REFERENCE", "nominal_K": 0, "branch": branch, "confirmed_by_whole_PSP": True})
            for k in BUDGETS:
                for method in POLICIES:
                    selected = choices[(p, method, k)]; found = set(); mrddays = []; sample_recalls = []; discordant = set()
                    for sid, records in by_sample:
                        mrd, detected, npos, nused = call(records, selected, k, full)
                        day = float(records.days_post_surgery.iloc[0])
                        reference = ref_per_sample[sid]
                        recall = len(detected & reference) / len(reference) if reference else 0.0
                        assert 0 <= recall <= 1
                        sample_recalls.append(recall); found.update(detected); discordant.update(detected - reference)
                        if mrd: mrddays.append(day)
                        sample_rows.append({"patient_id": p, "sample_id": sid, "days_post_surgery": day, "split": splits[p], "scenario": scenario, "method": method, "nominal_K": k, "selected_markers": len(selected), "recorded_markers": nused, "positive_markers": npos, "panel_MRD": mrd, "nontrunk_detected": len(detected), "reference_detected": len(reference), "sample_reference_recall": recall})
                        for branch in detected:
                            clone_rows.append({"patient_id": p, "sample_id": sid, "scenario": scenario, "method": method, "nominal_K": k, "branch": branch, "confirmed_by_whole_PSP": branch in reference})
                    recall = len(found & reference_union) / len(reference_union) if reference_union else 0.0
                    if scope == "UNKNOWN_TREE": recall = np.nan
                    predates = [x for x in mrddays if np.isfinite(recurrence) and x <= recurrence]
                    patient_rows.append({"patient_id": p, "split": splits[p], "reference_scope": scope, "primary_LUAD_GE3": primary_pop,
                        "scenario": scenario, "method": method, "nominal_K": k, "selected_markers": len(selected), "capacity_shortfall": len(selected) < k,
                        "postoperative_samples": len(by_sample), "whole_reference_branches_detected": len(reference_union), "selected_branches_detected": len(found),
                        "confirmed_recovered_branches": len(found & reference_union), "unconfirmed_discordant_union": len(found - reference_union),
                        "unconfirmed_discordant_sample_union": len(discordant), "zero_reference_denominator": not bool(reference_union), "reference_recall": recall,
                        "sample_recall_mean": float(np.mean(sample_recalls)) if sample_recalls and scope != "UNKNOWN_TREE" else (0.0 if scope != "UNKNOWN_TREE" else np.nan),
                        "panel_MRD_ever": bool(mrddays), "first_MRD_day": min(mrddays) if mrddays else np.nan,
                        "whole_panel_MRD_ever": bool(ref_mrd_days), "event_status": cl.Event_status, "confirmed_recurrence_day": recurrence,
                        "MRD_before_recurrence": bool(predates) if np.isfinite(recurrence) else np.nan,
                        "recurrence_lead_days": recurrence - min(predates) if predates else np.nan})
    for (p, sid), records in normal.groupby(["PublicationID", "tracerx_id"]):
        if p not in splits: continue
        for scenario, full in [("INTRINSIC_QC", False), ("FULL_QC_SENSITIVITY", True)]:
            for k in BUDGETS:
                for method in POLICIES:
                    plan = choices[(p, method, k)]
                    mrd, detected, npos, nused = call(records, plan, k, full, normal=True)
                    normal_rows.append({"PublicationID": p, "control_id": sid, "source_group": records.source_group.iloc[0], "split": records.split.iloc[0], "scenario": scenario,
                        "method": method, "nominal_K": k, "selected_markers": len(plan), "recorded_markers": nused, "positive_markers": npos, "panel_MRD_false_positive": mrd})
    return pd.DataFrame(patient_rows), pd.DataFrame(sample_rows), pd.DataFrame(clone_rows), pd.DataFrame(normal_rows)


def summarize(patients, controls):
    means = []
    contrasts = []
    for scenario in ["INTRINSIC_QC", "FULL_QC_SENSITIVITY"]:
        for population in ["ALL197_CT_DNA", "NATIVE_ONLY", "LUAD_PRIMARY_REGIONS_GE3", "AUTHOR_PRIMARY_PROXY_OUTSIDE_NATIVE"]:
            frame = patients[patients.scenario == scenario]
            if population == "NATIVE_ONLY": frame = frame[frame.reference_scope == "NATIVE_PRIMARY_TREE"]
            if population == "LUAD_PRIMARY_REGIONS_GE3": frame = frame[frame.primary_LUAD_GE3]
            if population == "AUTHOR_PRIMARY_PROXY_OUTSIDE_NATIVE": frame = frame[frame.reference_scope == population]
            for (split, k, method), group in frame.groupby(["split", "nominal_K", "method"]):
                means.append({"population": population, "scenario": scenario, "split": split, "nominal_K": k, "method": method,
                    "patient_n": len(group), "branch_score_available_n": int(group.reference_recall.notna().sum()), "recall_mean": group.reference_recall.mean(),
                    "whole_reference_positive_patient_n": int((group.whole_reference_branches_detected > 0).sum()), "postoperative_sample_available_n": int((group.postoperative_samples > 0).sum()),
                    "capacity_shortfall_n": int(group.capacity_shortfall.sum()), "MRD_ever_rate": group.panel_MRD_ever.mean(),
                    "whole_MRD_ever_rate": group.whole_panel_MRD_ever.mean(), "true_recurrence_n": int(group.confirmed_recurrence_day.notna().sum()),
                    "MRD_before_recurrence_rate": group.MRD_before_recurrence.mean(), "median_lead_days_detected_only": group.recurrence_lead_days.median()})
            for k in BUDGETS:
                wide = frame[(frame.split == "confirmation") & (frame.nominal_K == k)].pivot(index="patient_id", columns="method", values="reference_recall")
                if not len(wide): continue
                for comparator in ["CLONAL_ONLY", "BRANCH_DIVERSE", "RANDOM_MARKERS"]:
                    delta = wide.INFORMATION_ADAPTIVE - wide[comparator]
                    result = paired_inference(delta, scenario + population + str(k) + comparator)
                    contrasts.append({"population": population, "scenario": scenario, "nominal_K": k, "method": "INFORMATION_ADAPTIVE", "comparator": comparator, **result})
    table = pd.DataFrame(contrasts)
    table["holm_primary_p"] = np.nan
    primary = table[(table.population == "ALL197_CT_DNA") & (table.scenario == "INTRINSIC_QC") & (table.nominal_K == 50)]
    order = primary.sort_values("p_value").index.tolist(); running = 0.0
    for i, ix in enumerate(order):
        running = max(running, min(1.0, table.at[ix, "p_value"] * (len(order) - i))); table.at[ix, "holm_primary_p"] = running
    source = controls.groupby(["source_group", "split", "scenario", "method", "nominal_K"], as_index=False).panel_MRD_false_positive.mean()
    control_summary = source.groupby(["split", "scenario", "method", "nominal_K"]).agg(normal_source_n=("source_group", "nunique"), false_positive_rate=("panel_MRD_false_positive", "mean")).reset_index()
    return pd.DataFrame(means), table, source, control_summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("Use --run after root scientific review of CTDNA_ASSAY_FREEZE.json")
    config = json.loads(CONFIG.read_text())
    for filename, expected in config["source_hashes"].items():
        assert digest(ROOT / filename) == expected, "Frozen input changed: " + filename
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(ROOT / "configs" / "CTDNA_PATIENT_SPLIT.tsv", sep="\t").set_index("patient_id").split.to_dict()
    clinical = read("Table_8_patient_data"); assert len(clinical) == 197
    assert set(clinical.PublicationID) <= set(split)
    plasma = read("Table_17_patient_mutation_data")
    assert not plasma.duplicated(["tracerx_id", "chromosome", "position", "reference", "alternate"]).any()
    samples = read("Table_13_sample_ctDNA")
    valid_ids = set(samples.loc[samples.matches_wes == True, "tracerx_id"])
    assert set(plasma.tracerx_id) <= valid_ids
    library, chosen = build_library(plasma)
    normal, parameters = normal_parameters(read("Table_20_PBMC_variant_data"))
    choices, selections = select_all(library, parameters, clinical)
    library.to_csv(OUT / "SURGERY_ONLY_MARKER_LIBRARY.tsv.gz", sep="\t", index=False)
    selections.to_csv(OUT / "MARKER_SELECTION_FREEZE.tsv.gz", sep="\t", index=False)
    (OUT / "NORMAL_DEVELOPMENT_PARAMETERS.json").write_text(json.dumps(parameters, indent=2))
    # This selection freeze precedes all patient detection calls and clinical scores.
    (OUT / "REPLAY_SELECTION_RECEIPT.json").write_text(json.dumps({"assay_freeze_sha256": digest(CONFIG),
        "marker_selection_sha256": digest(OUT / "MARKER_SELECTION_FREEZE.tsv.gz"), "normal_parameters_sha256": digest(OUT / "NORMAL_DEVELOPMENT_PARAMETERS.json"),
        "status": "MARKER_SELECTION_AND_NORMAL_FIT_COMPLETED_BEFORE_PATIENT_REPLAY"}, indent=2))
    print("Selection frozen; starting count replay", flush=True)
    patient, sample, clone, control = replay(plasma, normal, library, choices, clinical, split, chosen)
    means, comparison, control_source, control_summary = summarize(patient, control)
    for name, frame in [("PATIENT_RECALL.tsv", patient), ("SAMPLE_DETECTIONS.tsv.gz", sample), ("CLONE_DETECTIONS.tsv.gz", clone),
                        ("NORMAL_CONTROL_RECORDS.tsv", control), ("POPULATION_SUMMARY.tsv", means), ("PAIRED_COMPARISONS.tsv", comparison),
                        ("NORMAL_CONTROL_SOURCE_SUMMARY.tsv", control_source), ("NORMAL_FPR_SUMMARY.tsv", control_summary)]:
        frame.to_csv(OUT / name, sep="\t", index=False)
    receipt = {"status": "EXECUTED", "assay_freeze_sha256": digest(CONFIG), "code_sha256": digest(__file__), "patient_n": 197,
               "raw_marker_record_n": len(plasma), "postoperative_sample_n": sample.loc[sample.method == "WHOLE_PSP_REFERENCE", "sample_id"].nunique(),
               "raw_marker_counts_used": True, "original_FASTQ_reprocessed": False, "case_outcomes_used_in_marker_selection": False,
               "scope": config["scope"], "outputs": {f.name: digest(f) for f in OUT.iterdir() if f.is_file()}}
    (OUT / "EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(comparison[(comparison.population == "ALL197_CT_DNA") & (comparison.scenario == "INTRINSIC_QC") & (comparison.nominal_K == 50)].to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
