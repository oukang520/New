#!/usr/bin/env python3
"""Descriptive, scorer-only independent future-tissue lineage validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ctdna"
CONFIG = ROOT / "configs" / "CTDNA_FUTURE_SCORER_FREEZE.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads(CONFIG.read_text())
    for name, expected in config["source_hashes"].items():
        assert digest(ROOT / name) == expected, name
    future = pd.read_csv(ROOT / "data/metastasis_audit/native_cluster_future_marker_support.tsv", sep="\t")
    regions = pd.read_csv(ROOT / "data/metastasis_audit/metastasis_regions.tsv", sep="\t")
    reference = pd.read_csv(ROOT / "data/native_audit/branch_reference.tsv", sep="\t")
    chosen = pd.read_csv(ROOT / "data/native_audit/patient_primary_tumour.tsv", sep="\t")
    patient = pd.read_csv(OUT / "PATIENT_RECALL.tsv", sep="\t")
    sample = pd.read_csv(OUT / "SAMPLE_DETECTIONS.tsv.gz", sep="\t")
    clone = pd.read_csv(OUT / "CLONE_DETECTIONS.tsv.gz", sep="\t")
    library = pd.read_csv(OUT / "SURGERY_ONLY_MARKER_LIBRARY.tsv.gz", sep="\t")
    future = future.merge(regions[["patient_id", "tumour_id", "RegionID", "sampleTypeDetail"]],
                          on=["patient_id", "tumour_id", "RegionID"], how="left", validate="many_to_one")
    future = future.merge(reference[["patient_id", "tumour_id", "cluster", "reference_markers", "eligible_multimarker_reference"]]
                          .rename(columns={"cluster": "native_cluster"}),
                          on=["patient_id", "tumour_id", "native_cluster"], how="left", validate="many_to_one")
    metachronous = future.sampleTypeDetail.eq("metachronousMet")
    valid = (metachronous & future.native_is_trunk.eq(False) & future.multimarker_future_lineage_support.eq(True)
             & future.eligible_multimarker_reference.eq(True) & (future.reference_markers >= 3))
    targets = future[valid].copy()
    targets["branch"] = targets.tumour_id.astype(str) + ":" + targets.native_cluster.astype(int).astype(str)
    assert (targets.markers_positive >= 3).all(), "Future tissue requires three shared owned markers"
    targets.to_csv(OUT / "FUTURE_TISSUE_TARGET_SUPPORT.tsv", sep="\t", index=False)
    target = targets.groupby(["patient_id", "tumour_id", "branch"], as_index=False).agg(
        native_cluster=("native_cluster", "first"), future_region_n=("RegionID", "nunique"),
        max_shared_positive_markers=("markers_positive", "max"),
        author_seeding_supported=("multimarker_author_seeding_support", "any"))
    capacity = (library[library.valid_surgery & library.branch_group.notna() & library.branch_group.ne("")]
                .groupby(["PublicationID", "branch_group"]).marker_key.nunique().to_dict())
    target["available_surgical_PSP_markers"] = [capacity.get((p, b), 0) for p, b in zip(target.patient_id, target.branch)]
    target["plasma_multimarker_measurable"] = target.available_surgical_PSP_markers >= 2
    target.to_csv(OUT / "FUTURE_TISSUE_TARGETS.tsv", sep="\t", index=False)
    # Source pairing records reflect only the selected primary tumour.
    selected = chosen.set_index("patient_id").tumour_id.to_dict()
    source_pair = set(regions.loc[(regions.sampleTypeDetail == "metachronousMet")
                                  & (regions.tumour_id == regions.patient_id.map(selected)), "patient_id"])
    target_sets = {p: set(g.branch) for p, g in target.groupby("patient_id")}
    seeded_sets = {p: set(g.loc[g.author_seeding_supported, "branch"]) for p, g in target.groupby("patient_id")}
    measurable_sets = {p: set(g.loc[g.plasma_multimarker_measurable, "branch"]) for p, g in target.groupby("patient_id")}
    sample_key = ["patient_id", "sample_id", "scenario", "method", "nominal_K"]
    assert not sample.duplicated(sample_key).any()
    clone = clone.merge(sample[sample_key + ["days_post_surgery"]], on=sample_key, how="left", validate="many_to_one")
    assert clone.days_post_surgery.notna().all() and (clone.days_post_surgery > 0).all()
    detected = dict(tuple(clone.groupby(["patient_id", "scenario", "method", "nominal_K"])))
    individual = patient.drop_duplicates("patient_id").set_index("patient_id")
    patient_rows = []
    for p, metadata in individual.iterrows():
        expected = target_sets.get(p, set()); seeded = seeded_sets.get(p, set())
        measurable = measurable_sets.get(p, set())
        status = ("PAIRED_FUTURE_TISSUE_NONTRUNK_TARGET" if expected else
                  "PAIRED_FUTURE_TISSUE_NO_ELIGIBLE_NONTRUNK_TARGET" if p in source_pair else "NO_PAIRED_FUTURE_TISSUE")
        for scenario in ["INTRINSIC_QC", "FULL_QC_SENSITIVITY"]:
            plans = [(m, k) for k in [20, 50, 100] for m in ["CLONAL_ONLY", "BRANCH_DIVERSE", "RANDOM_MARKERS", "INFORMATION_ADAPTIVE"]]
            plans.append(("WHOLE_PSP_REFERENCE", 0))
            for method, k in plans:
                rows = detected.get((p, scenario, method, k), clone.iloc[:0])
                found = set(rows.branch)
                recurrence = metadata.confirmed_recurrence_day
                before = set(rows.loc[rows.days_post_surgery <= recurrence, "branch"]) if pd.notna(recurrence) else set()
                patient_rows.append({"patient_id": p, "split": metadata.split, "reference_scope": metadata.reference_scope,
                    "primary_LUAD_GE3": metadata.primary_LUAD_GE3, "source_pair_status": status,
                    "postoperative_samples": metadata.postoperative_samples, "scenario": scenario, "method": method, "nominal_K": k,
                    "future_nontrunk_targets": len(expected), "future_targets_PSP_measurable": len(measurable),
                    "future_targets_below_PSP_capacity": len(expected - measurable), "future_targets_recovered": len(found & expected),
                    "future_target_recall": len(found & expected) / len(expected) if expected else np.nan,
                    "author_seeding_targets": len(seeded), "author_seeding_targets_recovered": len(found & seeded),
                    "author_seeding_target_recall": len(found & seeded) / len(seeded) if seeded else np.nan,
                    "confirmed_recurrence_day": recurrence,
                    "future_targets_before_recurrence": len(before & expected) if pd.notna(recurrence) else np.nan,
                    "future_target_recall_before_recurrence": len(before & expected) / len(expected) if expected and pd.notna(recurrence) else np.nan})
    scores = pd.DataFrame(patient_rows)
    assert scores.patient_id.nunique() == 197
    assert scores.future_target_recall.dropna().between(0, 1).all()
    assert (scores.future_targets_recovered <= scores.future_nontrunk_targets).all()
    scores.to_csv(OUT / "FUTURE_TISSUE_PATIENT_SCORES.tsv", sep="\t", index=False)
    summaries = []
    for population in ["ALL197_CT_DNA", "LUAD_PRIMARY_REGIONS_GE3"]:
        frame = scores if population == "ALL197_CT_DNA" else scores[scores.primary_LUAD_GE3]
        for (split, scenario, method, k), group in frame.groupby(["split", "scenario", "method", "nominal_K"]):
            applicable = group[group.future_nontrunk_targets > 0]
            measurable = applicable[applicable.postoperative_samples > 0]
            summaries.append({"population": population, "split": split, "scenario": scenario, "method": method, "nominal_K": k,
                "availability_patient_n": len(group), "paired_future_tissue_patient_n": int(group.source_pair_status.ne("NO_PAIRED_FUTURE_TISSUE").sum()),
                "future_target_patient_n": len(applicable), "future_target_plasma_available_patient_n": len(measurable),
                "future_target_n": int(group.future_nontrunk_targets.sum()),
                "below_PSP_capacity_target_n": int(group.future_targets_below_PSP_capacity.sum()),
                "future_target_recall_mean": applicable.future_target_recall.mean(),
                "future_target_recall_measured_only_mean": measurable.future_target_recall.mean(),
                "author_seeding_target_patient_n": int((group.author_seeding_targets > 0).sum()),
                "author_seeding_target_recall_mean": group.author_seeding_target_recall.mean(),
                "true_recurrence_date_target_patient_n": int((applicable.confirmed_recurrence_day.notna()).sum()),
                "future_target_recall_before_recurrence_mean": group.future_target_recall_before_recurrence.mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "FUTURE_TISSUE_SUMMARY.tsv", sep="\t", index=False)
    receipt = {"status": "DESCRIPTIVE_SCORER_EXECUTED", "code_sha256": digest(__file__), "scorer_freeze_sha256": digest(CONFIG),
               "assay_marker_choices_changed": False, "primary_family_changed": False, "scope": config["scope"],
               "source_hashes": config["source_hashes"],
               "outputs": {p.name: digest(p) for p in OUT.glob("FUTURE_TISSUE_*.tsv")}}
    (OUT / "FUTURE_TISSUE_SCORER_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(summary[(summary.population == "ALL197_CT_DNA") & (summary.split == "confirmation")
                  & (summary.scenario == "INTRINSIC_QC") & summary.nominal_K.isin([0, 50])].to_string(index=False))


if __name__ == "__main__":
    main()
