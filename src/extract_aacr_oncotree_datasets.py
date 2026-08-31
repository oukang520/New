"""Extract independent AACR OncoTree cancer-type datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from build_event_matrix import build_event_matrix, write_qc_report as write_event_qc_report
from build_state_table import build_state_table, write_qc_report as write_state_qc_report


MISSING = {"", "na", "nan", "none", "unknown", "not collected", "not reported", "not applicable"}
STAGE_PRIORITY = {"metastatic": 3, "local_advanced": 2, "primary": 1, "early": 1, "unknown": 0}


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "extract_aacr_oncotree_datasets.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_aacr_table(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, usecols=lambda c: c in set(columns), low_memory=False)


def standardize_stage_group(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in MISSING:
        return "unknown"
    if "metast" in text or "distant" in text or "m1" in text:
        return "metastatic"
    if "primary" in text:
        return "primary"
    if text in {"stage i", "i", "ia", "ib"}:
        return "early"
    if text in {"stage ii", "stage iii", "ii", "iii", "iia", "iib", "iic", "iiia", "iiib", "iiic"}:
        return "local_advanced"
    if text in {"stage iv", "iv", "iva", "ivb", "ivc"}:
        return "metastatic"
    return "unknown"


def normalize_survival_event(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in {"true", "1", "deceased", "dead", "yes"}:
        return "1"
    if text in {"false", "0", "alive", "living", "no"}:
        return "0"
    return ""


def choose_survival_time(row: pd.Series) -> str:
    dead = str(row.get("DEAD", "")).strip().lower() in {"true", "1", "deceased", "dead", "yes"}
    if dead and str(row.get("INT_DOD", "")).strip().lower() not in MISSING:
        return str(row.get("INT_DOD", ""))
    return str(row.get("INT_CONTACT", ""))


def nonmissing_rate(series: pd.Series) -> float:
    values = series.fillna("").astype(str).str.strip().str.lower()
    if len(values) == 0:
        return 0.0
    return float((~values.isin(MISSING)).mean())


def collapse_stage(stages: pd.Series) -> str:
    clean = stages.fillna("unknown").astype(str).map(standardize_stage_group)
    if clean.empty:
        return "unknown"
    ranked = sorted(clean.unique(), key=lambda x: STAGE_PRIORITY.get(x, 0), reverse=True)
    return ranked[0] if ranked else "unknown"


def first_nonmissing(series: pd.Series) -> str:
    for value in series:
        text = str(value if value is not None else "").strip()
        if text.lower() not in MISSING:
            return text
    return ""


def join_unique(series: pd.Series, limit: int = 20) -> str:
    values = []
    seen = set()
    for value in series.dropna().astype(str):
        text = value.strip()
        if text and text.lower() not in MISSING and text not in seen:
            seen.add(text)
            values.append(text)
        if len(values) >= limit:
            break
    suffix = "" if len(seen) <= limit else ";..."
    return ";".join(values) + suffix


def load_clinical(sample_path: Path, patient_path: Path) -> pd.DataFrame:
    sample_cols = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "AGE_AT_SEQ_REPORT",
        "ONCOTREE_CODE",
        "SAMPLE_TYPE",
        "SEQ_ASSAY_ID",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "SAMPLE_TYPE_DETAILED",
        "SAMPLE_CLASS",
    ]
    patient_cols = ["PATIENT_ID", "SEX", "INT_CONTACT", "INT_DOD", "DEAD"]
    samples = read_aacr_table(sample_path, sample_cols)
    patients = read_aacr_table(patient_path, patient_cols)
    clinical = samples.merge(patients, on="PATIENT_ID", how="left")
    clinical["cohort"] = "AACR"
    clinical["patient_id"] = clinical["PATIENT_ID"]
    clinical["sample_id"] = clinical["SAMPLE_ID"]
    clinical["oncotree_code"] = clinical["ONCOTREE_CODE"].fillna("").astype(str).str.strip()
    clinical["cancer_type"] = clinical["CANCER_TYPE"]
    clinical["cancer_type_detailed"] = clinical["CANCER_TYPE_DETAILED"]
    clinical["stage_raw"] = clinical["SAMPLE_TYPE"]
    clinical["stage_group"] = clinical["SAMPLE_TYPE"].map(standardize_stage_group)
    clinical["metastasis_status"] = clinical["SAMPLE_TYPE"]
    clinical["survival_time"] = clinical.apply(choose_survival_time, axis=1)
    clinical["survival_event"] = clinical["DEAD"].map(normalize_survival_event)
    clinical["age"] = clinical["AGE_AT_SEQ_REPORT"]
    clinical["sex"] = clinical["SEX"]
    clinical["source_file"] = str(sample_path)
    return clinical


def build_patient_metadata(sample_metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in sample_metadata.groupby("patient_id", dropna=False):
        rows.append(
            {
                "patient_id": patient_id,
                "sample_id": first_nonmissing(group["sample_id"]),
                "sample_ids": join_unique(group["sample_id"], limit=50),
                "n_samples": int(group["sample_id"].nunique()),
                "cohort": first_nonmissing(group["cohort"]),
                "oncotree_code": first_nonmissing(group["oncotree_code"]),
                "cancer_type": first_nonmissing(group["cancer_type"]),
                "cancer_type_detailed": first_nonmissing(group["cancer_type_detailed"]),
                "stage_raw": join_unique(group["stage_raw"], limit=10),
                "stage_group": collapse_stage(group["stage_group"]),
                "metastasis_status": join_unique(group["metastasis_status"], limit=10),
                "survival_time": first_nonmissing(group["survival_time"]),
                "survival_event": first_nonmissing(group["survival_event"]),
                "age": first_nonmissing(group["age"]),
                "sex": first_nonmissing(group["sex"]),
                "source_file": first_nonmissing(group["source_file"]),
            }
        )
    return pd.DataFrame(rows)


def choose_top_k(mutations: pd.DataFrame, metadata: pd.DataFrame, id_col: str) -> tuple[int, int]:
    n = metadata[id_col].nunique()
    if mutations.empty or n == 0:
        return 0, 0
    support = mutations.drop_duplicates([id_col, "gene"])["gene"].value_counts()
    events_at_3pct = int((support / n >= 0.03).sum())
    if events_at_3pct <= 0:
        return min(25, len(support)), events_at_3pct
    return min(25, max(10, events_at_3pct)), events_at_3pct


def write_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    shown = df.head(max_rows).copy()
    if shown.empty:
        return "(none)"
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_subset_qc(
    code: str,
    output_dir: Path,
    sample_metadata: pd.DataFrame,
    patient_metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    patient_event_frequency: pd.DataFrame,
    sample_event_frequency: pd.DataFrame,
    patient_state_occupancy: pd.DataFrame,
    sample_state_occupancy: pd.DataFrame,
    checks: dict,
) -> None:
    sample_stage_counts = sample_metadata["stage_group"].value_counts().to_dict()
    patient_stage_counts = patient_metadata["stage_group"].value_counts().to_dict()
    lines = [
        f"# AACR {code} Extraction QC",
        "",
        "## Completeness Checks",
        "",
        f"- OncoTree code exact match: {checks['oncotree_exact_match']}",
        f"- Mutation sample IDs all in selected samples: {checks['mutation_sample_ids_in_subset']}",
        f"- Patient event rows match unique patients: {checks['patient_event_rows_match']}",
        f"- Sample event rows match unique samples: {checks['sample_event_rows_match']}",
        f"- Patient state rows match event rows: {checks['patient_state_rows_match']}",
        f"- Sample state rows match event rows: {checks['sample_state_rows_match']}",
        "",
        "## Counts",
        "",
        f"- Unique patients: {patient_metadata['patient_id'].nunique()}",
        f"- Unique samples: {sample_metadata['sample_id'].nunique()}",
        f"- Patients with multiple selected samples: {(patient_metadata['n_samples'].astype(int) > 1).sum()}",
        f"- Mutation rows: {len(mutations)}",
        f"- Mutated patients: {mutations['patient_id'].nunique()}",
        f"- Mutated samples: {mutations['sample_id'].nunique()}",
        f"- Genes/events before top-k filtering: {mutations['gene'].nunique()}",
        f"- Sample stage missing rate: {1.0 - nonmissing_rate(sample_metadata['stage_raw']):.4f}",
        f"- Patient survival missing rate: {1.0 - nonmissing_rate(patient_metadata['survival_time']):.4f}",
        "",
        "## Stage Distribution",
        "",
        f"- Sample-level: {sample_stage_counts}",
        f"- Patient-level rollup: {patient_stage_counts}",
        "",
        "## Patient-Level Top Events",
        "",
        write_markdown_table(patient_event_frequency.head(15)),
        "",
        "## Sample-Level Top Events",
        "",
        write_markdown_table(sample_event_frequency.head(15)),
        "",
        "## Patient-Level Top States",
        "",
        write_markdown_table(patient_state_occupancy.head(15)),
        "",
        "## Sample-Level Top States",
        "",
        write_markdown_table(sample_state_occupancy.head(15)),
        "",
        "## Notes",
        "",
        "- Sample-level state is the cleanest representation for AACR primary/metastatic status because `SAMPLE_TYPE` is sample-specific.",
        "- Patient-level state uses a conservative rollup: metastatic overrides primary when a patient has both selected sample types.",
        "- Outputs are SNV/indel mutation based; CNA is not merged into these first-pass event matrices.",
        "",
    ]
    (output_dir / "extraction_qc.md").write_text("\n".join(lines), encoding="utf-8")


def extract_one(
    code: str,
    clinical: pd.DataFrame,
    mutations_all: pd.DataFrame,
    output_root: Path,
    reports_root: Path,
) -> dict:
    code = code.upper()
    out_dir = output_root / code
    rep_dir = reports_root / code
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    sample_metadata = clinical[clinical["oncotree_code"].str.upper() == code].copy()
    sample_metadata = sample_metadata.drop_duplicates(subset=["sample_id"])
    sample_metadata_out = sample_metadata[
        [
            "patient_id",
            "sample_id",
            "cohort",
            "oncotree_code",
            "cancer_type",
            "cancer_type_detailed",
            "stage_raw",
            "stage_group",
            "metastasis_status",
            "survival_time",
            "survival_event",
            "age",
            "sex",
            "source_file",
        ]
    ].copy()
    patient_metadata = build_patient_metadata(sample_metadata_out)

    selected_samples = set(sample_metadata_out["sample_id"])
    mutations = mutations_all[mutations_all["sample_id"].isin(selected_samples)].copy()
    mutations = mutations.drop_duplicates()
    mutations["oncotree_code"] = code
    mutations["cancer_type_detailed"] = sample_metadata_out["cancer_type_detailed"].mode().iloc[0]

    sample_top_k, sample_events_3pct = choose_top_k(mutations, sample_metadata_out, "sample_id")
    patient_top_k, patient_events_3pct = choose_top_k(mutations, patient_metadata, "patient_id")
    sample_top_k = max(sample_top_k, 1)
    patient_top_k = max(patient_top_k, 1)

    patient_matrix, patient_freq, patient_meta_for_matrix, patient_event_qc = build_event_matrix(
        mutations=mutations,
        clinical=patient_metadata,
        id_level="patient",
        min_frequency=0.0,
        top_k_events=patient_top_k,
        include_cna=False,
    )
    sample_matrix, sample_freq, sample_meta_for_matrix, sample_event_qc = build_event_matrix(
        mutations=mutations,
        clinical=sample_metadata_out,
        id_level="sample",
        min_frequency=0.0,
        top_k_events=sample_top_k,
        include_cna=False,
    )

    patient_state, patient_occupancy, patient_state_qc = build_state_table(
        event_matrix=patient_matrix,
        metadata=patient_meta_for_matrix,
        id_level="patient",
        min_state_count=5,
    )
    sample_state, sample_occupancy, sample_state_qc = build_state_table(
        event_matrix=sample_matrix,
        metadata=sample_meta_for_matrix,
        id_level="sample",
        min_state_count=5,
    )

    sample_metadata_out.to_csv(out_dir / "clinical_samples.csv", index=False)
    patient_metadata.to_csv(out_dir / "clinical_patients.csv", index=False)
    mutations.to_csv(out_dir / "mutations.csv", index=False)
    patient_matrix.to_csv(out_dir / "event_matrix_patient.csv", index=False)
    patient_freq.to_csv(out_dir / "event_frequency_patient.csv", index=False)
    patient_meta_for_matrix.to_csv(out_dir / "sample_metadata_patient.csv", index=False)
    patient_state.to_csv(out_dir / "state_table_patient.csv", index=False)
    patient_occupancy.to_csv(out_dir / "state_occupancy_patient.csv", index=False)
    sample_matrix.to_csv(out_dir / "event_matrix_sample.csv", index=False)
    sample_freq.to_csv(out_dir / "event_frequency_sample.csv", index=False)
    sample_meta_for_matrix.to_csv(out_dir / "sample_metadata_sample.csv", index=False)
    sample_state.to_csv(out_dir / "state_table_sample.csv", index=False)
    sample_occupancy.to_csv(out_dir / "state_occupancy_sample.csv", index=False)

    write_event_qc_report(rep_dir / "event_matrix_patient_qc.md", patient_event_qc, patient_freq)
    write_event_qc_report(rep_dir / "event_matrix_sample_qc.md", sample_event_qc, sample_freq)
    write_state_qc_report(rep_dir / "state_table_patient_qc.md", patient_state_qc, patient_occupancy)
    write_state_qc_report(rep_dir / "state_table_sample_qc.md", sample_state_qc, sample_occupancy)

    checks = {
        "oncotree_exact_match": sample_metadata_out["oncotree_code"].str.upper().eq(code).all(),
        "mutation_sample_ids_in_subset": set(mutations["sample_id"]).issubset(selected_samples),
        "patient_event_rows_match": len(patient_matrix) == patient_metadata["patient_id"].nunique(),
        "sample_event_rows_match": len(sample_matrix) == sample_metadata_out["sample_id"].nunique(),
        "patient_state_rows_match": len(patient_state) == len(patient_matrix),
        "sample_state_rows_match": len(sample_state) == len(sample_matrix),
    }
    write_subset_qc(
        code=code,
        output_dir=rep_dir,
        sample_metadata=sample_metadata_out,
        patient_metadata=patient_metadata,
        mutations=mutations,
        patient_event_frequency=patient_freq,
        sample_event_frequency=sample_freq,
        patient_state_occupancy=patient_occupancy,
        sample_state_occupancy=sample_occupancy,
        checks=checks,
    )

    summary = {
        "oncotree_code": code,
        "cancer_type": sample_metadata_out["cancer_type"].mode().iloc[0] if not sample_metadata_out.empty else "",
        "cancer_type_detailed": sample_metadata_out["cancer_type_detailed"].mode().iloc[0] if not sample_metadata_out.empty else "",
        "unique_patients": int(patient_metadata["patient_id"].nunique()),
        "unique_samples": int(sample_metadata_out["sample_id"].nunique()),
        "multi_sample_patients": int((patient_metadata["n_samples"].astype(int) > 1).sum()),
        "mutation_rows": int(len(mutations)),
        "mutated_patients": int(mutations["patient_id"].nunique()),
        "mutated_samples": int(mutations["sample_id"].nunique()),
        "genes_or_events": int(mutations["gene"].nunique()),
        "patient_top_k_events": int(patient_top_k),
        "sample_top_k_events": int(sample_top_k),
        "patient_events_at_3pct": int(patient_events_3pct),
        "sample_events_at_3pct": int(sample_events_3pct),
        "patient_event_matrix_rows": int(len(patient_matrix)),
        "patient_event_matrix_cols": int(patient_matrix.shape[1] - 1),
        "sample_event_matrix_rows": int(len(sample_matrix)),
        "sample_event_matrix_cols": int(sample_matrix.shape[1] - 1),
        "patient_valid_states": int((patient_occupancy["state_count_flag"] == "valid_state").sum()),
        "sample_valid_states": int((sample_occupancy["state_count_flag"] == "valid_state").sum()),
        "sample_missing_stage_rate": round(1.0 - nonmissing_rate(sample_metadata_out["stage_raw"]), 6),
        "patient_missing_survival_rate": round(1.0 - nonmissing_rate(patient_metadata["survival_time"]), 6),
        "primary_samples": int((sample_metadata_out["stage_group"] == "primary").sum()),
        "metastatic_samples": int((sample_metadata_out["stage_group"] == "metastatic").sum()),
        "unknown_stage_samples": int((sample_metadata_out["stage_group"] == "unknown").sum()),
        "all_checks_passed": bool(all(checks.values())),
        "output_dir": str(out_dir),
        "report_dir": str(rep_dir),
    }
    return summary


def write_combined_report(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        "oncotree_code",
        "cancer_type_detailed",
        "unique_patients",
        "unique_samples",
        "mutated_patients",
        "genes_or_events",
        "patient_event_matrix_cols",
        "sample_event_matrix_cols",
        "primary_samples",
        "metastatic_samples",
        "unknown_stage_samples",
        "patient_valid_states",
        "sample_valid_states",
        "all_checks_passed",
    ]
    lines = [
        "# AACR LUAD/COAD/IDC Extraction Summary",
        "",
        "Each dataset is filtered by exact `ONCOTREE_CODE` from the original AACR clinical sample table.",
        "Both patient-level and sample-level event/state tables are written for each cancer type.",
        "",
        write_markdown_table(summary[cols], max_rows=10),
        "",
        "## Output Layout",
        "",
        "- `processed/aacr_oncotree_subsets/{CODE}/clinical_samples.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/clinical_patients.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/mutations.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/event_matrix_patient.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/event_matrix_sample.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/state_table_patient.csv`",
        "- `processed/aacr_oncotree_subsets/{CODE}/state_table_sample.csv`",
        "- `reports/aacr_oncotree_subsets/{CODE}/extraction_qc.md`",
        "",
        "## Correctness Notes",
        "",
        "- Sample-level outputs preserve AACR sample-specific primary/metastatic labels exactly.",
        "- Patient-level outputs roll up multiple selected samples per patient; metastatic overrides primary.",
        "- All three subsets should have exact OncoTree matches and mutation sample IDs contained in the selected sample set.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AACR OncoTree subsets.")
    parser.add_argument("--codes", nargs="+", default=["LUAD", "COAD", "IDC"])
    parser.add_argument("--sample", default="data/AACR/AACR/data_clinical_sample.txt")
    parser.add_argument("--patient", default="data/AACR/AACR/data_clinical_patient.txt")
    parser.add_argument("--mutations", default="processed/standardized_mutations.csv")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    processed_root = root / "processed" / "aacr_oncotree_subsets"
    reports_root = root / "reports" / "aacr_oncotree_subsets"
    processed_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    setup_logging(root)

    clinical = load_clinical(Path(args.sample), Path(args.patient))
    mutation_cols = ["patient_id", "sample_id", "cohort", "cancer_type", "gene", "alteration_type", "alteration_binary", "source_file"]
    mutations_all = pd.read_csv(args.mutations, dtype=str, usecols=lambda c: c in set(mutation_cols), low_memory=False)
    mutations_all = mutations_all[mutations_all["alteration_binary"].fillna("1").astype(str) != "0"].copy()

    summaries = []
    for code in args.codes:
        logging.info("Extracting AACR subset %s", code)
        summaries.append(extract_one(code, clinical, mutations_all, processed_root, reports_root))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(reports_root / "subset_extraction_summary.csv", index=False)
    summary_df.to_csv(processed_root / "subset_extraction_summary.csv", index=False)
    write_combined_report(summary_df, reports_root / "subset_extraction_summary.md")

    print("AACR subset extraction summary")
    for row in summaries:
        print(
            f"{row['oncotree_code']}: patients={row['unique_patients']}, samples={row['unique_samples']}, "
            f"patient_events={row['patient_event_matrix_cols']}, sample_events={row['sample_event_matrix_cols']}, "
            f"checks={row['all_checks_passed']}"
        )


if __name__ == "__main__":
    main()
