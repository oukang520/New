"""Audit AACR/GENIE data at fine OncoTree cancer-type resolution."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


MISSING = {"", "na", "nan", "none", "unknown", "not collected", "not reported", "not applicable"}
UNKNOWN_CODES = {"UNKNOWN"}
BROAD_CODES = {
    "ADNOS",
    "BRCA",
    "COADREAD",
    "LUNG",
    "NSCLC",
}
UNKNOWN_PRIMARY_CODES = {"CUP"}


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "audit_aacr_by_cancer.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False, **kwargs)


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


def nonmissing_rate(series: pd.Series) -> float:
    values = series.fillna("").astype(str).str.strip().str.lower()
    if len(values) == 0:
        return 0.0
    return float((~values.isin(MISSING)).mean())


def normalize_survival_event(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series("", index=series.index, dtype=object)
    out[values.isin({"true", "1", "deceased", "dead", "yes"})] = "1"
    out[values.isin({"false", "0", "alive", "living", "no"})] = "0"
    return out


def choose_survival_time(df: pd.DataFrame) -> pd.Series:
    if "DEAD" in df.columns and "INT_DOD" in df.columns and "INT_CONTACT" in df.columns:
        dead = df["DEAD"].fillna("").astype(str).str.lower().isin({"true", "1", "dead", "deceased"})
        return df["INT_DOD"].where(dead, df["INT_CONTACT"])
    if "INT_CONTACT" in df.columns:
        return df["INT_CONTACT"]
    return pd.Series([""] * len(df), index=df.index)


def diagnosis_specificity(row: dict) -> str:
    code = str(row.get("oncotree_code", "")).strip().upper()
    detailed = str(row.get("cancer_type_detailed", "")).strip().lower()
    if code in UNKNOWN_CODES or detailed == "":
        return "unknown_diagnosis"
    if code in UNKNOWN_PRIMARY_CODES or "unknown primary" in detailed:
        return "unknown_primary"
    if code in BROAD_CODES or "adenocarcinoma, nos" in detailed or "sarcoma, nos" in detailed:
        return "broad_aggregate"
    return "fine_grained"


def score_feasibility(row: dict) -> tuple[str, str, str]:
    n = int(row["number_of_unique_patients"])
    genes = int(row["number_of_genes_or_events"])
    mutated_patients = int(row["number_of_mutated_patients"])
    stage_missing = float(row["missing_rate_stage"])
    primary = int(row["primary_samples"])
    metastatic = int(row["metastatic_samples"])

    has_stage = stage_missing <= 0.30 and (primary + metastatic) > 0
    has_two_groups = primary >= 20 and metastatic >= 20
    enough_events = genes >= 10 and mutated_patients >= 50
    specificity = diagnosis_specificity(row)

    if specificity == "unknown_diagnosis":
        return (
            "Not usable",
            "not recommended",
            "Diagnosis is unknown or blank, so it does not meet fine-grained cancer-type requirements.",
        )
    if specificity == "unknown_primary":
        return (
            "Low",
            "exploratory only",
            "Cancer of unknown primary is not a fine-grained cancer type; use only for exploratory sensitivity checks.",
        )

    if n >= 300 and enough_events and has_stage and has_two_groups:
        if specificity == "broad_aggregate":
            return (
                "Medium",
                "main cohort with caution",
                "Data support is strong, but the diagnosis is a broad aggregate category rather than a fine subtype.",
            )
        return (
            "High",
            "main cohort",
            "Adequate size, mutation events, and both primary/metastatic state groups are available.",
        )
    if n >= 300 and enough_events and has_stage:
        return (
            "Medium",
            "main cohort with caution",
            "Adequate size and mutation data, but primary/metastatic balance is limited.",
        )
    if n >= 100 and enough_events and has_stage:
        return (
            "Medium",
            "validation cohort",
            "Usable mutation and state information, but sample size is better suited for validation.",
        )
    if n >= 50 and genes >= 5 and has_stage:
        return (
            "Low",
            "exploratory only",
            "Some mutation and state information are present, but sample/event support is weak.",
        )
    return (
        "Not usable",
        "not recommended",
        "Insufficient sample size, mutation support, or usable state information for Rel-ObsTQ-MHN.",
    )


def build_aacr_by_cancer_summary(
    sample_path: Path,
    patient_path: Path,
    mutation_path: Path,
    cna_path: Path | None = None,
) -> pd.DataFrame:
    logging.info("Reading AACR sample table: %s", sample_path)
    sample_cols = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "AGE_AT_SEQ_REPORT",
        "ONCOTREE_CODE",
        "SAMPLE_TYPE",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "SAMPLE_TYPE_DETAILED",
        "SAMPLE_CLASS",
    ]
    samples = read_table(sample_path, usecols=lambda c: c in set(sample_cols))
    samples = samples.rename(columns=lambda c: c.strip())
    for col in sample_cols:
        if col not in samples.columns:
            samples[col] = ""
    samples["ONCOTREE_CODE"] = samples["ONCOTREE_CODE"].fillna("").astype(str).str.strip()
    samples = samples[samples["ONCOTREE_CODE"] != ""].copy()
    samples["stage_group"] = samples["SAMPLE_TYPE"].map(standardize_stage_group)

    logging.info("Reading AACR patient table: %s", patient_path)
    patient_cols = ["PATIENT_ID", "SEX", "INT_CONTACT", "INT_DOD", "DEAD"]
    patients = read_table(patient_path, usecols=lambda c: c in set(patient_cols))
    for col in patient_cols:
        if col not in patients.columns:
            patients[col] = ""
    clinical = samples.merge(patients, on="PATIENT_ID", how="left")
    clinical["survival_time"] = choose_survival_time(clinical)
    clinical["survival_event"] = normalize_survival_event(clinical["DEAD"])

    logging.info("Reading standardized mutation table: %s", mutation_path)
    mut_cols = ["patient_id", "sample_id", "gene", "alteration_type", "alteration_binary"]
    mutations = pd.read_csv(mutation_path, dtype=str, usecols=lambda c: c in set(mut_cols), low_memory=False)
    mutations = mutations[mutations["alteration_binary"].fillna("1").astype(str) != "0"].copy()
    sample_map = clinical[
        [
            "PATIENT_ID",
            "SAMPLE_ID",
            "ONCOTREE_CODE",
            "CANCER_TYPE",
            "CANCER_TYPE_DETAILED",
            "stage_group",
        ]
    ].rename(columns={"PATIENT_ID": "patient_id", "SAMPLE_ID": "sample_id"})
    mut_by_sample = mutations.merge(sample_map, on=["patient_id", "sample_id"], how="inner")

    cna_samples = load_cna_sample_ids(cna_path) if cna_path else set()
    clinical["has_cna_profile"] = clinical["SAMPLE_ID"].isin(cna_samples) if cna_samples else False

    rows = []
    grouped = clinical.groupby("ONCOTREE_CODE", dropna=False)
    for code, group in grouped:
        code = str(code)
        mut_group = mut_by_sample[mut_by_sample["ONCOTREE_CODE"] == code]
        patient_count = group["PATIENT_ID"].nunique()
        sample_count = group["SAMPLE_ID"].nunique()
        mutated_patients = mut_group["patient_id"].nunique()
        mutated_samples = mut_group["sample_id"].nunique()
        gene_count = mut_group["gene"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
        mutation_rows = len(mut_group)
        stage_missing = 1.0 - nonmissing_rate(group["SAMPLE_TYPE"])
        survival_missing = 1.0 - nonmissing_rate(group["survival_time"])
        duplicated_patient_rate = 1.0 - patient_count / len(group) if len(group) else 0.0
        stage_counts = group["stage_group"].value_counts()
        top_events = (
            mut_group.drop_duplicates(["patient_id", "gene"])["gene"].value_counts().head(15).index.tolist()
            if not mut_group.empty
            else []
        )
        event_support = mut_group.drop_duplicates(["patient_id", "gene"])["gene"].value_counts()
        events_ge_3pct = int((event_support / max(patient_count, 1) >= 0.03).sum()) if not event_support.empty else 0
        recommended_top_events = min(25, max(10, events_ge_3pct)) if events_ge_3pct else 0

        row = {
            "oncotree_code": code,
            "cancer_type": most_common(group["CANCER_TYPE"]),
            "cancer_type_detailed": most_common(group["CANCER_TYPE_DETAILED"]),
            "number_of_unique_patients": int(patient_count),
            "number_of_unique_samples": int(sample_count),
            "number_of_mutated_patients": int(mutated_patients),
            "number_of_mutated_samples": int(mutated_samples),
            "number_of_genes_or_events": int(gene_count),
            "mutation_rows": int(mutation_rows),
            "mutation_patient_coverage": round(mutated_patients / patient_count, 6) if patient_count else 0.0,
            "missing_rate_stage": round(stage_missing, 6),
            "missing_rate_survival": round(survival_missing, 6),
            "duplicated_patient_rate": round(duplicated_patient_rate, 6),
            "primary_samples": int(stage_counts.get("primary", 0)),
            "metastatic_samples": int(stage_counts.get("metastatic", 0)),
            "unknown_stage_samples": int(stage_counts.get("unknown", 0)),
            "cna_profile_samples": int(group["has_cna_profile"].sum()),
            "has_cna": bool(group["has_cna_profile"].any()),
            "top_events_by_patient": "; ".join(top_events),
            "events_at_3pct_frequency": events_ge_3pct,
            "recommended_top_events": recommended_top_events,
        }
        row["diagnosis_specificity"] = diagnosis_specificity(row)
        level, use, reason = score_feasibility(row)
        row["feasibility_level"] = level
        row["recommended_use"] = use
        row["reason"] = reason
        rows.append(row)

    out = pd.DataFrame(rows)
    level_rank = {"High": 3, "Medium": 2, "Low": 1, "Not usable": 0}
    out["_rank"] = out["feasibility_level"].map(level_rank)
    out = out.sort_values(
        ["_rank", "number_of_unique_patients", "number_of_genes_or_events"],
        ascending=[False, False, False],
    ).drop(columns=["_rank"])
    return out


def most_common(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values[~values.str.lower().isin(MISSING)]
    if values.empty:
        return ""
    return str(values.mode().iloc[0])


def load_cna_sample_ids(cna_path: Path | None) -> set[str]:
    if cna_path is None or not cna_path.exists():
        return set()
    try:
        with cna_path.open("r", encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().rstrip("\n\r").split("\t")
        if len(header) <= 1:
            return set()
        first = header[0].strip().lower()
        if first in {"hugo_symbol", "gene", "genes"}:
            return set(header[1:])
        return set(header)
    except Exception as exc:
        logging.warning("Could not inspect CNA header %s: %s", cna_path, exc)
        return set()


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    shown = df.loc[:, columns].head(max_rows).copy()
    if shown.empty:
        return "(none)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, report_path: Path) -> None:
    counts = summary["feasibility_level"].value_counts().to_dict()
    high = summary[summary["feasibility_level"] == "High"].copy()
    medium = summary[summary["feasibility_level"] == "Medium"].copy()
    low = summary[summary["feasibility_level"] == "Low"].copy()
    not_usable = summary[summary["feasibility_level"] == "Not usable"].copy()

    display_cols = [
        "oncotree_code",
        "cancer_type_detailed",
        "number_of_unique_patients",
        "number_of_mutated_patients",
        "number_of_genes_or_events",
        "primary_samples",
        "metastatic_samples",
        "missing_rate_survival",
        "recommended_top_events",
        "recommended_use",
    ]
    lines = [
        "# AACR Fine-Grained Cancer-Type Feasibility",
        "",
        "Classification key: `ONCOTREE_CODE` from `data_clinical_sample.txt`.",
        "Stage/progression status uses AACR `SAMPLE_TYPE`, standardized mainly as `primary`, `metastatic`, or `unknown`.",
        "",
        "## Feasibility Counts",
        "",
        f"- High: {counts.get('High', 0)}",
        f"- Medium: {counts.get('Medium', 0)}",
        f"- Low: {counts.get('Low', 0)}",
        f"- Not usable: {counts.get('Not usable', 0)}",
        "",
        "## High-Priority Cancer-Type Datasets",
        "",
        markdown_table(high, display_cols, max_rows=40),
        "",
        "## Medium-Priority Cancer-Type Datasets",
        "",
        markdown_table(medium, display_cols, max_rows=40),
        "",
        "## Low-Priority / Exploratory Cancer-Type Datasets",
        "",
        markdown_table(low, display_cols, max_rows=30),
        "",
        "## Not Recommended",
        "",
        markdown_table(not_usable, ["oncotree_code", "cancer_type_detailed", "number_of_unique_patients", "number_of_genes_or_events", "reason"], max_rows=30),
        "",
        "## Interpretation Notes",
        "",
        "- High datasets are the best candidates for cancer-specific Rel-ObsTQ-MHN analyses.",
        "- Medium datasets are suitable for validation or cautious main analysis, depending on state balance.",
        "- Low datasets should be exploratory only because state/event support is limited.",
        "- Not usable datasets are too small or lack enough mutation/state support for reliable state-level analysis.",
        "- CNA availability is recorded, but first-pass analysis should remain SNV/indel-only unless a CNA sensitivity analysis is planned.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AACR by fine cancer type.")
    parser.add_argument("--sample", default="data/AACR/AACR/data_clinical_sample.txt")
    parser.add_argument("--patient", default="data/AACR/AACR/data_clinical_patient.txt")
    parser.add_argument("--mutations", default="processed/standardized_mutations.csv")
    parser.add_argument("--cna", default="data/AACR/data_CNA.txt")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    reports = root / "reports"
    processed = root / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    setup_logging(root)

    summary = build_aacr_by_cancer_summary(
        sample_path=Path(args.sample),
        patient_path=Path(args.patient),
        mutation_path=Path(args.mutations),
        cna_path=Path(args.cna) if args.cna else None,
    )
    summary.to_csv(reports / "aacr_by_oncotree_feasibility_table.csv", index=False)
    summary.to_csv(processed / "aacr_by_oncotree_dataset_summary.csv", index=False)
    write_report(summary, reports / "aacr_by_oncotree_feasibility_report.md")

    counts = summary["feasibility_level"].value_counts().to_dict()
    print("AACR by cancer-type audit summary")
    print(f"oncotree_datasets={len(summary)}")
    print(
        "feasibility_counts="
        + ", ".join(f"{level}:{counts.get(level, 0)}" for level in ["High", "Medium", "Low", "Not usable"])
    )
    print("top_high=" + ", ".join(summary[summary["feasibility_level"] == "High"]["oncotree_code"].head(20).tolist()))


if __name__ == "__main__":
    main()
