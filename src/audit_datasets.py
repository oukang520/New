"""Audit local tumor cross-sectional datasets for Rel-ObsTQ-MHN readiness."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".maf",
    ".vcf",
    ".json",
    ".clinical",
    ".pdf",
}

STANDARD_FIELDS = [
    ("patient_id", "Unique patient or donor identifier."),
    ("sample_id", "Unique tumor/sample identifier."),
    ("cohort", "Dataset or project label."),
    ("cancer_type", "Cancer type or OncoTree/project label."),
    ("stage_raw", "Original clinical stage, tumor stage, or progression-status value."),
    ("stage_group", "Standardized stage/progression group: early, local_advanced, metastatic, primary, or unknown."),
    ("metastasis_status", "Primary/metastatic status if available."),
    ("gene", "Gene/event identifier."),
    ("alteration_type", "Mutation, CNA, or other alteration category."),
    ("alteration_binary", "1 if the alteration/event is present."),
    ("survival_time", "Survival/follow-up time if available; not used as an external time anchor."),
    ("survival_event", "Vital-status/event indicator if available."),
    ("age", "Age at diagnosis, enrollment, or sequencing if available."),
    ("sex", "Patient sex if available."),
    ("source_file", "Original file path from which a standardized row was derived."),
]

MISSING_LIKE = {
    "",
    "na",
    "nan",
    "none",
    "null",
    "unknown",
    "not reported",
    "not collected",
    "not applicable",
    "not available",
}


@dataclass
class FileInfo:
    path: Path
    rel_path: str
    file_name: str
    file_type: str
    file_size: int
    guessed_dataset_name: str = "unknown"
    guessed_content_type: str = "unknown"
    delimiter_or_format: str = "unknown"
    number_of_rows: int | None = None
    number_of_columns: int | None = None
    read_success: bool = False
    read_error_if_any: str = ""
    columns: list[str] = field(default_factory=list)
    sample_df: pd.DataFrame | None = None


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "audit_datasets.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def safe_read_lines(path: Path, limit: int = 40) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            with path.open("r", encoding=encoding, errors="replace") as handle:
                lines = []
                for _, line in zip(range(limit), handle):
                    lines.append(line.rstrip("\n\r"))
                return lines
        except Exception:
            continue
    return []


def infer_delimiter(path: Path, lines: list[str]) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".maf", ".clinical"}:
        return "\t"
    if suffix == ".csv":
        return ","
    non_comment = [line for line in lines if line.strip() and not line.startswith("#")]
    probe = non_comment[0] if non_comment else (lines[0] if lines else "")
    tab_count = probe.count("\t")
    comma_count = probe.count(",")
    semicolon_count = probe.count(";")
    if tab_count >= comma_count and tab_count >= semicolon_count and tab_count > 0:
        return "\t"
    if comma_count >= semicolon_count and comma_count > 0:
        return ","
    if semicolon_count > 0:
        return ";"
    return "\t"


def count_rows(path: Path) -> int | None:
    try:
        if path.stat().st_size > 200 * 1024 * 1024:
            line_count = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    line_count += chunk.count(b"\n")
            return max(line_count - 1, 0)
        total = 0
        non_comment = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                if not line.startswith("#"):
                    non_comment += 1
        if non_comment > 1:
            return non_comment - 1
        if total > 1:
            return total - 1
        return 0
    except Exception:
        return None


def read_table_sample(path: Path, nrows: int = 50) -> tuple[pd.DataFrame | None, str, int | None, int | None, bool, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return None, "pdf", None, None, False, "PDF is not parsed as a tabular data file."
    if suffix == ".json":
        try:
            df = pd.read_json(path, nrows=nrows)
            return df, "json", len(df), len(df.columns), True, ""
        except Exception as exc:
            return None, "json", None, None, False, str(exc)
    if suffix == ".xlsx":
        try:
            df = pd.read_excel(path, nrows=nrows, dtype=str)
            return df, "xlsx", None, len(df.columns), True, ""
        except Exception as exc:
            return None, "xlsx", None, None, False, str(exc)

    lines = safe_read_lines(path)
    delimiter = infer_delimiter(path, lines)
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            comment="#",
            nrows=nrows,
            dtype=str,
            low_memory=False,
        )
        rows = count_rows(path)
        return df, "tab" if delimiter == "\t" else delimiter, rows, len(df.columns), True, ""
    except Exception as exc:
        return None, "tab" if delimiter == "\t" else delimiter, None, None, False, str(exc)


def collect_files(data_dir: Path) -> list[Path]:
    return sorted([p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS])


def lower_columns(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c).strip() for c in columns}


def candidate_columns(columns: Iterable[str], standard_field: str) -> list[str]:
    cols = [str(c).strip() for c in columns]
    lower = lower_columns(cols)

    exact_map = {
        "patient_id": [
            "patient_id",
            "icgc_donor_id",
            "submitted_donor_id",
            "donor_id",
            "bcr_patient_barcode",
            "case_id",
        ],
        "sample_id": [
            "sample_id",
            "tumor_sample_barcode",
            "tumour_sample_barcode",
            "icgc_sample_id",
            "submitted_sample_id",
            "submitted_specimen_id",
            "icgc_specimen_id",
        ],
        "gene": [
            "hugo_symbol",
            "gene",
            "gene_symbol",
            "gene_affected",
            "ensembl_gene_id",
            "symbol",
        ],
        "alteration_type": [
            "variant_classification",
            "mutation_type",
            "consequence_type",
            "copy_number",
            "segment_mean",
            "alteration",
            "alteration_type",
        ],
        "stage_raw": [
            "stage",
            "clinical_stage",
            "pathologic_stage",
            "pathological_stage",
            "ajcc_stage",
            "tumor_stage",
            "tumour_stage",
            "donor_tumour_stage_at_diagnosis",
            "tumour_stage_at_diagnosis",
            "tumor_stage_at_diagnosis",
        ],
        "metastasis_status": [
            "metastasis_status",
            "metastatic_status",
            "sample_type",
            "specimen_type",
            "sample_type_detailed",
            "disease_status_last_followup",
        ],
        "survival_time": [
            "survival_time",
            "donor_survival_time",
            "days_to_death",
            "days_to_last_followup",
            "int_dod",
            "int_contact",
            "donor_interval_of_last_followup",
        ],
        "survival_event": [
            "survival_event",
            "vital_status",
            "donor_vital_status",
            "dead",
            "os_status",
            "event_status",
        ],
        "age": [
            "age",
            "age_at_diagnosis",
            "donor_age_at_diagnosis",
            "donor_age_at_enrollment",
            "age_at_seq_report",
        ],
        "sex": ["sex", "gender", "donor_sex"],
        "cohort": ["cohort", "project_code", "study", "center"],
        "cancer_type": ["cancer_type", "cancer_type_detailed", "oncotree_code", "project_code"],
    }
    out = []
    for key in exact_map.get(standard_field, []):
        if key in lower:
            out.append(lower[key])

    if standard_field == "stage_raw":
        out.extend([c for c in cols if "stage" in c.lower() and "system" not in c.lower()])
        out.extend([c for c in cols if "ajcc" in c.lower()])
    elif standard_field == "metastasis_status":
        out.extend([c for c in cols if "metast" in c.lower()])
        out.extend([c for c in cols if c.lower() in {"sample type", "sample_type", "specimen_type"}])
    elif standard_field == "survival_time":
        out.extend([c for c in cols if "survival" in c.lower() and "time" in c.lower()])
        out.extend([c for c in cols if "followup" in c.lower() or "follow_up" in c.lower()])
    elif standard_field == "survival_event":
        out.extend([c for c in cols if "vital" in c.lower() or c.lower() == "dead"])
    elif standard_field == "gene":
        out.extend([c for c in cols if "gene" in c.lower() and "genealogy" not in c.lower()])

    deduped = []
    seen = set()
    for col in out:
        if col and col not in seen:
            seen.add(col)
            deduped.append(col)
    return deduped


def nonmissing_rate(series: pd.Series) -> float:
    values = series.fillna("").astype(str).str.strip().str.lower()
    if len(values) == 0:
        return 0.0
    return float((~values.isin(MISSING_LIKE)).mean())


def guess_content_type(df: pd.DataFrame | None, path: Path) -> str:
    name = path.name.lower()
    if df is None or df.empty:
        if path.suffix.lower() == ".pdf":
            return "documentation"
        return "unknown"
    cols = list(df.columns)
    patient = candidate_columns(cols, "patient_id")
    sample = candidate_columns(cols, "sample_id")
    gene = candidate_columns(cols, "gene")
    alteration = candidate_columns(cols, "alteration_type")
    stage = candidate_columns(cols, "stage_raw")
    metastasis = candidate_columns(cols, "metastasis_status")
    survival = candidate_columns(cols, "survival_time") + candidate_columns(cols, "survival_event")

    lower = [c.lower() for c in cols]
    has_mutation_id = any("mutation" in c for c in lower)
    has_position = any(c in lower for c in ["chromosome_start", "start_position", "chromosome"])
    has_cna = any(c in lower for c in ["copy_number", "segment_mean", "segment_median"])
    has_expression = "exp_" in name or "expression" in name or "meth_" in name or "methyl" in name

    if has_expression:
        return "matrix"
    if has_cna or ("cna" in name and gene):
        return "CNA"
    if gene and (patient or sample) and (alteration or has_mutation_id or has_position) and not has_expression:
        return "mutation"
    if (patient or sample) and (stage or metastasis or survival or candidate_columns(cols, "age") or candidate_columns(cols, "sex")):
        return "clinical"
    if gene and len(cols) > 20:
        return "matrix"
    if "manifest" in name:
        return "manifest"
    if "meta" in name:
        return "metadata"
    return "unknown"


def guess_dataset_name(path: Path, df: pd.DataFrame | None, data_dir: Path) -> str:
    if df is not None and "project_code" in df.columns:
        values = df["project_code"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return str(values.iloc[0])
    rel_parts = path.relative_to(data_dir).parts
    for part in rel_parts:
        bracket = re.match(r"^\[([^\]]+)\]", part)
        if bracket:
            return bracket.group(1)
        if part.upper() == "AACR":
            return "AACR"
    if path.name.lower().startswith("hgnc"):
        return "reference_hgnc"
    return rel_parts[0] if rel_parts else "unknown"


def scan_data_files(data_dir: Path) -> list[FileInfo]:
    infos: list[FileInfo] = []
    for path in collect_files(data_dir):
        rel_path = str(path.relative_to(data_dir.parent))
        df, fmt, rows, cols, success, error = read_table_sample(path)
        info = FileInfo(
            path=path,
            rel_path=rel_path,
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            file_size=path.stat().st_size,
            delimiter_or_format=fmt,
            number_of_rows=rows,
            number_of_columns=cols,
            read_success=success,
            read_error_if_any=error,
            columns=list(df.columns) if df is not None else [],
            sample_df=df,
        )
        info.guessed_dataset_name = guess_dataset_name(path, df, data_dir)
        info.guessed_content_type = guess_content_type(df, path)
        infos.append(info)
        logging.info("Scanned %s: %s", rel_path, "ok" if success else error)
    return infos


def write_file_inventory(infos: list[FileInfo], reports_dir: Path) -> None:
    rows = []
    for info in infos:
        rows.append(
            {
                "file_path": info.rel_path,
                "file_name": info.file_name,
                "file_type": info.file_type,
                "file_size": info.file_size,
                "guessed_dataset_name": info.guessed_dataset_name,
                "guessed_content_type": info.guessed_content_type,
                "delimiter_or_format": info.delimiter_or_format,
                "number_of_rows": info.number_of_rows,
                "number_of_columns": info.number_of_columns,
                "read_success": info.read_success,
                "read_error_if_any": info.read_error_if_any,
            }
        )
    pd.DataFrame(rows).to_csv(reports_dir / "file_inventory.csv", index=False)


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    lines = safe_read_lines(path)
    delimiter = infer_delimiter(path, lines)
    return pd.read_csv(
        path,
        sep=delimiter,
        comment="#",
        usecols=lambda c: c in set(columns),
        dtype=str,
        low_memory=False,
    )


def choose_single(candidates: list[str], preferred: list[str] | None = None) -> str | None:
    if not candidates:
        return None
    if preferred:
        lower = {c.lower(): c for c in candidates}
        for item in preferred:
            if item.lower() in lower:
                return lower[item.lower()]
    if len(candidates) == 1:
        return candidates[0]
    return None


def unique_values_from_infos(infos: list[FileInfo], field: str) -> set[str]:
    values: set[str] = set()
    preferred = {
        "patient_id": ["PATIENT_ID", "icgc_donor_id", "submitted_donor_id"],
        "sample_id": ["SAMPLE_ID", "Tumor_Sample_Barcode", "icgc_sample_id", "submitted_sample_id"],
        "gene": ["Hugo_Symbol", "gene_affected", "gene"],
    }
    for info in infos:
        if info.guessed_content_type in {"matrix", "metadata", "manifest", "documentation", "unknown"}:
            continue
        candidates = candidate_columns(info.columns, field)
        selected = choose_single(candidates, preferred.get(field))
        if selected is None:
            continue
        try:
            if info.file_size > 1024 * 1024 * 1024 and info.sample_df is not None and selected in info.sample_df.columns:
                df = info.sample_df[[selected]].copy()
                logging.info("Using sampled values for %s from very large file %s", field, info.rel_path)
            else:
                df = read_columns(info.path, [selected])
            series = df[selected].dropna().astype(str).str.strip()
            series = series[(series != "") & (~series.str.lower().isin(MISSING_LIKE))]
            values.update(series.unique().tolist())
        except Exception as exc:
            logging.warning("Could not count %s from %s: %s", field, info.rel_path, exc)
    return values


def best_missing_rate(infos: list[FileInfo], fields: list[str]) -> float | None:
    rates = []
    for info in infos:
        if not info.read_success:
            continue
        for field in fields:
            for col in candidate_columns(info.columns, field):
                try:
                    df = read_columns(info.path, [col])
                    if col in df.columns and len(df) > 0:
                        rates.append(1.0 - nonmissing_rate(df[col]))
                except Exception:
                    continue
    if not rates:
        return None
    return float(min(rates))


def duplicated_patient_rate(infos: list[FileInfo]) -> float | None:
    clinical_infos = [info for info in infos if info.guessed_content_type == "clinical"]
    for info in clinical_infos + infos:
        candidates = candidate_columns(info.columns, "patient_id")
        selected = choose_single(candidates, ["PATIENT_ID", "icgc_donor_id", "submitted_donor_id"])
        if selected is None:
            continue
        try:
            df = read_columns(info.path, [selected])
            values = df[selected].dropna().astype(str).str.strip()
            values = values[(values != "") & (~values.str.lower().isin(MISSING_LIKE))]
            if len(values) == 0:
                continue
            return float(1.0 - values.nunique() / len(values))
        except Exception:
            continue
    return None


def infer_dataset_feasibility(dataset_name: str, infos: list[FileInfo]) -> dict:
    readable = [info for info in infos if info.read_success]
    all_columns = []
    for info in readable:
        all_columns.extend(info.columns)

    mutation_infos = [i for i in readable if i.guessed_content_type == "mutation"]
    clinical_infos = [i for i in readable if i.guessed_content_type == "clinical"]
    cna_infos = [i for i in readable if i.guessed_content_type == "CNA"]

    patient_candidates = sorted(set(sum([candidate_columns(i.columns, "patient_id") for i in readable], [])))
    sample_candidates = sorted(set(sum([candidate_columns(i.columns, "sample_id") for i in readable], [])))
    gene_candidates = sorted(set(sum([candidate_columns(i.columns, "gene") for i in mutation_infos + cna_infos], [])))
    alteration_candidates = sorted(set(sum([candidate_columns(i.columns, "alteration_type") for i in mutation_infos + cna_infos], [])))
    stage_candidates = sorted(set(sum([candidate_columns(i.columns, "stage_raw") for i in clinical_infos], [])))
    metastasis_candidates = sorted(set(sum([candidate_columns(i.columns, "metastasis_status") for i in clinical_infos], [])))
    survival_time_candidates = sorted(set(sum([candidate_columns(i.columns, "survival_time") for i in clinical_infos], [])))
    survival_event_candidates = sorted(set(sum([candidate_columns(i.columns, "survival_event") for i in clinical_infos], [])))

    patients = unique_values_from_infos(readable, "patient_id")
    samples = unique_values_from_infos(readable, "sample_id")
    genes = unique_values_from_infos(mutation_infos + cna_infos, "gene")
    stage_missing = best_missing_rate(clinical_infos, ["stage_raw", "metastasis_status"])
    survival_missing = best_missing_rate(clinical_infos, ["survival_time"])
    dup_rate = duplicated_patient_rate(readable)

    has_mutation = bool(mutation_infos and (patient_candidates or sample_candidates) and gene_candidates)
    has_stage_or_metastasis = stage_missing is not None and stage_missing < 0.8
    sample_n = max(len(samples), len(patients))
    if not readable or not (patient_candidates or sample_candidates):
        level = "Not usable"
        recommended = "not recommended"
        reason = "No readable patient/sample identifier columns were found."
    elif not has_mutation:
        level = "Low" if clinical_infos else "Not usable"
        recommended = "not recommended"
        reason = "Clinical or auxiliary data are present, but no clear mutation/gene event table with IDs was found."
    elif has_stage_or_metastasis and sample_n >= 100:
        level = "High"
        recommended = "main cohort"
        reason = "Mutation/gene events, identifiers, and stage/metastasis information are available with adequate sample size."
    elif has_stage_or_metastasis:
        level = "Medium"
        recommended = "validation cohort"
        reason = "Mutation and stage/metastasis information are available, but the sample size is below the preferred main-cohort threshold."
    elif sample_n >= 50:
        level = "Medium"
        recommended = "only mutation matrix"
        reason = "Mutation and identifiers are available, but usable stage/metastasis information is missing or incomplete."
    else:
        level = "Low"
        recommended = "not recommended"
        reason = "Only partial mutation/clinical information is available or sample size is too small."

    if level == "High" and dataset_name != "AACR" and sample_n < 300:
        recommended = "validation cohort"

    return {
        "dataset_name": dataset_name,
        "available_files": "; ".join(sorted(i.rel_path for i in infos)),
        "sample_id_column_candidates": "; ".join(sorted(set(patient_candidates + sample_candidates))),
        "mutation_gene_column_candidates": "; ".join(gene_candidates),
        "mutation_status_column_candidates": "; ".join(alteration_candidates),
        "stage_column_candidates": "; ".join(stage_candidates),
        "metastasis_column_candidates": "; ".join(metastasis_candidates),
        "survival_time_column_candidates": "; ".join(survival_time_candidates),
        "survival_event_column_candidates": "; ".join(survival_event_candidates),
        "number_of_unique_patients": len(patients) if patients else np.nan,
        "number_of_unique_samples": len(samples) if samples else np.nan,
        "number_of_genes_or_events": len(genes) if genes else np.nan,
        "missing_rate_stage": stage_missing if stage_missing is not None else np.nan,
        "missing_rate_survival": survival_missing if survival_missing is not None else np.nan,
        "duplicated_patient_rate": dup_rate if dup_rate is not None else np.nan,
        "has_cna": bool(cna_infos),
        "feasibility_level": level,
        "reason": reason,
        "recommended_use": recommended,
    }


def write_dataset_reports(rows: list[dict], reports_dir: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "dataset_feasibility_table.csv", index=False)
    counts = df["feasibility_level"].value_counts().to_dict() if not df.empty else {}
    lines = [
        "# Dataset Feasibility Report",
        "",
        "This report is based on detected table headers and sampled file contents. It does not train MHN or compute R*.",
        "",
        "## Feasibility Counts",
        "",
    ]
    for level in ["High", "Medium", "Low", "Not usable"]:
        lines.append(f"- {level}: {counts.get(level, 0)}")
    lines.extend(["", "## Dataset Details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['dataset_name']}",
                "",
                f"- Feasibility: {row['feasibility_level']}",
                f"- Recommended use: {row['recommended_use']}",
                f"- Unique patients: {row['number_of_unique_patients']}",
                f"- Unique samples: {row['number_of_unique_samples']}",
                f"- Genes/events: {row['number_of_genes_or_events']}",
                f"- Stage/metastasis missing rate: {row['missing_rate_stage']}",
                f"- Survival missing rate: {row['missing_rate_survival']}",
                f"- Reason: {row['reason']}",
                f"- ID candidates: {row['sample_id_column_candidates']}",
                f"- Gene candidates: {row['mutation_gene_column_candidates']}",
                f"- Stage candidates: {row['stage_column_candidates']}",
                f"- Metastasis candidates: {row['metastasis_column_candidates']}",
                "",
            ]
        )
    (reports_dir / "dataset_feasibility_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_column_mapping(infos: list[FileInfo], reports_dir: Path) -> None:
    rows = []
    for info in infos:
        for field, _ in STANDARD_FIELDS:
            candidates = candidate_columns(info.columns, field)
            if not candidates:
                continue
            selected = choose_single(candidates)
            rows.append(
                {
                    "dataset_name": info.guessed_dataset_name,
                    "source_file": info.rel_path,
                    "standard_field": field,
                    "original_column_candidates": "; ".join(candidates),
                    "selected_column_if_unambiguous": selected or "",
                    "note": "" if selected else "Multiple or ambiguous candidates retained for review.",
                }
            )
    pd.DataFrame(rows).to_csv(reports_dir / "column_mapping_report.csv", index=False)


def write_data_dictionary(processed_dir: Path) -> None:
    rows = []
    for field, description in STANDARD_FIELDS:
        rows.append(
            {
                "standard_field": field,
                "description": description,
                "required_for": "Rel-ObsTQ-MHN input" if field in {"patient_id", "sample_id", "gene", "alteration_binary"} else "metadata/validation",
                "allowed_or_expected_values": allowed_values(field),
            }
        )
    pd.DataFrame(rows).to_csv(processed_dir / "data_dictionary.csv", index=False)


def allowed_values(field: str) -> str:
    if field == "stage_group":
        return "early; local_advanced; metastatic; primary; unknown"
    if field == "alteration_binary":
        return "0; 1"
    if field == "survival_event":
        return "0; 1; unknown"
    return ""


def standardize_stage(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in MISSING_LIKE:
        return "unknown"
    if "metast" in text or "distant" in text or re.search(r"\bm1\b", text):
        return "metastatic"
    if "primary" in text:
        return "primary"
    clean = text.replace("stage", "").replace("ajcc", "").strip()
    clean = re.sub(r"[^ivxabc0-9]+", "", clean)
    if clean in {"i", "ia", "ib", "1", "1a", "1b"}:
        return "early"
    if clean in {"ii", "iia", "iib", "iic", "iii", "iiia", "iiib", "iiic", "2", "2a", "2b", "2c", "3", "3a", "3b", "3c"}:
        return "local_advanced"
    if clean in {"iv", "iva", "ivb", "ivc", "4", "4a", "4b", "4c"}:
        return "metastatic"
    if re.search(r"t\d+n\d+m0", text):
        return "local_advanced"
    return "unknown"


def normalize_event_status(value: object) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in MISSING_LIKE:
        return ""
    if text in {"true", "deceased", "dead", "1", "yes"}:
        return "1"
    if text in {"false", "alive", "living", "0", "no"}:
        return "0"
    return ""


def load_hgnc_map(data_dir: Path) -> dict[str, str]:
    hgnc = data_dir / "hgnc_complete_set.txt"
    if not hgnc.exists():
        return {}
    try:
        df = pd.read_csv(hgnc, sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"])
        df = df.dropna()
        return dict(zip(df["ensembl_gene_id"].astype(str), df["symbol"].astype(str)))
    except Exception as exc:
        logging.warning("Could not read HGNC mapping: %s", exc)
        return {}


def find_info(infos: list[FileInfo], dataset: str, content_type: str, name_contains: str | None = None) -> FileInfo | None:
    candidates = [i for i in infos if i.guessed_dataset_name == dataset and i.guessed_content_type == content_type]
    if name_contains:
        contains = [i for i in candidates if name_contains.lower() in i.file_name.lower()]
        if contains:
            candidates = contains
    if not candidates:
        return None
    return sorted(candidates, key=lambda i: i.file_size, reverse=True)[0]


def standardize_aacr(infos: list[FileInfo], processed_dir: Path) -> tuple[Path, Path] | None:
    mutation_info = find_info(infos, "AACR", "mutation", "mutations")
    sample_info = None
    patient_info = None
    for info in infos:
        if info.guessed_dataset_name != "AACR" or info.guessed_content_type != "clinical":
            continue
        cols = {c.upper(): c for c in info.columns}
        if "SAMPLE_ID" in cols and "PATIENT_ID" in cols:
            sample_info = info
        elif "PATIENT_ID" in cols and "SAMPLE_ID" not in cols:
            patient_info = info
    if not mutation_info or not sample_info:
        logging.warning("AACR standardization skipped because mutation or sample clinical file is missing.")
        return None

    sample_cols = [
        c
        for c in [
            "PATIENT_ID",
            "SAMPLE_ID",
            "AGE_AT_SEQ_REPORT",
            "ONCOTREE_CODE",
            "SAMPLE_TYPE",
            "CANCER_TYPE",
            "CANCER_TYPE_DETAILED",
            "SAMPLE_TYPE_DETAILED",
        ]
        if c in sample_info.columns
    ]
    sample = read_columns(sample_info.path, sample_cols)
    patient = pd.DataFrame()
    if patient_info:
        patient_cols = [c for c in ["PATIENT_ID", "SEX", "INT_CONTACT", "INT_DOD", "DEAD"] if c in patient_info.columns]
        patient = read_columns(patient_info.path, patient_cols)
    clinical = sample.merge(patient, on="PATIENT_ID", how="left") if not patient.empty else sample.copy()
    clinical_out = pd.DataFrame(
        {
            "patient_id": clinical.get("PATIENT_ID", ""),
            "sample_id": clinical.get("SAMPLE_ID", ""),
            "cohort": "AACR",
            "cancer_type": clinical.get("CANCER_TYPE", clinical.get("ONCOTREE_CODE", "")),
            "stage_raw": clinical.get("SAMPLE_TYPE", ""),
            "stage_group": clinical.get("SAMPLE_TYPE", "").map(standardize_stage),
            "metastasis_status": clinical.get("SAMPLE_TYPE", ""),
            "survival_time": choose_survival_time(clinical),
            "survival_event": clinical.get("DEAD", "").map(normalize_event_status) if "DEAD" in clinical.columns else "",
            "age": clinical.get("AGE_AT_SEQ_REPORT", ""),
            "sex": clinical.get("SEX", ""),
            "source_file": sample_info.rel_path,
        }
    )
    clinical_path = processed_dir / "standardized_clinical.csv"
    clinical_out.to_csv(clinical_path, index=False)

    usecols = [c for c in ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "Variant_Type", "Mutation_Status"] if c in mutation_info.columns]
    mutation = read_columns(mutation_info.path, usecols)
    sample_map = clinical_out[["patient_id", "sample_id", "cohort", "cancer_type"]].drop_duplicates()
    mutation_out = pd.DataFrame(
        {
            "sample_id": mutation.get("Tumor_Sample_Barcode", ""),
            "gene": mutation.get("Hugo_Symbol", ""),
            "alteration_type": mutation.get("Variant_Classification", mutation.get("Variant_Type", "")),
            "alteration_binary": 1,
            "source_file": mutation_info.rel_path,
        }
    )
    mutation_out = mutation_out.merge(sample_map, on="sample_id", how="left")
    mutation_out["patient_id"] = mutation_out["patient_id"].fillna(mutation_out["sample_id"].astype(str).str.rsplit("-", n=1).str[0])
    mutation_out = mutation_out[
        ["patient_id", "sample_id", "cohort", "cancer_type", "gene", "alteration_type", "alteration_binary", "source_file"]
    ]
    mutation_out = mutation_out.dropna(subset=["gene"])
    mutation_out["gene"] = mutation_out["gene"].astype(str).str.strip()
    mutation_out = mutation_out[mutation_out["gene"] != ""].drop_duplicates()
    mutation_path = processed_dir / "standardized_mutations.csv"
    mutation_out.to_csv(mutation_path, index=False)
    return mutation_path, clinical_path


def choose_survival_time(clinical: pd.DataFrame) -> pd.Series:
    if "DEAD" in clinical.columns and "INT_DOD" in clinical.columns and "INT_CONTACT" in clinical.columns:
        dead = clinical["DEAD"].astype(str).str.lower().isin({"true", "1", "dead", "deceased"})
        return clinical["INT_DOD"].where(dead, clinical["INT_CONTACT"])
    for col in ["donor_survival_time", "survival_time", "INT_DOD", "INT_CONTACT"]:
        if col in clinical.columns:
            return clinical[col]
    return pd.Series([""] * len(clinical))


def standardize_icgc(dataset: str, infos: list[FileInfo], data_dir: Path, processed_dir: Path) -> tuple[Path, Path] | None:
    mutation_info = find_info(infos, dataset, "mutation", "simple_somatic")
    donor_info = None
    sample_info = None
    specimen_info = None
    for info in infos:
        if info.guessed_dataset_name != dataset:
            continue
        if info.file_name.lower().startswith("donor."):
            donor_info = info
        elif info.file_name.lower().startswith("sample."):
            sample_info = info
        elif info.file_name.lower().startswith("specimen."):
            specimen_info = info
    if not mutation_info or not donor_info:
        logging.warning("%s standardization skipped because mutation or donor file is missing.", dataset)
        return None

    hgnc_map = load_hgnc_map(data_dir)
    donor_cols = [
        c
        for c in [
            "icgc_donor_id",
            "project_code",
            "donor_sex",
            "donor_vital_status",
            "donor_age_at_diagnosis",
            "donor_tumour_stage_at_diagnosis",
            "donor_survival_time",
            "donor_interval_of_last_followup",
        ]
        if c in donor_info.columns
    ]
    donor = read_columns(donor_info.path, donor_cols)
    clinical = donor.copy()
    if sample_info:
        sample_cols = [c for c in ["icgc_sample_id", "icgc_specimen_id", "icgc_donor_id", "submitted_sample_id", "project_code"] if c in sample_info.columns]
        sample = read_columns(sample_info.path, sample_cols)
        clinical = sample.merge(donor, on=["icgc_donor_id", "project_code"], how="left")
    if specimen_info:
        specimen_cols = [
            c
            for c in [
                "icgc_specimen_id",
                "icgc_donor_id",
                "specimen_type",
                "tumour_stage",
                "tumour_stage_supplemental",
                "tumour_confirmed",
                "project_code",
            ]
            if c in specimen_info.columns
        ]
        specimen = read_columns(specimen_info.path, specimen_cols)
        merge_cols = [c for c in ["icgc_specimen_id", "icgc_donor_id", "project_code"] if c in clinical.columns and c in specimen.columns]
        if merge_cols:
            clinical = clinical.merge(specimen, on=merge_cols, how="left", suffixes=("", "_specimen"))
    stage_raw = clinical.get("tumour_stage", clinical.get("donor_tumour_stage_at_diagnosis", ""))
    if isinstance(stage_raw, str):
        stage_raw = pd.Series([stage_raw] * len(clinical))
    donor_stage = clinical.get("donor_tumour_stage_at_diagnosis", pd.Series([""] * len(clinical)))
    stage_raw = stage_raw.fillna("")
    stage_raw = stage_raw.where(stage_raw.astype(str).str.strip() != "", donor_stage)
    clinical_out = pd.DataFrame(
        {
            "patient_id": clinical.get("icgc_donor_id", ""),
            "sample_id": clinical.get("icgc_sample_id", clinical.get("icgc_specimen_id", "")),
            "cohort": dataset,
            "cancer_type": clinical.get("project_code", dataset),
            "stage_raw": stage_raw,
            "stage_group": stage_raw.map(standardize_stage),
            "metastasis_status": clinical.get("specimen_type", ""),
            "survival_time": clinical.get("donor_survival_time", clinical.get("donor_interval_of_last_followup", "")),
            "survival_event": clinical.get("donor_vital_status", "").map(normalize_event_status)
            if "donor_vital_status" in clinical.columns
            else "",
            "age": clinical.get("donor_age_at_diagnosis", ""),
            "sex": clinical.get("donor_sex", ""),
            "source_file": donor_info.rel_path,
        }
    )
    clinical_path = processed_dir / "standardized_clinical.csv"
    clinical_out.to_csv(clinical_path, index=False)

    mutation_cols = [
        c
        for c in [
            "icgc_donor_id",
            "icgc_sample_id",
            "project_code",
            "gene_affected",
            "consequence_type",
            "mutation_type",
            "icgc_mutation_id",
        ]
        if c in mutation_info.columns
    ]
    mutation = read_columns(mutation_info.path, mutation_cols)
    gene = mutation.get("gene_affected", "").map(lambda x: hgnc_map.get(str(x), str(x))) if "gene_affected" in mutation.columns else ""
    mutation_out = pd.DataFrame(
        {
            "patient_id": mutation.get("icgc_donor_id", ""),
            "sample_id": mutation.get("icgc_sample_id", ""),
            "cohort": dataset,
            "cancer_type": mutation.get("project_code", dataset),
            "gene": gene,
            "alteration_type": mutation.get("consequence_type", mutation.get("mutation_type", "")),
            "alteration_binary": 1,
            "source_file": mutation_info.rel_path,
        }
    )
    mutation_out = mutation_out.dropna(subset=["gene"])
    mutation_out["gene"] = mutation_out["gene"].astype(str).str.strip()
    mutation_out = mutation_out[mutation_out["gene"] != ""].drop_duplicates()
    mutation_path = processed_dir / "standardized_mutations.csv"
    mutation_out.to_csv(mutation_path, index=False)
    return mutation_path, clinical_path


def select_standardization_dataset(feasibility: pd.DataFrame) -> str | None:
    if feasibility.empty:
        return None
    ranked = feasibility.copy()
    level_rank = {"High": 3, "Medium": 2, "Low": 1, "Not usable": 0}
    ranked["_level_rank"] = ranked["feasibility_level"].map(level_rank).fillna(0)
    ranked["_patients"] = pd.to_numeric(ranked["number_of_unique_patients"], errors="coerce").fillna(0)
    ranked = ranked.sort_values(["_level_rank", "_patients"], ascending=[False, False])
    top = ranked.iloc[0]
    if top["_level_rank"] <= 0:
        return None
    return str(top["dataset_name"])


def write_final_summary(rows: list[dict], reports_dir: Path, standardized_dataset: str | None) -> None:
    df = pd.DataFrame(rows)
    high = df[df["feasibility_level"] == "High"]["dataset_name"].tolist() if not df.empty else []
    medium = df[df["feasibility_level"] == "Medium"]["dataset_name"].tolist() if not df.empty else []
    mhn_only = df[df["recommended_use"] == "only mutation matrix"]["dataset_name"].tolist() if not df.empty else []
    has_stage = bool((pd.to_numeric(df.get("missing_rate_stage", pd.Series(dtype=float)), errors="coerce") < 0.8).any()) if not df.empty else False
    survival_possible = bool((pd.to_numeric(df.get("missing_rate_survival", pd.Series(dtype=float)), errors="coerce") < 0.8).any()) if not df.empty else False
    cna_available = bool(df.get("has_cna", pd.Series(dtype=bool)).fillna(False).any()) if not df.empty else False
    min_ready = bool(high)
    lines = [
        "# Final Feasibility Summary",
        "",
        "1. Main cohort candidates: " + (", ".join(high) if high else "None"),
        "2. Validation cohort candidates: " + (", ".join([x for x in medium if x not in mhn_only]) if medium else "None"),
        "3. Mutation-matrix-only cohorts: " + (", ".join(mhn_only) if mhn_only else "None"),
        "4. Stage/metastasis information: " + ("Available in at least one dataset." if has_stage else "Insufficient or missing."),
        "5. Recommended top events: 15 for the first pass; rerun sensitivity at 10 and 20 if stable states are sparse.",
        "6. CNA recommendation: " + ("Available, but keep SNV/indel-only as the first pass and add CNA as sensitivity analysis." if cna_available else "No clear CNA input was detected."),
        "7. Survival validation: " + ("Possible as validation only; do not use survival as an external time anchor." if survival_possible else "Not recommended until survival fields are recovered or completed."),
        "8. Minimum Rel-ObsTQ-MHN input requirement: " + ("Satisfied by at least one dataset." if min_ready else "Not yet satisfied."),
        "9. If not satisfied: recover explicit stage/metastasis and patient/sample ID links, then rerun this audit.",
        "",
        f"Standardized files were generated from: {standardized_dataset or 'None'}",
        "",
        "No MHN training, R* calculation, survival modeling, or external data download was performed.",
        "",
    ]
    (reports_dir / "final_feasibility_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local tumor datasets.")
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument("--output-dir", default=".", help="Project root output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    reports_dir = root / "reports"
    processed_dir = root / "processed"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(root)

    logging.info("Starting dataset audit for %s", data_dir)
    infos = scan_data_files(data_dir)
    write_file_inventory(infos, reports_dir)
    write_column_mapping(infos, reports_dir)
    write_data_dictionary(processed_dir)

    by_dataset: dict[str, list[FileInfo]] = {}
    for info in infos:
        by_dataset.setdefault(info.guessed_dataset_name, []).append(info)
    rows = [infer_dataset_feasibility(name, group) for name, group in sorted(by_dataset.items())]
    write_dataset_reports(rows, reports_dir)

    feasibility = pd.DataFrame(rows)
    selected = select_standardization_dataset(feasibility)
    if selected == "AACR":
        standardize_aacr(infos, processed_dir)
    elif selected:
        standardize_icgc(selected, infos, data_dir, processed_dir)
    write_final_summary(rows, reports_dir, selected)

    readable = sum(1 for info in infos if info.read_success)
    counts = feasibility["feasibility_level"].value_counts().to_dict() if not feasibility.empty else {}
    print("Audit summary")
    print(f"data_files={len(infos)}")
    print(f"readable_files={readable}")
    print(
        "feasibility_counts="
        + ", ".join(f"{level}:{counts.get(level, 0)}" for level in ["High", "Medium", "Low", "Not usable"])
    )
    print(f"recommended_main_cohorts={', '.join(feasibility[feasibility['feasibility_level'] == 'High']['dataset_name'].tolist()) if not feasibility.empty else 'None'}")
    print(f"standardized_dataset={selected or 'None'}")


if __name__ == "__main__":
    main()
