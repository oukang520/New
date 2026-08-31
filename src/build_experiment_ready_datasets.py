"""Build experiment-ready inputs for Rel-ObsTQ-MHN.

Outputs one standardized package per dataset:

- analysis_metadata.csv
- mutations_long.csv
- event_matrix.csv
- mhn_training_matrix.csv
- mhn_row_index_map.csv
- event_frequency.csv
- state_table.csv
- state_occupancy.csv
- dataset_manifest.json
- qc_report.md

No MHN training and no R* calculation are performed here.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AACR_CODES = ["LUAD", "COAD", "IDC"]
ICGC_DATASETS = ["PACA-CA", "PBCA-US", "PEME-CA", "PRAD-CA", "PRAD-UK"]
MISSING = {"", "na", "nan", "none", "unknown", "not collected", "not reported", "not applicable", "not available"}
STAGE_PRIORITY = {"metastatic": 4, "local_advanced": 3, "early": 2, "primary": 1, "unknown": 0}

ICGC_FUNCTIONAL_PATTERNS = [
    "missense",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "start_retained",
    "frameshift",
    "frame_shift",
    "splice_acceptor",
    "splice_donor",
    "splice_region",
    "inframe",
    "protein_altering",
    "coding_sequence",
    "initiator_codon",
    "transcript_ablation",
]
ICGC_NONFUNCTIONAL_PATTERNS = [
    "synonymous",
    "intron",
    "intergenic",
    "upstream",
    "downstream",
    "utr",
    "regulatory",
    "tf_binding",
    "mature_mirna",
    "non_coding",
    "nmd_transcript",
]
MAF_EXCLUDE_CLASSES = {
    "silent",
    "intron",
    "igr",
    "3'utr",
    "5'utr",
    "3'flank",
    "5'flank",
    "rna",
    "lincrna",
    "targeted_region",
    "de_novo_start_inframe",
    "de_novo_start_outofframe",
}


@dataclass
class DatasetResult:
    dataset_name: str
    output_dir: str
    analysis_units: int
    patients: int
    mutation_rows: int
    mutated_units: int
    genes_before_filtering: int
    events_retained: int
    stable_states: int
    rare_states: int
    stage_missing_rate: float
    survival_missing_rate: float
    zero_event_fraction: float
    mhn_ready: bool
    relobstq_ready: bool
    all_checks_passed: bool
    warnings: str


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "build_experiment_ready_datasets.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def is_missing(value: object) -> bool:
    return safe_text(value).lower() in MISSING


def nonmissing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    values = series.fillna("").astype(str).str.strip().str.lower()
    return float((~values.isin(MISSING)).mean())


def sanitize_event_name(value: object) -> str:
    text = safe_text(value).upper()
    text = re.sub(r"[^A-Z0-9_.:-]+", "_", text)
    text = text.strip("_")
    return text


def standardize_stage_group(stage_value: object, metastasis_value: object = "") -> str:
    text = f"{safe_text(stage_value)} {safe_text(metastasis_value)}".strip().lower()
    if not text or text in MISSING:
        return "unknown"
    if text in {"early", "local_advanced", "primary", "metastatic"}:
        return text
    if "metast" in text or "distant" in text or re.search(r"\bm1\b", text):
        return "metastatic"
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
    if "primary" in text or "tumour" in text or "tumor" in text:
        return "primary"
    return "unknown"


def normalize_survival_event(value: object) -> str:
    text = safe_text(value).lower()
    if text in {"true", "1", "deceased", "dead", "yes"}:
        return "1"
    if text in {"false", "0", "alive", "living", "no"}:
        return "0"
    return ""


def collapse_stage(stages: pd.Series) -> str:
    clean = stages.fillna("unknown").astype(str).map(standardize_stage_group)
    ranked = sorted(clean.unique(), key=lambda x: STAGE_PRIORITY.get(x, 0), reverse=True)
    return ranked[0] if ranked else "unknown"


def first_nonmissing(series: pd.Series) -> str:
    for value in series:
        text = safe_text(value)
        if text.lower() not in MISSING:
            return text
    return ""


def join_unique(series: pd.Series, limit: int = 30) -> str:
    values = []
    seen = set()
    for value in series.dropna().astype(str):
        text = value.strip()
        if text and text.lower() not in MISSING and text not in seen:
            values.append(text)
            seen.add(text)
        if len(values) >= limit:
            break
    return ";".join(values)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False, **kwargs)


def load_hgnc_map(data_dir: Path) -> dict[str, str]:
    path = data_dir / "hgnc_complete_set.txt"
    if not path.exists():
        logging.warning("HGNC map not found at %s", path)
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"], low_memory=False)
    df = df.dropna(subset=["symbol", "ensembl_gene_id"])
    return dict(zip(df["ensembl_gene_id"].astype(str), df["symbol"].astype(str)))


def is_functional_icgc(consequence: object) -> bool:
    text = safe_text(consequence).lower()
    if not text:
        return False
    if any(pattern in text for pattern in ICGC_FUNCTIONAL_PATTERNS):
        return True
    if any(pattern in text for pattern in ICGC_NONFUNCTIONAL_PATTERNS):
        return False
    return False


def is_functional_maf(alteration_type: object) -> bool:
    text = safe_text(alteration_type).lower()
    if not text:
        return True
    return text not in MAF_EXCLUDE_CLASSES


def find_icgc_dir(data_dir: Path, dataset_name: str) -> Path:
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith(f"[{dataset_name}]"):
            return item
    raise FileNotFoundError(f"Could not find ICGC dataset directory for {dataset_name}")


def make_analysis_metadata_template(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "analysis_id",
        "patient_id",
        "sample_id",
        "cohort",
        "dataset_name",
        "cancer_code",
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
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df[required].copy()


def load_aacr_subset(dataset_name: str, processed_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = dataset_name.replace("AACR_", "")
    subset_dir = processed_root / "aacr_oncotree_subsets" / code
    if not subset_dir.exists():
        raise FileNotFoundError(f"AACR subset directory not found: {subset_dir}")
    meta = read_csv(subset_dir / "clinical_samples.csv")
    muts = read_csv(subset_dir / "mutations.csv")
    meta = meta.rename(columns={"oncotree_code": "cancer_code"})
    meta["analysis_id"] = meta["sample_id"]
    meta["dataset_name"] = dataset_name
    meta["cancer_code"] = code
    meta = make_analysis_metadata_template(meta)
    muts = muts.copy()
    muts["analysis_id"] = muts["sample_id"]
    muts["dataset_name"] = dataset_name
    muts["cancer_code"] = code
    muts["gene_original"] = muts["gene"]
    muts["gene"] = muts["gene"].map(sanitize_event_name)
    muts["consequence"] = muts.get("alteration_type", "")
    muts = muts[muts["alteration_type"].map(is_functional_maf)].copy()
    return meta, standardize_mutation_columns(muts)


def load_icgc_metadata(dataset_name: str, data_dir: Path) -> pd.DataFrame:
    d = find_icgc_dir(data_dir, dataset_name)
    donor_path = d / f"donor.{dataset_name}.tsv"
    sample_path = d / f"sample.{dataset_name}.tsv"
    specimen_path = d / f"specimen.{dataset_name}.tsv"

    donor_cols = [
        "icgc_donor_id",
        "project_code",
        "submitted_donor_id",
        "donor_sex",
        "donor_vital_status",
        "donor_age_at_diagnosis",
        "donor_tumour_stage_at_diagnosis",
        "donor_survival_time",
        "donor_interval_of_last_followup",
    ]
    sample_cols = [
        "icgc_sample_id",
        "project_code",
        "submitted_sample_id",
        "icgc_specimen_id",
        "submitted_specimen_id",
        "icgc_donor_id",
        "submitted_donor_id",
    ]
    specimen_cols = [
        "icgc_specimen_id",
        "project_code",
        "submitted_specimen_id",
        "icgc_donor_id",
        "submitted_donor_id",
        "specimen_type",
        "tumour_confirmed",
        "tumour_histological_type",
        "tumour_stage",
        "tumour_stage_supplemental",
        "percentage_cellularity",
    ]
    donor = read_tsv(donor_path, usecols=lambda c: c in set(donor_cols))
    sample = read_tsv(sample_path, usecols=lambda c: c in set(sample_cols))
    specimen = read_tsv(specimen_path, usecols=lambda c: c in set(specimen_cols))

    meta = sample.merge(specimen, on=["icgc_specimen_id", "project_code", "submitted_specimen_id", "icgc_donor_id", "submitted_donor_id"], how="left")
    meta = meta.merge(donor, on=["icgc_donor_id", "project_code", "submitted_donor_id"], how="left")

    specimen_type = meta.get("specimen_type", pd.Series([""] * len(meta))).fillna("").astype(str)
    is_tumor = specimen_type.str.lower().str.contains("tumour|tumor|metast", regex=True) & ~specimen_type.str.lower().str.contains("normal", regex=False)
    meta = meta[is_tumor].copy()

    stage_raw = meta.get("tumour_stage", pd.Series([""] * len(meta))).fillna("")
    donor_stage = meta.get("donor_tumour_stage_at_diagnosis", pd.Series([""] * len(meta))).fillna("")
    stage_raw = stage_raw.where(stage_raw.astype(str).str.strip() != "", donor_stage)
    meta["analysis_id"] = meta["icgc_sample_id"]
    meta["patient_id"] = meta["icgc_donor_id"]
    meta["sample_id"] = meta["icgc_sample_id"]
    meta["cohort"] = dataset_name
    meta["dataset_name"] = dataset_name
    meta["cancer_code"] = dataset_name
    meta["cancer_type"] = meta["project_code"]
    meta["cancer_type_detailed"] = meta.get("tumour_histological_type", "")
    meta["stage_raw"] = stage_raw
    meta["stage_group"] = [standardize_stage_group(s, t) for s, t in zip(meta["stage_raw"], meta.get("specimen_type", ""))]
    meta["metastasis_status"] = meta.get("specimen_type", "")
    meta["survival_time"] = meta.get("donor_survival_time", "")
    if "donor_interval_of_last_followup" in meta.columns:
        meta["survival_time"] = meta["survival_time"].where(meta["survival_time"].fillna("").astype(str).str.strip() != "", meta["donor_interval_of_last_followup"])
    meta["survival_event"] = meta.get("donor_vital_status", "").map(normalize_survival_event)
    meta["age"] = meta.get("donor_age_at_diagnosis", "")
    meta["sex"] = meta.get("donor_sex", "")
    meta["source_file"] = str(sample_path)
    meta = make_analysis_metadata_template(meta)
    meta = meta.drop_duplicates(subset=["analysis_id"])
    return meta


def process_icgc_mutations(dataset_name: str, data_dir: Path, hgnc_map: dict[str, str], chunksize: int = 250_000) -> tuple[pd.DataFrame, dict]:
    d = find_icgc_dir(data_dir, dataset_name)
    path = d / f"simple_somatic_mutation.open.{dataset_name}.tsv"
    usecols = [
        "icgc_mutation_id",
        "icgc_donor_id",
        "project_code",
        "icgc_specimen_id",
        "icgc_sample_id",
        "submitted_sample_id",
        "mutation_type",
        "consequence_type",
        "gene_affected",
    ]
    rows = []
    total_rows = 0
    functional_rows = 0
    unmapped = set()
    for chunk in pd.read_csv(path, sep="\t", dtype=str, usecols=lambda c: c in set(usecols), chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        chunk["consequence_type"] = chunk.get("consequence_type", "").fillna("")
        chunk = chunk[chunk["consequence_type"].map(is_functional_icgc)].copy()
        functional_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["gene_original"] = chunk["gene_affected"].fillna("").astype(str)
        chunk["gene"] = chunk["gene_original"].map(lambda x: hgnc_map.get(x, x))
        missing_map = chunk.loc[~chunk["gene_original"].isin(hgnc_map), "gene_original"]
        unmapped.update([x for x in missing_map.unique().tolist() if x])
        chunk["gene"] = chunk["gene"].map(sanitize_event_name)
        chunk = chunk[chunk["gene"] != ""].copy()
        out = pd.DataFrame(
            {
                "analysis_id": chunk["icgc_sample_id"],
                "patient_id": chunk["icgc_donor_id"],
                "sample_id": chunk["icgc_sample_id"],
                "cohort": dataset_name,
                "dataset_name": dataset_name,
                "cancer_code": dataset_name,
                "cancer_type": chunk["project_code"],
                "cancer_type_detailed": "",
                "gene": chunk["gene"],
                "gene_original": chunk["gene_original"],
                "alteration_type": chunk["mutation_type"],
                "consequence": chunk["consequence_type"],
                "alteration_binary": 1,
                "mutation_id": chunk["icgc_mutation_id"],
                "source_file": str(path),
            }
        )
        rows.append(out.drop_duplicates())
    mutations = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not mutations.empty:
        mutations = standardize_mutation_columns(mutations).drop_duplicates(
            subset=["analysis_id", "patient_id", "sample_id", "gene", "mutation_id", "consequence"]
        )
    stats = {
        "raw_mutation_rows": total_rows,
        "functional_mutation_rows": functional_rows,
        "unmapped_gene_ids": len(unmapped),
    }
    return mutations, stats


def standardize_mutation_columns(mutations: pd.DataFrame) -> pd.DataFrame:
    required = [
        "analysis_id",
        "patient_id",
        "sample_id",
        "cohort",
        "dataset_name",
        "cancer_code",
        "cancer_type",
        "cancer_type_detailed",
        "gene",
        "gene_original",
        "alteration_type",
        "consequence",
        "alteration_binary",
        "mutation_id",
        "source_file",
    ]
    for col in required:
        if col not in mutations.columns:
            mutations[col] = ""
    mutations["analysis_id"] = mutations["analysis_id"].astype(str).str.strip()
    mutations["patient_id"] = mutations["patient_id"].astype(str).str.strip()
    mutations["sample_id"] = mutations["sample_id"].astype(str).str.strip()
    mutations["gene"] = mutations["gene"].map(sanitize_event_name)
    mutations["alteration_binary"] = 1
    return mutations[required].copy()


def add_mutation_samples_to_metadata(metadata: pd.DataFrame, mutations: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if mutations.empty:
        return metadata
    missing_ids = sorted(set(mutations["analysis_id"]) - set(metadata["analysis_id"]))
    if not missing_ids:
        return metadata
    add = mutations[mutations["analysis_id"].isin(missing_ids)].drop_duplicates("analysis_id").copy()
    add_meta = pd.DataFrame(
        {
            "analysis_id": add["analysis_id"],
            "patient_id": add["patient_id"],
            "sample_id": add["sample_id"],
            "cohort": dataset_name,
            "dataset_name": dataset_name,
            "cancer_code": dataset_name,
            "cancer_type": add.get("cancer_type", dataset_name),
            "cancer_type_detailed": add.get("cancer_type_detailed", ""),
            "stage_raw": "",
            "stage_group": "unknown",
            "metastasis_status": "",
            "survival_time": "",
            "survival_event": "",
            "age": "",
            "sex": "",
            "source_file": add.get("source_file", ""),
        }
    )
    return pd.concat([metadata, add_meta], ignore_index=True).drop_duplicates("analysis_id")


def select_events(mutations: pd.DataFrame, metadata: pd.DataFrame, min_frequency: float, top_k: int | None) -> tuple[list[str], pd.DataFrame]:
    n = metadata["analysis_id"].nunique()
    if n == 0 or mutations.empty:
        return [], pd.DataFrame(columns=["event", "sample_count", "frequency", "selected"])
    support = mutations.drop_duplicates(["analysis_id", "gene"]).groupby("gene")["analysis_id"].nunique().sort_values(ascending=False)
    freq = support / n
    candidates = freq[freq >= min_frequency]
    if candidates.empty:
        candidates = freq
    if top_k is None:
        top_k = min(25, max(10, int((freq >= min_frequency).sum()))) if len(freq) else 0
    top_k = min(max(top_k, 0), 25)
    selected = list(candidates.head(top_k).index)
    if len(selected) < min(10, len(freq)):
        selected = list(freq.head(min(10, len(freq))).index)
    event_frequency = pd.DataFrame(
        {
            "event": support.index,
            "sample_count": support.values.astype(int),
            "frequency": freq.loc[support.index].round(6).values,
        }
    )
    event_frequency["selected"] = event_frequency["event"].isin(selected)
    return selected, event_frequency


def build_event_matrix(metadata: pd.DataFrame, mutations: pd.DataFrame, events: list[str]) -> pd.DataFrame:
    ids = metadata["analysis_id"].drop_duplicates().astype(str)
    matrix = pd.DataFrame({"analysis_id": ids})
    for event in events:
        matrix[event] = 0
    if events and not mutations.empty:
        work = mutations[mutations["gene"].isin(events)][["analysis_id", "gene"]].drop_duplicates()
        work["_value"] = 1
        pivot = work.pivot_table(index="analysis_id", columns="gene", values="_value", aggfunc="max", fill_value=0)
        pivot = pivot.reindex(index=ids, columns=events, fill_value=0).astype(int).reset_index()
        matrix = pivot
    return matrix


def make_genotype_signature(row: pd.Series, events: list[str]) -> str:
    active = sorted([event for event in events if int(row[event]) == 1])
    return "+".join(active) if active else "WT"


def build_state_tables(metadata: pd.DataFrame, event_matrix: pd.DataFrame, events: list[str], min_state_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = metadata.merge(event_matrix, on="analysis_id", how="left")
    for event in events:
        merged[event] = pd.to_numeric(merged[event], errors="coerce").fillna(0).astype(int)
    merged["event_count"] = merged[events].sum(axis=1).astype(int) if events else 0
    merged["genotype_signature"] = merged.apply(lambda row: make_genotype_signature(row, events), axis=1)
    merged["stage_group"] = merged["stage_group"].fillna("unknown").astype(str).map(lambda x: standardize_stage_group(x))
    merged["state_id"] = merged["stage_group"] + "::" + merged["genotype_signature"]
    counts = merged["state_id"].value_counts()
    merged["state_count"] = merged["state_id"].map(counts).astype(int)
    merged["state_count_flag"] = np.where(merged["state_count"] >= min_state_count, "valid_state", "rare_state")
    merged["usable_for_mhn"] = len(events) >= 1
    has_stage = ~merged["stage_group"].str.lower().isin(MISSING)
    merged["usable_for_relobstq"] = has_stage & (merged["state_count_flag"] == "valid_state")
    state_cols = [
        "analysis_id",
        "patient_id",
        "sample_id",
        "cohort",
        "dataset_name",
        "cancer_code",
        "cancer_type",
        "cancer_type_detailed",
        "stage_raw",
        "stage_group",
        "metastasis_status",
        "genotype_signature",
        "event_count",
        "state_id",
        "state_count",
        "state_count_flag",
        "usable_for_mhn",
        "usable_for_relobstq",
        "survival_time",
        "survival_event",
        "age",
        "sex",
    ]
    occupancy = (
        merged.groupby(["stage_group", "genotype_signature", "state_id", "state_count_flag"], dropna=False)
        .size()
        .reset_index(name="state_count")
        .sort_values(["state_count", "state_id"], ascending=[False, True])
    )
    occupancy["occupancy_fraction"] = occupancy["state_count"] / len(merged) if len(merged) else 0
    return merged[state_cols], occupancy


def validate_dataset(
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    event_matrix: pd.DataFrame,
    mhn_matrix: pd.DataFrame,
    state_table: pd.DataFrame,
    events: list[str],
) -> tuple[dict, list[str]]:
    checks = {
        "unique_analysis_id": metadata["analysis_id"].is_unique,
        "event_matrix_rows_match_metadata": len(event_matrix) == len(metadata),
        "mhn_matrix_rows_match_metadata": len(mhn_matrix) == len(metadata),
        "state_table_rows_match_metadata": len(state_table) == len(metadata),
        "mhn_matrix_has_no_id_columns": not any(c.lower().endswith("_id") or c == "analysis_id" for c in mhn_matrix.columns),
        "mhn_matrix_binary": bool(((mhn_matrix.fillna(0).astype(int).isin([0, 1])).all()).all()) if not mhn_matrix.empty else False,
        "event_columns_match": list(mhn_matrix.columns) == events,
        "mutation_analysis_ids_in_metadata": set(mutations["analysis_id"]).issubset(set(metadata["analysis_id"])) if not mutations.empty else True,
        "no_duplicate_event_columns": len(events) == len(set(events)),
    }
    warnings = []
    if len(events) < 10:
        warnings.append("Fewer than 10 retained events; not ideal for MHN main experiments.")
    if len(events) > 25:
        warnings.append("More than 25 retained events; this should not happen with default top-k filtering.")
    if len(metadata) < 300:
        warnings.append("Fewer than 300 analysis units; not recommended as a primary MHN/Rel-ObsTQ cohort.")
    zero_event_fraction = float((state_table["event_count"].astype(int) == 0).mean()) if len(state_table) else 1.0
    if zero_event_fraction > 0.5:
        warnings.append("More than 50% zero-event analysis units after selected-event filtering.")
    if (state_table["stage_group"] == "unknown").mean() > 0.3:
        warnings.append("More than 30% unknown stage/progression states.")
    non_unknown_stage_groups = sorted(set(state_table.loc[state_table["stage_group"] != "unknown", "stage_group"]))
    if len(non_unknown_stage_groups) < 2:
        warnings.append("Fewer than two non-unknown stage/progression groups; stage-specific Rel-ObsTQ is limited.")
    if state_table["usable_for_relobstq"].sum() < 100:
        warnings.append("Fewer than 100 Rel-ObsTQ-usable analysis units.")
    return checks, warnings


def write_qc_report(
    output_dir: Path,
    dataset_name: str,
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    event_frequency: pd.DataFrame,
    state_table: pd.DataFrame,
    state_occupancy: pd.DataFrame,
    checks: dict,
    warnings: list[str],
    source_stats: dict,
) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 15) -> str:
        shown = df.head(max_rows)
        if shown.empty:
            return "(none)"
        cols = list(shown.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in shown.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    selected_freq = event_frequency[event_frequency["selected"]].copy() if "selected" in event_frequency else event_frequency
    lines = [
        f"# Experiment-Ready QC: {dataset_name}",
        "",
        "## Source And Processing",
        "",
        f"- Analysis unit: sample-level",
        f"- Functional mutation filtering: enabled",
        f"- Raw mutation rows: {source_stats.get('raw_mutation_rows', 'NA')}",
        f"- Functional mutation rows before event collapse: {source_stats.get('functional_mutation_rows', 'NA')}",
        f"- Unmapped gene IDs retained as original IDs: {source_stats.get('unmapped_gene_ids', 0)}",
        "",
        "## Dataset Counts",
        "",
        f"- Analysis units: {len(metadata)}",
        f"- Unique patients: {metadata['patient_id'].nunique()}",
        f"- Mutation rows after filtering/deduplication: {len(mutations)}",
        f"- Mutated analysis units: {mutations['analysis_id'].nunique() if not mutations.empty else 0}",
        f"- Genes/events before top-k filtering: {mutations['gene'].nunique() if not mutations.empty else 0}",
        f"- Retained events: {int(selected_freq.shape[0])}",
        f"- Zero-event analysis-unit fraction: {(state_table['event_count'] == 0).mean():.4f}",
        f"- Stage/progression missing rate: {(state_table['stage_group'] == 'unknown').mean():.4f}",
        f"- Survival-time missing rate: {1.0 - nonmissing_rate(metadata['survival_time']):.4f}",
        f"- Valid states: {(state_table['state_count_flag'] == 'valid_state').sum()} analysis units across {(state_occupancy['state_count_flag'] == 'valid_state').sum()} states",
        "",
        "## Compatibility Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Selected Events",
            "",
            md_table(selected_freq[["event", "sample_count", "frequency"]]),
            "",
            "## Top States",
            "",
            md_table(state_occupancy),
            "",
        ]
    )
    (output_dir / "qc_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    output_dir: Path,
    dataset_name: str,
    events: list[str],
    checks: dict,
    warnings: list[str],
    source_stats: dict,
    counts: dict,
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "analysis_unit": "sample",
        "files": {
            "analysis_metadata": "analysis_metadata.csv",
            "mutations_long": "mutations_long.csv",
            "event_matrix": "event_matrix.csv",
            "mhn_training_matrix": "mhn_training_matrix.csv",
            "mhn_row_index_map": "mhn_row_index_map.csv",
            "event_frequency": "event_frequency.csv",
            "state_table": "state_table.csv",
            "state_occupancy": "state_occupancy.csv",
            "qc_report": "qc_report.md",
        },
        "events": events,
        "counts": counts,
        "checks": checks,
        "warnings": warnings,
        "source_stats": source_stats,
        "notes": [
            "mhn_training_matrix.csv is pure binary event data with no ID or clinical columns.",
            "mhn_row_index_map.csv maps MHN matrix row order back to analysis metadata and states.",
            "No MHN training or R* calculation was performed during preprocessing.",
        ],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def process_dataset(
    dataset_name: str,
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    output_root: Path,
    min_frequency: float,
    top_k: int | None,
    min_state_count: int,
    source_stats: dict | None = None,
) -> DatasetResult:
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stats = source_stats or {}

    metadata = make_analysis_metadata_template(metadata).drop_duplicates("analysis_id").copy()
    mutations = standardize_mutation_columns(mutations) if not mutations.empty else pd.DataFrame(columns=standardize_mutation_columns(pd.DataFrame()).columns)
    mutations = mutations[mutations["analysis_id"].isin(set(metadata["analysis_id"]))].copy()
    mutations = mutations.drop_duplicates(subset=["analysis_id", "gene"])

    events, event_frequency = select_events(mutations, metadata, min_frequency=min_frequency, top_k=top_k)
    event_matrix = build_event_matrix(metadata, mutations, events)
    mhn_matrix = event_matrix[events].copy().astype(int)
    state_table, state_occupancy = build_state_tables(metadata, event_matrix, events, min_state_count=min_state_count)
    row_map = state_table[
        [
            "analysis_id",
            "patient_id",
            "sample_id",
            "dataset_name",
            "cancer_code",
            "stage_group",
            "genotype_signature",
            "event_count",
            "state_id",
            "state_count_flag",
            "usable_for_mhn",
            "usable_for_relobstq",
        ]
    ].copy()
    row_map.insert(0, "row_index", range(len(row_map)))

    checks, warnings = validate_dataset(metadata, mutations, event_matrix, mhn_matrix, state_table, events)
    counts = {
        "analysis_units": int(len(metadata)),
        "patients": int(metadata["patient_id"].nunique()),
        "mutation_rows": int(len(mutations)),
        "mutated_units": int(mutations["analysis_id"].nunique() if not mutations.empty else 0),
        "genes_before_filtering": int(mutations["gene"].nunique() if not mutations.empty else 0),
        "events_retained": int(len(events)),
        "valid_state_count": int((state_occupancy["state_count_flag"] == "valid_state").sum() if not state_occupancy.empty else 0),
    }

    metadata.to_csv(output_dir / "analysis_metadata.csv", index=False)
    mutations.to_csv(output_dir / "mutations_long.csv", index=False)
    event_matrix.to_csv(output_dir / "event_matrix.csv", index=False)
    mhn_matrix.to_csv(output_dir / "mhn_training_matrix.csv", index=False)
    row_map.to_csv(output_dir / "mhn_row_index_map.csv", index=False)
    event_frequency.to_csv(output_dir / "event_frequency.csv", index=False)
    state_table.to_csv(output_dir / "state_table.csv", index=False)
    state_occupancy.to_csv(output_dir / "state_occupancy.csv", index=False)
    write_qc_report(output_dir, dataset_name, metadata, mutations, event_frequency, state_table, state_occupancy, checks, warnings, source_stats)
    write_manifest(output_dir, dataset_name, events, checks, warnings, source_stats, counts)

    stable_states = int((state_occupancy["state_count_flag"] == "valid_state").sum() if not state_occupancy.empty else 0)
    rare_states = int((state_occupancy["state_count_flag"] == "rare_state").sum() if not state_occupancy.empty else 0)
    mhn_ready = bool(10 <= len(events) <= 25 and len(metadata) >= 300 and checks["mhn_matrix_binary"])
    non_unknown_stage_groups = sorted(set(state_table.loc[state_table["stage_group"] != "unknown", "stage_group"]))
    relobstq_ready = bool(
        mhn_ready
        and (state_table["usable_for_relobstq"].sum() >= 100)
        and (state_table["stage_group"] == "unknown").mean() <= 0.3
        and len(non_unknown_stage_groups) >= 2
    )
    return DatasetResult(
        dataset_name=dataset_name,
        output_dir=str(output_dir),
        analysis_units=int(len(metadata)),
        patients=int(metadata["patient_id"].nunique()),
        mutation_rows=int(len(mutations)),
        mutated_units=int(mutations["analysis_id"].nunique() if not mutations.empty else 0),
        genes_before_filtering=int(mutations["gene"].nunique() if not mutations.empty else 0),
        events_retained=int(len(events)),
        stable_states=stable_states,
        rare_states=rare_states,
        stage_missing_rate=float((state_table["stage_group"] == "unknown").mean()),
        survival_missing_rate=float(1.0 - nonmissing_rate(metadata["survival_time"])),
        zero_event_fraction=float((state_table["event_count"] == 0).mean()),
        mhn_ready=mhn_ready,
        relobstq_ready=relobstq_ready,
        all_checks_passed=bool(all(checks.values())),
        warnings="; ".join(warnings),
    )


def write_schema(output_root: Path) -> None:
    schema = {
        "analysis_metadata.csv": [
            "analysis_id",
            "patient_id",
            "sample_id",
            "cohort",
            "dataset_name",
            "cancer_code",
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
        ],
        "mutations_long.csv": [
            "analysis_id",
            "patient_id",
            "sample_id",
            "cohort",
            "dataset_name",
            "cancer_code",
            "cancer_type",
            "cancer_type_detailed",
            "gene",
            "gene_original",
            "alteration_type",
            "consequence",
            "alteration_binary",
            "mutation_id",
            "source_file",
        ],
        "event_matrix.csv": "analysis_id plus selected binary event columns",
        "mhn_training_matrix.csv": "selected binary event columns only; no ID or clinical columns",
        "state_table.csv": "sample metadata plus genotype_signature, event_count, state_id, and usability flags",
    }
    (output_root / "experiment_ready_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_summary_report(results: list[DatasetResult], output_root: Path) -> None:
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(output_root / "experiment_ready_summary.csv", index=False)
    cols = [
        "dataset_name",
        "analysis_units",
        "patients",
        "events_retained",
        "stable_states",
        "stage_missing_rate",
        "zero_event_fraction",
        "mhn_ready",
        "relobstq_ready",
        "all_checks_passed",
    ]
    shown = df[cols].copy()
    lines = [
        "# Experiment-Ready Dataset Summary",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    lines.extend(
        [
            "",
            "All datasets use sample-level analysis units and include a pure `mhn_training_matrix.csv` for MHN.",
            "Datasets with `relobstq_ready=False` may still be usable for mutation-only MHN sensitivity analyses but should not be used as primary Rel-ObsTQ cohorts without review.",
            "",
        ]
    )
    (output_root / "experiment_ready_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment-ready Rel-ObsTQ-MHN datasets.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--output-dir", default="processed/experiment_ready")
    parser.add_argument("--min-frequency", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=0, help="0 means auto-select 10-25 events.")
    parser.add_argument("--min-state-count", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=250000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(".").resolve()
    data_dir = Path(args.data_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(project_root)
    write_schema(output_root)

    top_k = None if args.top_k == 0 else args.top_k
    results: list[DatasetResult] = []

    for code in AACR_CODES:
        dataset_name = f"AACR_{code}"
        logging.info("Processing %s", dataset_name)
        metadata, mutations = load_aacr_subset(dataset_name, processed_dir)
        result = process_dataset(
            dataset_name=dataset_name,
            metadata=metadata,
            mutations=mutations,
            output_root=output_root,
            min_frequency=args.min_frequency,
            top_k=top_k,
            min_state_count=args.min_state_count,
            source_stats={"raw_mutation_rows": "from pre-extracted AACR subset", "functional_mutation_rows": len(mutations), "unmapped_gene_ids": 0},
        )
        results.append(result)

    hgnc_map = load_hgnc_map(data_dir)
    for dataset_name in ICGC_DATASETS:
        logging.info("Processing %s", dataset_name)
        metadata = load_icgc_metadata(dataset_name, data_dir)
        mutations, stats = process_icgc_mutations(dataset_name, data_dir, hgnc_map, chunksize=args.chunksize)
        metadata = add_mutation_samples_to_metadata(metadata, mutations, dataset_name)
        result = process_dataset(
            dataset_name=dataset_name,
            metadata=metadata,
            mutations=mutations,
            output_root=output_root,
            min_frequency=args.min_frequency,
            top_k=top_k,
            min_state_count=args.min_state_count,
            source_stats=stats,
        )
        results.append(result)

    write_summary_report(results, output_root)
    print("Experiment-ready processing summary")
    for r in results:
        print(
            f"{r.dataset_name}: units={r.analysis_units}, events={r.events_retained}, "
            f"states={r.stable_states + r.rare_states}, checks={r.all_checks_passed}, "
            f"MHN={r.mhn_ready}, RelObsTQ={r.relobstq_ready}"
        )


if __name__ == "__main__":
    main()
