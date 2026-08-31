"""Build a binary event matrix for Rel-ObsTQ-MHN inputs.

The script expects a standardized mutation table. It can optionally merge a
standardized clinical table so samples without detected alterations are kept as
zero rows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_META_COLUMNS = [
    "patient_id",
    "sample_id",
    "cohort",
    "cancer_type",
    "stage_raw",
    "stage_group",
    "metastasis_status",
    "survival_time",
    "survival_event",
    "age",
    "sex",
    "source_file",
]


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "build_event_matrix.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=int)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).gt(0).astype(int)
    values = series.fillna("").astype(str).str.strip().str.lower()
    false_values = {"", "0", "false", "no", "na", "nan", "none", "not applicable"}
    return (~values.isin(false_values)).astype(int)


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def simplify_cna_type(value: str) -> str:
    text = str(value).strip().lower()
    if "ampl" in text or "gain" in text or text in {"2", "amp"}:
        return "AMP"
    if "deep" in text and "del" in text:
        return "DEL"
    if "loss" in text or "delet" in text or text in {"-2", "homdel"}:
        return "DEL"
    if "copy" in text or "cna" in text or "cnv" in text:
        return "CNA"
    return "CNA"


def is_cna_alteration(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.lower()
    patterns = ["cna", "cnv", "copy number", "amplification", "deep deletion", "copy_number"]
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask = mask | values.str.contains(pattern, regex=False)
    return mask


def read_csv_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def build_event_matrix(
    mutations: pd.DataFrame,
    clinical: pd.DataFrame | None = None,
    id_level: str = "patient",
    min_frequency: float = 0.03,
    top_k_events: int = 15,
    include_cna: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return event matrix, event frequency table, metadata, and QC metrics."""
    if mutations.empty:
        raise ValueError("Mutation table is empty.")

    id_candidates = ["patient_id"] if id_level == "patient" else ["sample_id"]
    fallback_ids = ["sample_id", "patient_id"] if id_level == "patient" else ["patient_id", "sample_id"]
    id_col = first_present(mutations.columns, id_candidates + fallback_ids)
    if id_col is None:
        raise ValueError("No patient_id or sample_id column found in mutation table.")

    gene_col = first_present(mutations.columns, ["gene", "event", "hugo_symbol", "gene_symbol"])
    if gene_col is None:
        raise ValueError("No gene/event column found in mutation table.")

    work = mutations.copy()
    work[id_col] = work[id_col].fillna("").astype(str).str.strip()
    work[gene_col] = work[gene_col].fillna("").astype(str).str.strip()
    work = work[(work[id_col] != "") & (work[gene_col] != "")]

    if "alteration_binary" in work.columns:
        work["_alteration_binary"] = normalize_bool_series(work["alteration_binary"])
        work = work[work["_alteration_binary"] == 1]

    alteration_col = first_present(work.columns, ["alteration_type", "variant_classification", "mutation_type"])
    if alteration_col and not include_cna:
        work = work[~is_cna_alteration(work[alteration_col])]

    if work.empty:
        raise ValueError("No alteration rows remain after filtering.")

    if include_cna and alteration_col:
        cna_mask = is_cna_alteration(work[alteration_col])
        work["_event"] = work[gene_col].where(
            ~cna_mask,
            work[gene_col].astype(str) + "__" + work[alteration_col].map(simplify_cna_type),
        )
    else:
        work["_event"] = work[gene_col]

    work["_event"] = work["_event"].astype(str).str.strip()
    work = work[(work["_event"] != "") & (work["_event"].str.lower() != "nan")]
    work = work[[id_col, "_event"]].drop_duplicates()
    work["_value"] = 1

    matrix = work.pivot_table(index=id_col, columns="_event", values="_value", aggfunc="max", fill_value=0)
    matrix = matrix.astype(int)

    if clinical is not None and not clinical.empty:
        clinical_id_col = id_col if id_col in clinical.columns else None
        if clinical_id_col is None:
            fallback = "sample_id" if id_col == "patient_id" else "patient_id"
            clinical_id_col = fallback if fallback in clinical.columns else None
        all_ids = (
            clinical[clinical_id_col].dropna().astype(str).str.strip()
            if clinical_id_col is not None
            else pd.Series(dtype=str)
        )
        all_ids = all_ids[all_ids != ""].drop_duplicates()
        if not all_ids.empty:
            matrix = matrix.reindex(all_ids, fill_value=0)

    sample_count = int(matrix.shape[0])
    if sample_count == 0:
        raise ValueError("Event matrix has zero samples.")

    frequencies = matrix.sum(axis=0).sort_values(ascending=False)
    freq_fraction = frequencies / sample_count
    if min_frequency >= 1:
        keep = frequencies[frequencies >= min_frequency]
    else:
        keep = freq_fraction[freq_fraction >= min_frequency]
    keep_events = list(keep.index[:top_k_events]) if top_k_events and top_k_events > 0 else list(keep.index)
    if not keep_events:
        keep_events = list(frequencies.index[:top_k_events]) if top_k_events and top_k_events > 0 else list(frequencies.index)

    matrix = matrix.loc[:, keep_events].astype(int)
    frequencies = matrix.sum(axis=0).sort_values(ascending=False)
    event_frequency = pd.DataFrame(
        {
            "event": frequencies.index,
            "sample_count": frequencies.values.astype(int),
            "frequency": (frequencies.values / sample_count).round(6),
        }
    )

    matrix_out = matrix.reset_index().rename(columns={id_col: id_col})
    metadata = make_metadata(clinical, matrix.index, id_col)

    event_counts = matrix.sum(axis=1)
    qc = {
        "id_column": id_col,
        "sample_count": sample_count,
        "event_count": int(matrix.shape[1]),
        "zero_event_sample_fraction": float((event_counts == 0).mean()),
        "event_count_min": int(event_counts.min()) if not event_counts.empty else 0,
        "event_count_median": float(event_counts.median()) if not event_counts.empty else 0.0,
        "event_count_max": int(event_counts.max()) if not event_counts.empty else 0,
        "mhn_ready": bool(10 <= matrix.shape[1] <= 25 and sample_count > 300),
    }
    return matrix_out, event_frequency, metadata, qc


def make_metadata(clinical: pd.DataFrame | None, ids: Iterable[str], id_col: str) -> pd.DataFrame:
    ids_series = pd.Series(list(ids), name=id_col).astype(str)
    if clinical is None or clinical.empty:
        return pd.DataFrame({id_col: ids_series})

    metadata = clinical.copy()
    if id_col not in metadata.columns:
        fallback = "sample_id" if id_col == "patient_id" else "patient_id"
        if fallback in metadata.columns:
            metadata[id_col] = metadata[fallback]
        else:
            metadata[id_col] = ""

    keep_columns = [id_col]
    keep_columns.extend([c for c in STANDARD_META_COLUMNS if c in metadata.columns and c != id_col])
    metadata = metadata[keep_columns].drop_duplicates(subset=[id_col])
    metadata[id_col] = metadata[id_col].fillna("").astype(str).str.strip()
    metadata = pd.DataFrame({id_col: ids_series}).merge(metadata, on=id_col, how="left")
    return metadata


def write_qc_report(path: Path, qc: dict, event_frequency: pd.DataFrame) -> None:
    high = markdown_table(event_frequency.head(10))
    low = markdown_table(event_frequency.tail(10))
    lines = [
        "# Event Matrix QC",
        "",
        f"- ID column: `{qc['id_column']}`",
        f"- Samples: {qc['sample_count']}",
        f"- Events retained: {qc['event_count']}",
        f"- Zero-event sample fraction: {qc['zero_event_sample_fraction']:.3f}",
        f"- Per-sample event count: min={qc['event_count_min']}, median={qc['event_count_median']:.2f}, max={qc['event_count_max']}",
        f"- MHN training recommendation met: {qc['mhn_ready']} (target: 10-25 events and >300 samples)",
        "",
        "## High-frequency Events",
        "",
        high,
        "",
        "## Low-frequency Events",
        "",
        low,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a binary event matrix.")
    parser.add_argument("--input", required=True, help="Standardized mutation CSV.")
    parser.add_argument("--clinical", default=None, help="Optional standardized clinical CSV.")
    parser.add_argument("--id-level", choices=["patient", "sample"], default="patient")
    parser.add_argument("--min-frequency", type=float, default=0.03)
    parser.add_argument("--top-k-events", type=int, default=15)
    parser.add_argument("--include-cna", choices=["yes", "no"], default="no")
    parser.add_argument("--output-dir", default=".", help="Project root output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    processed_dir = output_dir / "processed"
    reports_dir = output_dir / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("Reading mutation table from %s", args.input)
    mutations = read_csv_table(Path(args.input))
    clinical = read_csv_table(Path(args.clinical)) if args.clinical else None

    matrix, event_frequency, metadata, qc = build_event_matrix(
        mutations=mutations,
        clinical=clinical,
        id_level=args.id_level,
        min_frequency=args.min_frequency,
        top_k_events=args.top_k_events,
        include_cna=args.include_cna == "yes",
    )

    matrix.to_csv(processed_dir / "event_matrix.csv", index=False)
    event_frequency.to_csv(processed_dir / "event_frequency.csv", index=False)
    metadata.to_csv(processed_dir / "sample_metadata.csv", index=False)
    write_qc_report(reports_dir / "event_matrix_qc.md", qc, event_frequency)
    logging.info("Event matrix complete: %s samples, %s events", qc["sample_count"], qc["event_count"])


if __name__ == "__main__":
    main()
