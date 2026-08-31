"""Build a Rel-ObsTQ state table from an event matrix and sample metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


UNKNOWN_VALUES = {"", "na", "nan", "none", "unknown", "not reported", "not collected", "not applicable"}


def setup_logging(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "build_state_table.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def infer_id_column(event_matrix: pd.DataFrame, preferred: str) -> str:
    if preferred == "patient" and "patient_id" in event_matrix.columns:
        return "patient_id"
    if preferred == "sample" and "sample_id" in event_matrix.columns:
        return "sample_id"
    for candidate in ["patient_id", "sample_id"]:
        if candidate in event_matrix.columns:
            return candidate
    return str(event_matrix.columns[0])


def normalize_stage_group(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[lower.isin({"primary", "primary tumour", "primary tumor"})] = "primary"
    out[lower.str.contains("metast", regex=False) | lower.str.contains("distant", regex=False) | lower.str.contains("m1", regex=False)] = "metastatic"
    out[lower.str.match(r"^(stage\s*)?i[a-b]?$", na=False) | lower.str.match(r"^i[a-b]?$", na=False)] = "early"
    out[
        lower.str.match(r"^(stage\s*)?(ii|iii)[a-c]?$", na=False)
        | lower.str.match(r"^(ii|iii)[a-c]?$", na=False)
        | lower.str.contains("local_advanced", regex=False)
    ] = "local_advanced"
    out[lower.str.match(r"^(stage\s*)?iv[a-c]?$", na=False) | lower.str.match(r"^iv[a-c]?$", na=False)] = "metastatic"
    out[lower.isin(UNKNOWN_VALUES)] = "unknown"
    return out


def build_state_table(
    event_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    id_level: str = "patient",
    min_state_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    id_col = infer_id_column(event_matrix, id_level)
    event_columns = [c for c in event_matrix.columns if c != id_col]
    if not event_columns:
        raise ValueError("Event matrix contains no event columns.")

    matrix = event_matrix.copy()
    matrix[id_col] = matrix[id_col].astype(str)
    for col in event_columns:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0).astype(int)

    if metadata is None or metadata.empty:
        meta = pd.DataFrame({id_col: matrix[id_col]})
    else:
        meta = metadata.copy()
        if id_col not in meta.columns:
            fallback = "sample_id" if id_col == "patient_id" else "patient_id"
            if fallback in meta.columns:
                meta[id_col] = meta[fallback]
            else:
                meta[id_col] = ""
        meta[id_col] = meta[id_col].astype(str)
        meta = meta.drop_duplicates(subset=[id_col])

    merged = matrix.merge(meta, on=id_col, how="left", suffixes=("", "_meta"))
    if "patient_id" not in merged.columns:
        merged["patient_id"] = merged[id_col] if id_col == "patient_id" else ""
    if "sample_id" not in merged.columns:
        merged["sample_id"] = merged[id_col] if id_col == "sample_id" else ""
    if "cohort" not in merged.columns:
        merged["cohort"] = ""

    if "stage_group" in merged.columns:
        merged["stage_group"] = normalize_stage_group(merged["stage_group"])
    elif "stage_raw" in merged.columns:
        merged["stage_group"] = normalize_stage_group(merged["stage_raw"])
    elif "metastasis_status" in merged.columns:
        merged["stage_group"] = normalize_stage_group(merged["metastasis_status"])
    else:
        merged["stage_group"] = "unknown"

    event_values = merged[event_columns].astype(int)
    genotype = []
    for _, row in event_values.iterrows():
        active = sorted([event_columns[i] for i, value in enumerate(row.values) if value == 1])
        genotype.append("+".join(active) if active else "WT")
    merged["genotype_signature"] = genotype
    merged["event_count"] = event_values.sum(axis=1).astype(int)
    merged["state_id"] = merged["stage_group"].astype(str) + "::" + merged["genotype_signature"].astype(str)

    state_counts = merged["state_id"].value_counts()
    merged["state_count"] = merged["state_id"].map(state_counts).astype(int)
    merged["state_count_flag"] = merged["state_count"].apply(lambda x: "rare_state" if x < min_state_count else "valid_state")
    merged["usable_for_mhn"] = True
    has_stage = ~merged["stage_group"].fillna("").astype(str).str.lower().isin(UNKNOWN_VALUES)
    merged["usable_for_relobstq"] = has_stage & (merged["state_count_flag"] == "valid_state")
    merged["usable_for_relo bstq"] = merged["usable_for_relobstq"]

    output_columns = [
        "patient_id",
        "sample_id",
        "cohort",
        "stage_group",
        "genotype_signature",
        "event_count",
        "state_id",
        "state_count",
        "state_count_flag",
        "usable_for_mhn",
        "usable_for_relobstq",
        "usable_for_relo bstq",
    ]
    for optional in ["cancer_type", "stage_raw", "metastasis_status", "survival_time", "survival_event", "age", "sex"]:
        if optional in merged.columns and optional not in output_columns:
            output_columns.append(optional)

    state_table = merged[output_columns].copy()
    occupancy = (
        state_table.groupby(["stage_group", "genotype_signature", "state_id", "state_count_flag"], dropna=False)
        .size()
        .reset_index(name="state_count")
    )
    occupancy["occupancy_fraction"] = occupancy["state_count"] / len(state_table)
    occupancy = occupancy.sort_values(["state_count", "state_id"], ascending=[False, True])

    qc = {
        "sample_count": int(len(state_table)),
        "state_count": int(occupancy.shape[0]),
        "valid_state_count": int((occupancy["state_count_flag"] == "valid_state").sum()),
        "rare_state_count": int((occupancy["state_count_flag"] == "rare_state").sum()),
        "unknown_stage_fraction": float((state_table["stage_group"] == "unknown").mean()),
        "relobstq_usable_sample_count": int(state_table["usable_for_relobstq"].sum()),
    }
    return state_table, occupancy, qc


def markdown_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    shown = df.head(max_rows).copy()
    if shown.empty:
        return "(none)"
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_qc_report(path: Path, qc: dict, occupancy: pd.DataFrame) -> None:
    lines = [
        "# State Table QC",
        "",
        f"- Samples: {qc['sample_count']}",
        f"- States: {qc['state_count']}",
        f"- Valid states: {qc['valid_state_count']}",
        f"- Rare states: {qc['rare_state_count']}",
        f"- Unknown stage fraction: {qc['unknown_stage_fraction']:.3f}",
        f"- Rel-ObsTQ usable samples: {qc['relobstq_usable_sample_count']}",
        "",
        "## Top Occupied States",
        "",
        markdown_table(occupancy.head(20)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Rel-ObsTQ state table.")
    parser.add_argument("--event-matrix", default="processed/event_matrix.csv")
    parser.add_argument("--metadata", default="processed/sample_metadata.csv")
    parser.add_argument("--id-level", choices=["patient", "sample"], default="patient")
    parser.add_argument("--min-state-count", type=int, default=5)
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

    event_matrix = pd.read_csv(args.event_matrix, dtype=str)
    metadata_path = Path(args.metadata)
    metadata = pd.read_csv(metadata_path, dtype=str) if metadata_path.exists() else None
    state_table, occupancy, qc = build_state_table(
        event_matrix=event_matrix,
        metadata=metadata,
        id_level=args.id_level,
        min_state_count=args.min_state_count,
    )
    state_table.to_csv(processed_dir / "state_table.csv", index=False)
    occupancy.to_csv(processed_dir / "state_occupancy.csv", index=False)
    write_qc_report(reports_dir / "state_table_qc.md", qc, occupancy)
    logging.info("State table complete: %s samples, %s states", qc["sample_count"], qc["state_count"])


if __name__ == "__main__":
    main()
