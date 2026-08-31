"""Validate experiment-ready Rel-ObsTQ-MHN dataset packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "analysis_metadata.csv",
    "mutations_long.csv",
    "event_matrix.csv",
    "mhn_training_matrix.csv",
    "mhn_row_index_map.csv",
    "event_frequency.csv",
    "state_table.csv",
    "state_occupancy.csv",
    "dataset_manifest.json",
    "qc_report.md",
]


def validate_one(dataset_dir: Path) -> dict:
    problems = []
    missing = [name for name in REQUIRED_FILES if not (dataset_dir / name).exists()]
    if missing:
        problems.append("missing:" + ",".join(missing))
        return {"dataset": dataset_dir.name, "problems": ";".join(problems)}

    metadata = pd.read_csv(dataset_dir / "analysis_metadata.csv", dtype=str)
    event_matrix = pd.read_csv(dataset_dir / "event_matrix.csv", dtype=str)
    mhn_matrix = pd.read_csv(dataset_dir / "mhn_training_matrix.csv")
    row_map = pd.read_csv(dataset_dir / "mhn_row_index_map.csv", dtype=str)
    state_table = pd.read_csv(dataset_dir / "state_table.csv", dtype=str)
    occupancy = pd.read_csv(dataset_dir / "state_occupancy.csv", dtype=str)
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    events = manifest.get("events", [])

    if len(metadata) != len(event_matrix):
        problems.append("metadata_event_rows")
    if len(metadata) != len(mhn_matrix):
        problems.append("metadata_mhn_rows")
    if len(metadata) != len(row_map):
        problems.append("metadata_rowmap_rows")
    if len(metadata) != len(state_table):
        problems.append("metadata_state_rows")
    if list(mhn_matrix.columns) != events:
        problems.append("mhn_event_columns")
    if any(col.lower().endswith("_id") or col == "analysis_id" for col in mhn_matrix.columns):
        problems.append("id_in_mhn_matrix")
    if not mhn_matrix.empty:
        values = set(pd.unique(mhn_matrix.fillna(0).astype(int).values.ravel()))
        if not values.issubset({0, 1}):
            problems.append("nonbinary_mhn_matrix")
    if set(event_matrix["analysis_id"]) != set(metadata["analysis_id"]):
        problems.append("event_metadata_id_mismatch")
    if set(row_map["analysis_id"]) != set(metadata["analysis_id"]):
        problems.append("rowmap_metadata_id_mismatch")
    if state_table["state_id"].isna().any():
        problems.append("missing_state_id")
    if metadata["analysis_id"].duplicated().any():
        problems.append("duplicate_analysis_id")
    if len(events) != len(set(events)):
        problems.append("duplicate_event_names")

    return {
        "dataset": dataset_dir.name,
        "rows": len(metadata),
        "events": len(events),
        "states": state_table["state_id"].nunique(),
        "valid_states": int((occupancy["state_count_flag"] == "valid_state").sum()) if "state_count_flag" in occupancy.columns else "",
        "mhn_has_only_binary_events": "nonbinary_mhn_matrix" not in problems,
        "row_alignment_ok": all(
            p not in problems
            for p in ["metadata_event_rows", "metadata_mhn_rows", "metadata_rowmap_rows", "metadata_state_rows"]
        ),
        "id_linkage_ok": all(
            p not in problems
            for p in ["event_metadata_id_mismatch", "rowmap_metadata_id_mismatch", "duplicate_analysis_id"]
        ),
        "problems": ";".join(problems) if problems else "OK",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate experiment-ready dataset packages.")
    parser.add_argument("--input-dir", default="processed/experiment_ready")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    rows = [validate_one(path) for path in sorted(root.iterdir()) if path.is_dir()]
    result = pd.DataFrame(rows)
    result.to_csv(root / "experiment_ready_validation.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
