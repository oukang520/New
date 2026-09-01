"""Verify and freeze final manuscript evidence under tracked reference results."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from relobstq_mhn.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "final_manuscript_evidence"
DESTINATION_ROOT = ROOT / "reference_results" / "final_manuscript_evidence"
COHORTS = ("AACR_LUAD", "AACR_COAD", "AACR_IDC")


def _units() -> list[tuple[str, Path, Path]]:
    units = []
    for cohort in COHORTS:
        units.append(
            (
                f"cross_sectional_{cohort}",
                SOURCE_ROOT / "cross_sectional" / cohort,
                DESTINATION_ROOT / "cross_sectional" / cohort,
            )
        )
        units.append(
            (
                f"core_evidence_{cohort}",
                SOURCE_ROOT / "core_evidence" / cohort,
                DESTINATION_ROOT / "core_evidence" / cohort,
            )
        )
    units.extend(
        [
            (
                "simulation_dwell_gradient",
                SOURCE_ROOT / "simulation_dwell_gradient",
                DESTINATION_ROOT / "simulation_dwell_gradient",
            ),
            (
                "simulation_topology_robustness",
                SOURCE_ROOT / "simulation_topology_robustness",
                DESTINATION_ROOT / "simulation_topology_robustness",
            ),
        ]
    )
    return units


def _verify_unit(unit: str, source: Path) -> tuple[dict, int]:
    manifest_path = source / "result_manifest.tsv"
    metadata_path = source / "run_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{unit}: missing result manifest or run metadata")
    manifest = pd.read_csv(manifest_path, sep="\t")
    for row in manifest.itertuples(index=False):
        path = source / str(row.path)
        if not path.is_file():
            raise FileNotFoundError(f"{unit}: manifest file is absent: {row.path}")
        if path.stat().st_size != int(row.size_bytes) or sha256_file(path) != str(row.sha256):
            raise ValueError(f"{unit}: manifest mismatch: {row.path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("git", {}).get("dirty") is not False:
        raise ValueError(f"{unit}: final evidence must originate from a clean Git worktree")
    return metadata, len(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION_ROOT)
    args = parser.parse_args()
    destination_root = args.destination.resolve()
    expected_parent = (ROOT / "reference_results").resolve()
    if destination_root.parent != expected_parent:
        raise ValueError("Destination must be a direct child of reference_results")
    if destination_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze: {destination_root}")

    run_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    verified: list[tuple[str, Path, Path, dict]] = []
    for unit, source, default_destination in _units():
        metadata, manifest_entries = _verify_unit(unit, source)
        relative_destination = default_destination.relative_to(DESTINATION_ROOT)
        destination = destination_root / relative_destination
        verified.append((unit, source, destination, metadata))
        run_rows.append(
            {
                "unit": unit,
                "git_commit": metadata["git"]["commit"],
                "git_dirty": metadata["git"]["dirty"],
                "python": metadata["python"]["version"],
                "mhn": metadata["packages"].get("mhn"),
                "numpy": metadata["packages"].get("numpy"),
                "pandas": metadata["packages"].get("pandas"),
                "manifest_entries_verified": manifest_entries,
            }
        )
        if unit.startswith("cross_sectional_"):
            cohort = unit.removeprefix("cross_sectional_")
            fit = json.loads((source / "fit_metadata.json").read_text(encoding="utf-8"))
            cv = pd.read_csv(source / "tables" / "cv_scores.tsv", sep="\t")
            selected = cv[cv["selected"].astype(bool)].iloc[0]
            best = cv.loc[cv["mean_test_log_likelihood"].idxmax()]
            model_rows.append(
                {
                    "cohort": cohort,
                    "backend": fit["backend"],
                    "samples": fit["sample_count"],
                    "events": fit["event_count"],
                    "selected_lambda": fit["selected_lambda"],
                    "selected_lambda_multiplier": selected["lambda_multiplier"],
                    "best_mean_cv_multiplier": best["lambda_multiplier"],
                    "selected_at_search_boundary": bool(
                        selected.name in {cv.index.min(), cv.index.max()}
                    ),
                    "selection_rule": "five-fold CV, one-standard-error rule",
                    "theta_finite": fit["finite_theta"],
                }
            )
            for item in metadata["inputs"]:
                input_rows.append(
                    {
                        "cohort": cohort,
                        "file": Path(item["path"]).name,
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                )

    for _, source, destination, _ in verified:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)

    pd.DataFrame(run_rows).to_csv(destination_root / "RUN_ENVIRONMENT_AUDIT.tsv", sep="\t", index=False)
    pd.DataFrame(input_rows).to_csv(destination_root / "P15_INPUT_AUDIT.tsv", sep="\t", index=False)
    pd.DataFrame(model_rows).to_csv(destination_root / "MHN_MODEL_SELECTION_AUDIT.tsv", sep="\t", index=False)
    freeze_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT.resolve()),
        "destination_root": str(destination_root),
        "units": [unit for unit, *_ in verified],
        "note": "All source manifests were verified before copying; source runs had git dirty=false.",
    }
    (destination_root / "FREEZE_METADATA.json").write_text(
        json.dumps(freeze_metadata, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for path in sorted(destination_root.rglob("*")):
        if path.is_file() and path.name != "FINAL_EVIDENCE_INDEX.tsv":
            rows.append(
                {
                    "path": path.relative_to(destination_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(destination_root / "FINAL_EVIDENCE_INDEX.tsv", sep="\t", index=False)
    print(f"frozen {len(rows)} evidence files under {destination_root}")


if __name__ == "__main__":
    main()
