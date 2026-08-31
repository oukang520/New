from __future__ import annotations

import hashlib
import shutil
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "public_release" / "RelObsTQ_MHN_reproducible_code"


ROOT_TEXT_FILES = [
    "requirements.txt",
    "README_RUN.md",
    "METHOD_CODE_MAPPING.md",
    "CLAIM_EVIDENCE_MATRIX.md",
    "RESULTS_MASTER_TABLE.md",
    "FIGURE_PLAN.md",
]

SRC_PY_FILES = [
    "audit_aacr_by_cancer.py",
    "audit_datasets.py",
    "audit_experiment_figures.py",
    "audit_publication_single_figures.py",
    "build_event_matrix.py",
    "build_experiment_ready_datasets.py",
    "build_manuscript_master_plan.py",
    "build_panel_reassembly_index.py",
    "build_public_release_package.py",
    "build_state_table.py",
    "extract_aacr_oncotree_datasets.py",
    "figure_style.py",
    "render_palette_options.py",
    "render_publication_single_figures.py",
    "run_experiments_01_02.py",
    "run_experiment_03.py",
    "run_experiment_04.py",
    "run_experiment_05.py",
    "run_experiment_06.py",
    "run_experiment_06_dwell_gradient.py",
    "run_experiment_06_enhanced.py",
    "run_experiment_07.py",
    "run_experiment_08.py",
    "run_experiment_09.py",
    "run_experiment_10.py",
    "run_experiment_11.py",
    "run_experiment_12.py",
    "run_experiment_13.py",
    "run_experiment_14.py",
    "run_experiment_15.py",
    "run_experiment_16.py",
    "run_experiment_17_longitudinal_extension.py",
    "run_experiment_17_longitudinal_public.py",
    "run_experiment_17_new_four_cohorts.py",
    "validate_experiments_01_02.py",
    "validate_experiment_03.py",
    "validate_experiment_04.py",
    "validate_experiment_05.py",
    "validate_experiment_06.py",
    "validate_experiment_06_enhanced.py",
    "validate_experiment_07.py",
    "validate_experiment_09.py",
    "validate_experiment_10.py",
    "validate_experiment_11.py",
    "validate_experiment_12.py",
    "validate_experiment_13.py",
    "validate_experiment_14.py",
    "validate_experiment_15.py",
    "validate_experiment_16.py",
    "validate_experiment_ready.py",
]


def ensure_clean_target() -> None:
    target = TARGET.resolve()
    allowed_root = (ROOT / "public_release").resolve()
    if allowed_root not in target.parents:
        raise RuntimeError(f"Refusing to remove unexpected target: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def copy_file(src_rel: str, dst_rel: str | None = None) -> None:
    src = ROOT / src_rel
    if not src.exists():
        return
    dst = TARGET / (dst_rel or src_rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src_rel: str, dst_rel: str | None = None) -> None:
    src = ROOT / src_rel
    if not src.exists():
        return
    dst = TARGET / (dst_rel or src_rel)
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "*.docx",
        "*.pdf",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tif",
        "*.tiff",
        "*.zip",
    )
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def write_text(path: str, text: str) -> None:
    p = TARGET / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> None:
    rows = ["path\tsize_bytes\tsha256"]
    for path in sorted(TARGET.rglob("*")):
        if path.is_file():
            rows.append(f"{path.relative_to(TARGET).as_posix()}\t{path.stat().st_size}\t{sha256(path)}")
    write_text("RELEASE_MANIFEST.tsv", "\n".join(rows))


def build_readme() -> str:
    return f"""
# Rel-ObsTQ-MHN Reproducible Code

This repository contains the reproducible method and experiment code for the
Rel-ObsTQ-MHN project. The method estimates a state-level relative dwell/stasis
index from cross-sectional genomic cohorts by contrasting observed state
occupancy with progression-expected inflow derived from a Mutual Hazard Network
(MHN) transition backbone.

Prepared for public scientific-code release on {date.today().isoformat()}.

## Core Method

For a stage-specific genotype state `v`, the primary score is:

```text
L_v      = N_v / N
F_hat_v  = sum_u L_u * P(u -> v | theta)
R_raw_v  = L_v / (F_hat_v + epsilon)
R*_v     = R_raw_v / median(R_raw among eligible states)
```

`R* > 1` indicates a state that is more frequently observed than expected from
its MHN-derived progression inflow, and is interpreted as a relative dwell/stasis
proxy. It is not an absolute calendar-time estimate.

## Repository Layout

```text
src/relobstq_mhn/     Reusable method package: states, transitions, scoring,
                      bootstrap, topology, data processing and simulation.
src/run_experiment_*  Experiment runners E1-E17.
src/validate_*        Validation and QC scripts for experiment outputs.
configs/              Experiment configuration files.
tests/                Smoke and core unit tests.
docs/                 Method API, figure-style notes and manuscript planning
                      support documents.
external/mhn/         Source note for the official MHN dependency.
```

Raw patient-level data, processed datasets, generated figures and large result
directories are intentionally not included in this public code package.

## Environment

Use Python 3.11 or 3.12 for the full MHN-backed workflow. The official `mhn`
package is pinned as:

```text
mhn==1.2.3; python_version < "3.13"
```

Quick setup:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda-style setup:

```powershell
conda env create -f environment.yml
conda activate relobstq-mhn
```

## Data Availability

The main cross-sectional experiments use AACR Project GENIE v18.0-public
subcohorts (LUAD, COAD and IDC) as configured in
`configs/selected_experiment_datasets.yaml`. Public longitudinal validation uses
the cohorts documented in the E17 configuration and scripts. Users must download
public datasets from their original providers and place them under the expected
local paths before running the full pipeline.

No raw AACR/GENIE, ICGC, cBioPortal or other patient-level tables are redistributed
here.

## Minimal Verification

The two test commands should be run separately because the root tests and package
tests intentionally contain similarly named test modules:

```powershell
python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q
python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q
```

## Full Reproduction Order

The scripts are designed to be run from the repository root after data have been
placed under the configured paths.

```powershell
python src/build_experiment_ready_datasets.py
python src/validate_experiment_ready.py

python src/run_experiments_01_02.py
python src/validate_experiments_01_02.py

python src/run_experiment_03.py
python src/run_experiment_04.py
python src/run_experiment_05.py
python src/run_experiment_06_enhanced.py
python src/run_experiment_06_dwell_gradient.py
python src/run_experiment_07.py
python src/run_experiment_08.py
python src/run_experiment_09.py
python src/run_experiment_10.py
python src/run_experiment_11.py
python src/run_experiment_12.py
python src/run_experiment_13.py
python src/run_experiment_14.py
python src/run_experiment_15.py
python src/run_experiment_16.py
python src/run_experiment_17_longitudinal_public.py
python src/run_experiment_17_longitudinal_extension.py
```

See `docs/CODE_AVAILABILITY.md`, `docs/REPRODUCIBILITY_CHECKLIST.md` and
`docs/METHOD_CODE_MAPPING.md` for more detail.
"""


def main() -> None:
    ensure_clean_target()

    for file in ROOT_TEXT_FILES:
        copy_file(file, f"docs/audit/{Path(file).name}" if file.endswith(".md") and file != "README_RUN.md" else file)

    copy_tree("src/relobstq_mhn")
    for file in SRC_PY_FILES:
        copy_file(f"src/{file}")
    copy_tree("configs")
    copy_tree("tests")
    copy_tree("docs")
    copy_file("external/mhn/MHN_SOURCE.md")

    write_text("README.md", build_readme())
    write_text(
        ".gitignore",
        """
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
Data/
data/
processed/
results/
logs/
tmp/
*.png
*.pdf
*.docx
*.zip
""",
    )
    write_text(
        "environment.yml",
        """
name: relobstq-mhn
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - pandas>=2.0
      - numpy>=1.24
      - pyyaml>=6.0
      - pytest>=7.0
      - matplotlib>=3.8
      - seaborn>=0.13
      - scipy>=1.11
      - statsmodels>=0.14
      - pillow>=10.0
      - mhn==1.2.3
""",
    )
    write_text(
        "pyproject.toml",
        """
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "relobstq-mhn"
version = "0.1.0"
description = "Relative observed-state topology quantification with MHN-derived inflow"
requires-python = ">=3.11,<3.13"
readme = "README.md"

[tool.setuptools.packages.find]
where = ["src"]
""",
    )
    write_text(
        "CITATION.cff",
        """
cff-version: 1.2.0
message: "Please cite the associated Rel-ObsTQ-MHN manuscript when available."
title: "Rel-ObsTQ-MHN reproducible code"
version: "0.1.0"
date-released: 2026-08-31
authors:
  - name: "Rel-ObsTQ-MHN authors"
""",
    )
    write_text(
        "LICENSE_NOTICE.md",
        """
# License Notice

This public code package is prepared for scientific review and reproducibility.
The final open-source license should be selected by the authors before formal
publication. Until then, reuse should cite the associated manuscript/code
repository and follow the data-provider licenses for all external datasets.
""",
    )
    write_text(
        "docs/CODE_AVAILABILITY.md",
        """
# Code Availability

This repository contains reusable method code and experiment runners for all
major Rel-ObsTQ-MHN analyses. Raw data and generated result folders are not
included because they may be large, provider-controlled, or patient-level.

Core reusable code is under `src/relobstq_mhn/`. Experiment code is under `src/`
as `run_experiment_*.py` and `validate_experiment_*.py`.
""",
    )
    write_text(
        "docs/DATA_AVAILABILITY.md",
        """
# Data Availability

The main cross-sectional analyses use AACR Project GENIE v18.0-public-derived
LUAD, COAD and IDC subcohorts. PACA-CA and other initially screened datasets are
not part of the current main analysis chain.

The public longitudinal validation is implemented in the E17 scripts and
configuration. Download all external datasets from their official public sources
and place them under the paths expected by the corresponding configuration files.

No raw patient-level data are redistributed in this code repository.
""",
    )
    write_text(
        "docs/REPRODUCIBILITY_CHECKLIST.md",
        """
# Reproducibility Checklist

- Python: use 3.11 or 3.12 for the full MHN dependency.
- Main dependency: `mhn==1.2.3`.
- Random seeds: defined in experiment configuration files where applicable.
- Data: not bundled; download from original public providers.
- Minimal tests:
  - `python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q`
  - `python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q`
- Main cross-sectional cohorts: AACR_LUAD, AACR_COAD, AACR_IDC.
- Excluded from current main chain: PACA-CA and other low-feasibility datasets
  listed in `configs/selected_experiment_datasets.yaml`.
""",
    )
    write_text(
        "scripts/reproduce_minimal.ps1",
        """
$ErrorActionPreference = "Stop"
python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q
python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q
""",
    )
    build_manifest()
    print(f"Built public release package: {TARGET}")


if __name__ == "__main__":
    main()
