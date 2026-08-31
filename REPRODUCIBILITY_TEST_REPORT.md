# Reproducibility Test Report

Audit date: 2026-08-31

## Repository and environment

- Frozen audit commit: `e4215608dd394581da19e1ae0d8a0206c4d33798`.
- Audit OS: Windows 11 x86-64 (`10.0.26200`).
- Python: 3.11.16 and 3.12.13, both 64-bit.
- Package: `relobstq-mhn==0.2.0` installed editable.
- NumPy 1.26.4; pandas 3.0.5; SciPy 1.14.1; statsmodels 0.15.0.
- Official backend: `mhn==1.2.3`.

The first `mhn` build attempt in each environment selected an incomplete MSVC 14.33 x86 toolchain and failed because `MSVCRT.lib` was unavailable. Re-running after `vcvars64.bat -vcvars_ver=14.44` built and installed official CPython 3.11 and 3.12 x64 wheels successfully. This was an environment/toolchain failure, not a method-code failure.

## Tests

| condition | command | result | runtime | warnings |
|---|---|---:|---:|---|
| Restored preprocessing package, before MHN install | `python -m pytest -q` | 9 passed | 9.26 s | none material |
| Official `mhn==1.2.3` installed | `python -m pytest -q` | 9 passed | 6.92 s | one small-sample Wilcoxon warning |
| Provenance and fixed-panel preparation added | `python -m pytest -q` | 10 passed | 2.39 s | same small-sample warning |
| Final Phase-0 code before reports | `python -m pytest -q` | 10 passed | 2.32 s | same small-sample warning |
| Patient-cluster bootstrap API and direct test | `python -m pytest -q` | 11 passed | 2.14 s | same small-sample warning |
| Python 3.11.16 compatibility | `python -m pytest -q` | 11 passed | 9.04 s | same small-sample warning |
| Final publication tree, Python 3.12.13 | `python -m pytest -q` | 11 passed | 2.31 s | same small-sample warning |
| Final publication tree, Python 3.11.16 | `python -m pytest -q` | 11 passed | 2.27 s | same small-sample warning |
| Selected E17 restoration and frozen-result contract | `python -m pytest -q` | 14 passed | 1.99 s | same small-sample warning |
| Selected E17 restoration, Python 3.11.16 | `python -m pytest -q` | 14 passed | 3.52 s | same small-sample warning |

The restored E17 command also passed byte-code compilation and
`python experiments/run_longitudinal.py --help` after installing the declared
`seaborn>=0.13` figure dependency. Regression tests freeze the selected cohort
counts, AUCs and fallback-backend metadata.

Imports resolved to the audited source tree under `src/relobstq_mhn` in both environments. Official-backend smoke fits on the same 120-sample, 3-event binary matrix returned finite `3 x 3` theta matrices and selected lambda `0.025`; three-fold CV completed in 0.64 s on Python 3.12 and 1.716 s on Python 3.11. The optimizer warned in both runs that the selected lambda was at the search-grid boundary.

## Frozen package omission

The frozen Git tree omitted `src/relobstq_mhn/data/__init__.py` and `processing.py`, although both were declared by README/package imports and `RELEASE_MANIFEST.tsv`. Exact copies were recovered from the historical workspace. Their sizes and SHA-256 values match the frozen manifest exactly:

| file | bytes | SHA-256 |
|---|---:|---|
| `src/relobstq_mhn/data/__init__.py` | 496 | `cbb6f26bf523d7e05bfe317564d3c7b85f8927f7caf8905d1f962e84f7a5787b` |
| `src/relobstq_mhn/data/processing.py` | 9347 | `ae4b9a4781f8dfe944d6611b7f999854484e7db823b9a5518d03285035792a96` |

**Is the original frozen commit self-contained? NO.** The repaired successor is self-contained for package installation and unit/workflow tests; restricted data remain external by design.

A clean no-dependency wheel build succeeded. Inspection of the wheel archive confirmed that it contains `relobstq_mhn/data/__init__.py`, `relobstq_mhn/data/processing.py`, and `relobstq_mhn/workflows/preparation.py`; the repaired modules are therefore present in the distributable package, not only in the working tree.

## Status

`RESOLVED_WITH_CAVEAT`: code and official-backend tests pass, but a compiler toolchain may be required on Windows when no prebuilt `mhn` wheel is available.
