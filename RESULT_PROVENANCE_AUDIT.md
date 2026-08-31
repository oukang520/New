# Result Provenance Audit

Audit started: 2026-08-31

## Repository Boundary

- Frozen Git repository: `public_release/RelObsTQ_MHN_reproducible_code`.
- Frozen commit and initial HEAD: `e4215608dd394581da19e1ae0d8a0206c4d33798`.
- Selected original E17 restoration commit: `1a03c8183719286d5f822963d8653cd71e230814`.
- Initial Git state: clean; `main` synchronized with `origin/main`.
- Historical development workspace: `D:/project/Rel_ObsHN`; this directory is not a Git repository.
- Historical `results/`, `processed/`, `configs/`, and plotting-heavy scripts therefore have no locally verifiable commit identity. They are treated as legacy evidence until another immutable source is found.

## Initial Classification

The initial matrix intentionally does not treat any legacy numerical result as reproduced by v0.2.0. `reproduced=NO` means that no isolated Phase-0 rerun had been completed when the matrix was created; it does not mean that the reported value is false.

Status vocabulary is restricted to `RESOLVED`, `RESOLVED_WITH_CAVEAT`, `UNRESOLVED`, `NOT_REPRODUCIBLE`, `LEGACY_ONLY`, `REQUIRES_DATA`, and `REQUIRES_MANUAL_DECISION`.

## Frozen-Commit Self-Containment Finding

**Is frozen commit self-contained? NO.**

Evidence:

1. `README.md` declares a `src/relobstq_mhn/data/` package.
2. `RELEASE_MANIFEST.tsv` records `src/relobstq_mhn/data/__init__.py` and `src/relobstq_mhn/data/processing.py` with SHA-256 hashes.
3. `git ls-tree -r e4215608...` contains neither file.
4. Neither file exists in any of the three commits in the public repository history.
5. Source copies with the exact manifest sizes exist in the non-Git historical workspace under `D:/project/Rel_ObsHN/src/relobstq_mhn/data/`.

This was confirmed as a release-packaging omission. Exact historical workspace
copies matched both manifest sizes and SHA-256 hashes and were restored without
algorithmic rewriting. Tests then passed with and without the optional MHN
backend.

## Initial High-Risk Gaps

| issue | initial status | evidence |
|---|---|---|
| Frozen package omits declared preprocessing module | UNRESOLVED | Git tree versus `RELEASE_MANIFEST.tsv` |
| No canonical raw GENIE to experiment-ready command | UNRESOLVED | `experiments/` inventory |
| E5 legacy uses 200 bootstrap replicates while frozen config requests 500 | UNRESOLVED | `RESULTS_MASTER_TABLE.md`; `configs/cross_sectional.yaml` |
| E17 original versus strict patient-grouped cross-fit provenance | RESOLVED_WITH_CAVEAT | selected implementation, frozen references and comparison record |
| `run_all.py` does not generate every E3-E17 manuscript evidence contract | UNRESOLVED | runner inventory and `docs/EXPERIMENT_MAPPING.md` |
| Frozen code has not yet been tested in Python 3.11/3.12 with official `mhn==1.2.3` | UNRESOLVED | Phase-0 environment pending |

## Completed high-priority resolutions

- The new canonical p15 preparation entry reproduced all three historical E1
  cMHN matrices byte-for-byte by SHA-256.
- Python 3.11.16 and 3.12.13 both passed the full test suite and an official
  `mhn==1.2.3` cMHN smoke fit after selecting the MSVC 14.44 x64 toolchain.
- Historical E5 200-replicate stability metrics reproduced to floating-point
  precision. The 500-replicate refinement was run separately.
- The exact original E17 runner, configuration, aggregate result tables and fit
  metadata are restored as the selected primary contract.
- The strict patient-grouped official-cMHN audit is explicitly superseded rather
  than relabeled as equivalent evidence; its values remain in Git history.
- E17 is interpreted as supportive external longitudinal consistency with a
  disclosed full-cohort fallback backbone and limited changed classes.
- Clinical subgroup directions and all E16 route edges are retained in dedicated
  TSV audits.

Canonical-runner gaps for several legacy experiments remain explicit in the
matrix. A callable function is not counted as reproduced experiment evidence.

## Non-Overwrite Rule

Historical `results/experiment_*` directories remain unchanged. The selected
E17 aggregates are frozen under `reference_results/experiment_17_legacy/`.
Superseded strict-audit artifacts are removed from the current release tree but
remain recoverable from Git history.
