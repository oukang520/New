# Reproducibility Test Report v3

Audit date: 2026-09-02

## Environment

- Windows 11 x86-64
- Python 3.12.13 (final evidence runs)
- Python 3.11.16 (compatibility audit)
- `relobstq-mhn==0.2.0`
- `mhn==1.2.3`
- NumPy 1.26.4, pandas 3.0.5, SciPy 1.14.1

## Final verification

| check | result |
|---|---|
| unit and workflow tests | 17 passed |
| expected warnings | one tiny-sample Wilcoxon smoke-test warning |
| Python 3.11/3.12 package compatibility | passed |
| official-cMHN smoke fit | finite theta, passed |
| AACR_LUAD full current fit | passed |
| AACR_COAD full current fit | passed |
| AACR_IDC full current fit | passed |
| p15 genotype alignment mismatches | 0 / 0 / 0 |
| source-run Git worktree state | clean for every frozen unit |
| source result manifests | verified before freeze |
| final evidence index | 99 file hashes |

Run the test suite with:

```powershell
python -m pytest -q
```

Verify the frozen evidence index by recomputing SHA-256 for every path listed
in `reference_results/final_manuscript_evidence/FINAL_EVIDENCE_INDEX.tsv`.

## Packaging note

The original audit commit `e4215608...` omitted the declared data-processing
package. Exact manifest-matching copies were restored in the repaired
successor and are included in wheel builds. Restricted patient-level datasets
remain external by design; provider-authorized inputs must be placed according
to `docs/DATA_AVAILABILITY.md`.

On Windows, building `mhn` from source may require a correctly configured x64
MSVC toolchain when no compatible wheel is available. This is an environment
requirement, not a method-code failure.

Status: **PASS**.
