# MHN Official Component Snapshot

Downloaded for use as the transition-backbone foundation of the Rel-ObsTQ-MHN project.
No MHN training or Rel-ObsTQ scoring was run during this step.

## Official Sources

- Package name: `mhn`
- Version: `1.2.3`
- PyPI project: https://pypi.org/project/mhn/
- Official GitHub source: https://github.com/spang-lab/LearnMHN
- Documentation: https://learnmhn.readthedocs.io/en/latest/index.html
- License: MIT License

## Local Files

- PyPI source distribution: `external/mhn/pypi/mhn-1.2.3.tar.gz`
- GitHub release zip: `external/mhn/github/LearnMHN-v1.2.3.zip`
- Extracted GitHub source: `external/mhn/github/LearnMHN-v1.2.3/LearnMHN-1.2.3/`

## SHA256

- `mhn-1.2.3.tar.gz`: `3EA33F131BAEC3B209424C81B7E95C511F791A92E5332742B78B17323D1C4166`
- `LearnMHN-v1.2.3.zip`: `B0407C6930EE3F3F4C5DE7D9CE6047DDE892A909C37AB102E10BFC9E1612EC40`

## Compatibility Notes

PyPI metadata for `mhn==1.2.3` declares:

```text
Requires-Python: >=3.8,<3.13
```

The current base conda environment used for data auditing is Python 3.13.5, so the package
was downloaded but not installed into that environment. For the next MHN step, create a
Python 3.12 or 3.11 environment with a C compiler available.

Suggested environment:

```powershell
conda create -n relobs-mhn python=3.12 pip
conda activate relobs-mhn
pip install mhn==1.2.3
```

Because `mhn` uses Cython extensions, installation may require MSVC on Windows or GCC/Clang
on Linux/macOS.

## Core Modules For This Project

The core code lives under:

- `mhn/model.py`
- `mhn/optimizers.py`
- `mhn/training/`
- `mhn/full_state_space/`
- `mhn/utilities.pyx`

For Rel-ObsTQ-MHN, we will use MHN only as the transition backbone. The downstream
Rel-ObsTQ layer should consume learned MHN state-dependent event hazards or next-event
probabilities and then compute `F_hat` and `R*` in our own project code.
