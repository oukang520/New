# Invariant-EvAM

This repository packages the current G1–G5 scientific release and the core code that produced its retained results. The scientific question is which cancer-evolution interaction targets can be answered from heterogeneous biased observations, how to give reliable answers, and which data to collect when information is missing.

The release contains **59 standalone panels / 118 PDF+SVG files**. It supports a conditional methodological information–inference–acquisition story with limited biological information bridges. It does not establish universal identification, independent diagnostic superiority in real TRACERx data, a new causal cancer mechanism, or clinical benefit.

Start with the [complete Chinese research narrative](docs/RESEARCH_STORY_CN.md) ([editable Word version](docs/RESEARCH_STORY_CN.docx)), [中文说明](README_CN.md), [current five-group integration report](paper_experiments_v1/FIVE_GROUP_INTEGRATION_REPORT_CN.md), and [reproduction guide](docs/REPRODUCE_CN.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -B scripts/verify_release.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Large downloaded data, local environments, upstream vendor packages and reference screenshots are external inputs, documented in [data dependencies](docs/DATA_DEPENDENCIES_CN.md). Frozen study code and data hashes remain unchanged; current figures are preserved rather than regenerated during packaging.
