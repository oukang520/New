
$ErrorActionPreference = "Stop"
python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q
python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q
