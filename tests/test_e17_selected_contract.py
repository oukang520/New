from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results" / "experiment_17_legacy"


def test_selected_e17_metrics_are_frozen() -> None:
    expected = {
        "GLASS": ("72 (51/21)", "0.67 [0.53, 0.79]", "1.19"),
        "CRC-triplets": ("23 (20/3)", "0.65 [0.35, 0.91]", "1.08"),
        "MNM-WashU": ("10 (9/1)", "0.89 [0.75, 1.00]", "1.1"),
    }
    with (REFERENCE / "integrated_longitudinal_metrics_table.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["cohort"]: row for row in csv.DictReader(handle, delimiter="\t")}

    assert set(rows) == set(expected)
    for cohort, values in expected.items():
        assert (rows[cohort]["n_P_C"], rows[cohort]["auc_95ci"], rows[cohort]["ap_lift"]) == values


def test_selected_e17_backend_is_disclosed_as_fallback() -> None:
    metadata_files = sorted(REFERENCE.glob("fit_metadata_*.json"))
    assert len(metadata_files) == 3
    for path in metadata_files:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["backend"] == "frequency_cooccurrence_backbone"


def test_default_e17_runner_uses_selected_config() -> None:
    from sirdwell_experiments.e17 import CONFIG_PATH

    run_all = (ROOT / "experiments" / "run_all.py").read_text(encoding="utf-8")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    external = yaml.safe_load((ROOT / "configs/longitudinal.yaml").read_text(encoding="utf-8"))
    assert config == external
    assert '"run_longitudinal.py"' in run_all
    assert '"prepare_longitudinal.py"' not in run_all
    assert config["mhn"]["enabled"] is False
    assert config["mhn"]["fallback"] == "frequency_cooccurrence_backbone"


def test_e17_config_contains_only_selected_cohorts() -> None:
    config = yaml.safe_load((ROOT / "configs" / "longitudinal.yaml").read_text(encoding="utf-8"))
    assert set(config["studies"]) == {"difg_glass", "coadread_mskcc", "mnm_washu_2016"}
    assert "reported_sensitivity_studies" not in config
