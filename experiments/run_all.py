"""Run numerical dependencies for the 42 figures and one reference table."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def commands(root: Path) -> list[list[str]]:
    def module(name):
        return [sys.executable, "-m", "sirdwell_experiments", name]

    def script(name, config):
        return [
            sys.executable,
            str(root / "experiments" / name),
            "--config",
            str(root / "configs" / config),
        ]

    return [
        module("e01_02"),
        module("e03"),
        module("e04"),
        module("e05_split_input"),
        script("prepare_cross_sectional.py", "cross_sectional_preparation.yaml"),
        script("run_cross_sectional.py", "cross_sectional.yaml"),
        script("run_simulation.py", "simulation.yaml"),
        script("run_secondary.py", "secondary.yaml"),
        module("e09"),
        module("e13"),
        script("run_longitudinal.py", "longitudinal.yaml"),
        module("e17_tables"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve(strict=True)
    for command in commands(root):
        if args.dry_run:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command, cwd=workspace, check=True)


if __name__ == "__main__":
    main()
