"""Run the currently canonical core workflows in dependency order.

This command does not claim to regenerate every historical E1-E17 table. See
RESULT_PROVENANCE_MATRIX.tsv for experiment-level coverage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scripts = [
        "prepare_cross_sectional.py",
        "run_cross_sectional.py",
        "run_secondary.py",
        "run_simulation.py",
        "run_topology_robustness.py",
        "run_longitudinal.py",
    ]
    for script in scripts:
        if args.dry_run:
            print(f"{sys.executable} experiments/{script}")
            continue
        subprocess.run([sys.executable, str(root / "experiments" / script)], cwd=root, check=True)


if __name__ == "__main__":
    main()
