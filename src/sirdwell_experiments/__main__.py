"""Run the numerical contracts accompanying the selected manuscript panels."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "experiment",
        choices=[
            "e01_02",
            "e03",
            "e04",
            "e05_split_input",
            "e09",
            "e13",
            "e17",
            "e17_tables",
        ],
    )
    parser.add_argument(
        "--workspace", type=Path, help="Resolve relative input/output paths here."
    )
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    if args.workspace is not None:
        os.chdir(args.workspace.resolve(strict=True))
    module = importlib.import_module(f"sirdwell_experiments.{args.experiment}")
    sys.argv = [args.experiment, *remaining]
    module.main()


if __name__ == "__main__":
    main()
