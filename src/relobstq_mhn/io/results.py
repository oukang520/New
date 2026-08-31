"""Consistent machine-readable output for all public workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class ResultWriter:
    """Write tables and metadata under one result directory.

    Workflows return DataFrames to callers and optionally use this writer.
    Plotting is deliberately outside this class and outside the workflows.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []

    def table(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.tables / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        self._files.append(path)
        return path

    def json(self, name: str, value: Any) -> Path:
        path = self.root / f"{name}.json"
        if is_dataclass(value):
            value = asdict(value)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        self._files.append(path)
        return path

    def manifest(self) -> Path:
        rows = []
        for path in sorted(set(self._files)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
        path = self.root / "result_manifest.tsv"
        pd.DataFrame(rows, columns=["path", "size_bytes", "sha256"]).to_csv(path, sep="\t", index=False)
        return path
