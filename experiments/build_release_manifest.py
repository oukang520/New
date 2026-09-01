"""Build the file-level SHA-256 manifest for the distributable repository."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.tsv"
EXCLUDED_TOP_LEVEL = {
    ".git",
    ".pytest_cache",
    ".conda_reaudit_py311",
    ".venv_reaudit",
    "build",
    "data",
    "outputs",
    "tmp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    rows = ["path\tsize_bytes\tsha256"]
    for path in included_files():
        relative = path.relative_to(ROOT).as_posix()
        rows.append(f"{relative}\t{path.stat().st_size}\t{sha256(path)}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows) - 1} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
