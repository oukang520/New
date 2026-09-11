from pathlib import Path


def config_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "parameters" / name
