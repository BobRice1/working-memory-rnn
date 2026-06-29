"""File-system helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_run_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create and return the standard output directories for one run."""
    root = Path(output_dir)
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "metrics": root / "metrics",
        "figures": root / "figures",
        "arrays": root / "arrays",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a dictionary as indented JSON and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> Path:
    """Write a list of metric rows to CSV and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return target
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return target
