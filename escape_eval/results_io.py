"""Provenance and atomic writers for results/.

Every analysis writes results/{analysis_id}/ with parquet + csv tables (each
table carries a `preliminary` column) and a params.json, and is registered in
results/manifest.json as analysis_id -> {path, kind, created, preliminary}.
All writes go through escape_common.io (temp file, fsync, os.replace).
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
from pathlib import Path

import pandas as pd

from escape_common.io import atomic_to_parquet, atomic_write_json, atomic_write_text, update_json

SMOKE_DIRNAME = "smoke"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_info(root) -> dict:
    """HEAD commit and a dirty flag (tracked changes OR untracked files)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=60
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True, timeout=120
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None, "git_status_lines": None}
    lines = [line for line in status.splitlines() if line.strip()]
    return {"git_commit": commit, "git_dirty": bool(lines), "git_status_lines": len(lines)}


def package_versions() -> dict:
    import numpy
    import pyarrow
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
    }


def effective_results_root(results_root, smoke: bool) -> Path:
    """SMOKE outputs live under results/smoke/, never next to real analyses."""
    root = Path(results_root)
    return root / SMOKE_DIRNAME if smoke else root


def write_table(df: pd.DataFrame, directory, name: str, *, preliminary: bool = True) -> list:
    """Writes {name}.parquet and {name}.csv atomically; adds preliminary=True."""
    out = df.copy()
    if "preliminary" not in out.columns:
        out["preliminary"] = preliminary
    directory = Path(directory)
    atomic_to_parquet(out, directory / f"{name}.parquet")
    atomic_write_text(directory / f"{name}.csv", out.to_csv(index=False))
    return [f"{name}.parquet", f"{name}.csv"]


def write_json(obj, path) -> None:
    atomic_write_json(path, obj)


def register_analysis(results_root, analysis_id: str, analysis_dir, kind: str, *, smoke: bool = False,
                      extra: dict | None = None) -> dict:
    """Adds/replaces analysis_id in {results_root}/manifest.json (locked update)."""
    results_root = Path(results_root)
    entry = {
        "path": os.path.relpath(Path(analysis_dir), results_root),
        "kind": kind,
        "created": utc_now_iso(),
        "preliminary": True,
    }
    if smoke:
        entry["smoke"] = True
    if extra:
        entry.update(extra)

    def _update(current: dict) -> dict:
        current = dict(current)
        current[analysis_id] = entry
        return current

    update_json(results_root / "manifest.json", _update)
    return entry
