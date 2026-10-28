"""Repository paths, resolved at runtime with pathlib (no hard-coded absolute paths)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    The ``ALGO_ROOT`` environment variable takes precedence; otherwise we walk up
    from this file until we find ``pyproject.toml``.
    """
    env = os.environ.get("ALGO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root (pyproject.toml not found).")


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    path = Path(os.environ.get("ALGO_DATA_DIR", repo_root() / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    path = Path(os.environ.get("ALGO_RESULTS_DIR", repo_root() / "results"))
    path.mkdir(parents=True, exist_ok=True)
    return path
