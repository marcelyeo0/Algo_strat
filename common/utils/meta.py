"""Run metadata: every run writes ``results/<run_id>/meta.json`` for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from common.utils.paths import repo_root, results_dir

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "hmmlearn",
    "optuna",
    "torch",
    "yfinance",
)


def git_commit() -> str:
    """Return the current commit hash, suffixed with ``-dirty`` if the tree has local changes."""
    try:
        root = repo_root()
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a configuration dictionary (used to lock holdout runs)."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def new_run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def write_meta(run_id: str, config: dict[str, Any], seeds: dict[str, int], extra: dict | None = None,
               cost_profile: str | None = None) -> Path:
    """Write ``meta.json`` into ``results/<run_id>/`` and return the run directory.

    The cost profile and its banner ("Résultats provisoires — coûts estimés" for the
    estimates profile) come first so that they are visible at the top of the file.
    """
    from common.utils.settings import cost_profile as active_profile
    from common.utils.settings import profile_banner

    profile = cost_profile or active_profile()
    run_dir = results_dir() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "banner": profile_banner(profile),
        "cost_profile": profile,
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "config": config,
        "seeds": seeds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    if extra:
        meta.update(extra)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return run_dir
