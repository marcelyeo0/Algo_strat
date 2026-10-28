"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from common.utils.paths import configs_dir


def load_yaml(name: str | Path) -> dict[str, Any]:
    """Load a YAML file. Relative names are resolved inside ``configs/``."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = configs_dir() / path
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_universe() -> dict[str, dict[str, Any]]:
    """Return the instrument universe as ``{symbol: spec}``."""
    cfg = load_yaml("universe.yaml")
    return cfg["instruments"]
