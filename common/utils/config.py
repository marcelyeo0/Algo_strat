"""YAML configuration loading."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from common.utils.paths import configs_dir


class UnverifiedConfigWarning(UserWarning):
    """Emitted while a configuration still contains values marked ``TODO: vérifier``."""


def resolve_config_path(name: str | Path) -> Path:
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = configs_dir() / path
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    return path


def load_yaml(name: str | Path) -> dict[str, Any]:
    """Load a YAML file. Relative names are resolved inside ``configs/``."""
    with resolve_config_path(name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def warn_unverified(name: str | Path) -> int:
    """Warn once per call if the YAML file still contains unverified values; return their count."""
    path = resolve_config_path(name)
    count = sum("TODO" in line for line in path.read_text(encoding="utf-8").splitlines())
    if count:
        warnings.warn(
            f"{path.name}: {count} value(s) still marked 'TODO: vérifier' - results depend on unverified rules.",
            UnverifiedConfigWarning,
            stacklevel=2,
        )
    return count


def load_universe(include_auxiliary: bool = False) -> dict[str, dict[str, Any]]:
    """Return the tradable universe as ``{symbol: spec}``.

    Auxiliary series (FX rates only used to convert P&L, such as USDHKD) are added when
    ``include_auxiliary`` is true; they are never traded.
    """
    cfg = load_yaml("universe.yaml")
    out = dict(cfg["instruments"])
    if include_auxiliary:
        out.update(cfg.get("auxiliary", {}))
    return out
