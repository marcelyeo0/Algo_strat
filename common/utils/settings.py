"""Global settings (configs/settings.yaml) and the active cost profile."""

from __future__ import annotations

from common.utils.config import load_yaml, resolve_config_path

COST_PROFILES = ("estimates", "mt5")
PROVISIONAL_BANNER = "Résultats provisoires — coûts estimés"


def settings() -> dict:
    return load_yaml("settings.yaml")


def cost_profile(override: str | None = None) -> str:
    """Active cost profile: CLI override first, then ``cost_profile`` in settings.yaml."""
    profile = override or settings().get("cost_profile", "estimates")
    if profile not in COST_PROFILES:
        raise ValueError(f"Unknown cost profile '{profile}', expected one of {COST_PROFILES}.")
    path = resolve_config_path(f"costs/{profile}.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"Cost profile '{profile}' not found ({path}). Generate it from the MT5 export with "
            "`uv run python -m common.costs.mt5_specs`."
        )
    return profile


def is_provisional(profile: str) -> bool:
    return profile == "estimates"


def profile_banner(profile: str) -> str:
    """Header line shown in meta.json, report.html and SUMMARY.md."""
    text = f"Profil de coûts : {profile}"
    return f"{text} — {PROVISIONAL_BANNER}" if is_provisional(profile) else text


def holdout_allowed(profile: str) -> bool:
    return profile in settings().get("holdout_allowed_cost_profiles", ["mt5"])
