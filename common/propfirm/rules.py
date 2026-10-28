"""Prop-firm rule presets loaded from ``configs/propfirm/<preset>.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.utils.config import load_yaml, warn_unverified


@dataclass(frozen=True)
class Phase:
    name: str
    profit_target_pct: float
    min_trading_days: int = 0
    max_calendar_days: int | None = None


@dataclass(frozen=True)
class PropRules:
    name: str
    initial_capital: float
    daily_loss_pct: float
    daily_reference: str = "max_balance_equity"      # max_balance_equity | balance | equity
    total_loss_type: str = "static"                  # static | trailing_eod
    total_loss_pct: float = 0.10
    trail_basis: str = "balance"
    lock_at_initial: bool = True
    phases: tuple[Phase, ...] = ()
    best_day_rule_pct: float | None = None
    leverage: dict[str, float] = field(default_factory=dict)

    @property
    def daily_loss_amount(self) -> float:
        return self.daily_loss_pct * self.initial_capital

    @property
    def total_loss_amount(self) -> float:
        return self.total_loss_pct * self.initial_capital

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> PropRules:
        dl, tl = cfg["daily_loss"], cfg["total_loss"]
        return cls(
            name=cfg["name"],
            initial_capital=float(cfg["initial_capital"]),
            daily_loss_pct=float(dl["pct_of_initial"]),
            daily_reference=dl.get("reference", "max_balance_equity"),
            total_loss_type=tl.get("type", "static"),
            total_loss_pct=float(tl["pct_of_initial"]),
            trail_basis=tl.get("trail_basis", "balance"),
            lock_at_initial=bool(tl.get("lock_at_initial", True)),
            phases=tuple(Phase(**p) for p in cfg.get("phases", [])),
            best_day_rule_pct=cfg.get("best_day_rule_pct"),
            leverage={k: float(v) for k, v in cfg.get("leverage", {}).items()},
        )


def load_prop_rules(preset: str = "ftmo_2step_swing") -> PropRules:
    path = f"propfirm/{preset}.yaml"
    warn_unverified(path)
    return PropRules.from_dict(load_yaml(path))
