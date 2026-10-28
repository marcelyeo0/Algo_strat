"""Simulated prop-firm account evaluated on daily bars.

Each session the engine reports three numbers in account currency:

* ``balance``      closed P&L, swaps and commissions included, at the session close;
* ``equity``       balance plus floating P&L at the session close;
* ``equity_worst`` lowest equity reached during the session. Intraday equity is not observed
  with daily bars, so it is approximated by marking every open position at the adverse
  extreme of the bar (low for longs, high for shorts) at the same time. This is pessimistic.

Daily-loss reference: FTMO resets the daily loss at midnight Europe/Prague, about one hour
before the 00:00 server-time bar close. The reference of session *d* is therefore taken from
the close of session *d-1* (see docs/GUIDE.md, "Hypothèses").

A breach is recorded as soon as equity *reaches* a floor (``<=``), the conservative reading
of "must not decline below".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from common.propfirm.rules import Phase, PropRules


class Status(StrEnum):
    ACTIVE = "active"
    PASSED = "passed"
    FAILED_DAILY = "failed_daily_loss"
    FAILED_TOTAL = "failed_total_loss"


@dataclass
class Violation:
    date: pd.Timestamp
    cause: str
    equity_worst: float
    floor: float
    reference: float


@dataclass
class DayRecord:
    date: pd.Timestamp
    balance: float
    equity: float
    equity_worst: float
    daily_reference: float
    daily_floor: float
    total_floor: float
    daily_loss_used: float      # share of the daily limit consumed by the worst intraday equity
    traded: bool


@dataclass
class PropAccount:
    rules: PropRules
    phase: Phase | None = None
    status: Status = Status.ACTIVE
    balance: float = field(init=False)
    equity: float = field(init=False)
    trading_days: int = 0
    high_water_eod: float = field(init=False)
    best_day_profit: float = 0.0
    violation: Violation | None = None
    passed_on: pd.Timestamp | None = None
    history: list[DayRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.balance = self.rules.initial_capital
        self.equity = self.rules.initial_capital
        self.high_water_eod = self.rules.initial_capital

    # --- floors -----------------------------------------------------------------------------

    def daily_reference(self) -> float:
        ref = self.rules.daily_reference
        if ref == "balance":
            return self.balance
        if ref == "equity":
            return self.equity
        return max(self.balance, self.equity)

    def daily_floor(self) -> float:
        return self.daily_reference() - self.rules.daily_loss_amount

    def total_floor(self) -> float:
        initial = self.rules.initial_capital
        if self.rules.total_loss_type == "trailing_eod":
            floor = self.high_water_eod - self.rules.total_loss_amount
            return min(floor, initial) if self.rules.lock_at_initial else floor
        return initial - self.rules.total_loss_amount

    # --- daily step -------------------------------------------------------------------------

    def step(self, date: pd.Timestamp, balance: float, equity: float, equity_worst: float, traded: bool) -> Status:
        """Process one session. Floors are those in force at the start of the session."""
        if self.status is not Status.ACTIVE:
            return self.status
        reference = self.daily_reference()
        d_floor = self.daily_floor()
        t_floor = self.total_floor()
        worst = min(equity_worst, equity)
        prev_equity = self.equity
        self.history.append(DayRecord(
            date=date, balance=balance, equity=equity, equity_worst=worst, daily_reference=reference,
            daily_floor=d_floor, total_floor=t_floor,
            daily_loss_used=max(0.0, reference - worst) / self.rules.daily_loss_amount, traded=traded,
        ))
        if worst <= d_floor:
            self.status = Status.FAILED_DAILY
            self.violation = Violation(date, "daily_loss", worst, d_floor, reference)
        elif worst <= t_floor:
            self.status = Status.FAILED_TOTAL
            self.violation = Violation(date, "total_loss", worst, t_floor, reference)
        self.balance, self.equity = balance, equity
        if self.status is not Status.ACTIVE:
            return self.status
        self.trading_days += int(traded)
        basis = balance if self.rules.trail_basis == "balance" else equity
        self.high_water_eod = max(self.high_water_eod, basis)
        self.best_day_profit = max(self.best_day_profit, equity - prev_equity)
        if self.phase is not None and self._target_reached():
            self.status = Status.PASSED
            self.passed_on = date
        return self.status

    def _target_reached(self) -> bool:
        initial = self.rules.initial_capital
        profit = self.equity - initial
        if profit < self.phase.profit_target_pct * initial * (1 - 1e-9):
            return False
        if self.trading_days < self.phase.min_trading_days:
            return False
        if self.rules.best_day_rule_pct is not None and self.best_day_profit > self.rules.best_day_rule_pct * profit:
            return False
        return True

    def records(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.history]).set_index("date") if self.history else pd.DataFrame()
