"""Challenge simulator: start a fresh challenge at the beginning of every month of a test period.

Each phase is run on a *fresh* account: the caller provides a ``session_factory(start)`` that
replays the strategy from ``start`` on a new ledger funded with the prop initial capital,
opening the target positions of the current signal at the open of ``start`` with all entry
costs, and yields one session record per day (``date``, ``balance``, ``equity``,
``equity_worst``, ``traded``). No P&L of previously open positions is inherited.

Phases are run without a time limit until they pass, fail or the data ends; pass rates are
then computed for several horizons (63, 126 and 252 sessions, and unlimited).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common.propfirm.account import PropAccount, Status
from common.propfirm.rules import Phase, PropRules

DEFAULT_HORIZONS: tuple[int | None, ...] = (63, 126, 252, None)
DEFAULT_HORIZON = 126

SessionFactory = Callable[[pd.Timestamp], Iterable[Any]]


@dataclass
class PhaseOutcome:
    result: str          # passed | failed_daily_loss | failed_total_loss | timeout (calendar limit) | incomplete
    days: int            # sessions spent in the phase
    end_date: pd.Timestamp | None
    violation_cause: str | None = None


def _get(rec: Any, key: str):
    return rec[key] if isinstance(rec, dict) else getattr(rec, key)


def run_phase(sessions: Iterable[Any], rules: PropRules, phase: Phase) -> PhaseOutcome:
    account = PropAccount(rules, phase)
    n = 0
    first = last = None
    for rec in sessions:
        date = pd.Timestamp(_get(rec, "date"))
        first = first if first is not None else date
        if phase.max_calendar_days is not None and (date - first).days > phase.max_calendar_days:
            return PhaseOutcome("timeout", n, last)
        status = account.step(date, float(_get(rec, "balance")), float(_get(rec, "equity")),
                              float(_get(rec, "equity_worst")), bool(_get(rec, "traded")))
        n += 1
        last = date
        if status is Status.PASSED:
            return PhaseOutcome("passed", n, date)
        if status in (Status.FAILED_DAILY, Status.FAILED_TOTAL):
            return PhaseOutcome(status.value, n, date, account.violation.cause)
    return PhaseOutcome("incomplete", n, last)


def simulate_challenges(
    session_factory: SessionFactory,
    sessions: pd.DatetimeIndex,
    rules: PropRules,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per monthly start with the (unbounded) outcome and duration of each phase.

    ``sessions`` is the trading calendar of the strategy; starts are the first session of each
    month inside [start, end]. A challenge may run past ``end`` while data exists.
    """
    sessions = pd.DatetimeIndex(sessions).sort_values()
    lo = pd.Timestamp(start) if start is not None else sessions[0]
    hi = pd.Timestamp(end) if end is not None else sessions[-1]
    rows = []
    for m in pd.date_range(lo.to_period("M").to_timestamp(), hi, freq="MS"):
        pos = int(sessions.searchsorted(max(m, lo)))
        if pos >= len(sessions) or sessions[pos] > hi:
            continue
        row: dict[str, Any] = {"start": sessions[pos]}
        cursor = pos
        for k, phase in enumerate(rules.phases, start=1):
            out = run_phase(session_factory(sessions[cursor]), rules, phase)
            row[f"phase{k}_result"] = out.result
            row[f"phase{k}_days"] = out.days
            row[f"phase{k}_end"] = out.end_date
            if out.result != "passed":
                break
            cursor = int(sessions.searchsorted(out.end_date, side="right"))
            if cursor >= len(sessions):
                if k < len(rules.phases):
                    row[f"phase{k + 1}_result"] = "incomplete"
                    row[f"phase{k + 1}_days"] = 0
                break
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_at_horizon(row: pd.Series, n_phases: int, horizon: int | None) -> tuple[str, int]:
    """Overall result and total sessions of one start when each phase is capped at ``horizon``."""
    total = 0
    for k in range(1, n_phases + 1):
        result = row.get(f"phase{k}_result")
        days = row.get(f"phase{k}_days")
        if not isinstance(result, str):
            return "incomplete", total
        days = int(days)
        if horizon is not None and days > horizon:
            return "timeout", total + horizon
        if horizon is not None and result == "incomplete" and days >= horizon:
            return "timeout", total + horizon
        total += days
        if result != "passed":
            return result, total
    return "passed", total


def summarise(outcomes: pd.DataFrame, n_phases: int, horizon: int | None = DEFAULT_HORIZON) -> dict[str, Any]:
    """Pass / fail / timeout rates at ``horizon`` over the starts whose outcome is known."""
    label = "unlimited" if horizon is None else str(horizon)
    if outcomes.empty:
        return {"horizon": label, "n_starts": 0}
    res = [outcome_at_horizon(r, n_phases, horizon) for _, r in outcomes.iterrows()]
    results = pd.Series([r for r, _ in res])
    days = pd.Series([d for _, d in res])
    p1 = [outcome_at_horizon(r, 1, horizon) for _, r in outcomes.iterrows()]
    p1_res = pd.Series([r for r, _ in p1])
    p1_days = pd.Series([d for _, d in p1])
    known = results != "incomplete"
    n = int(known.sum())
    if n == 0:
        return {"horizon": label, "n_starts": 0, "n_incomplete": len(results)}
    k = results[known]
    passed = results == "passed"
    p1_passed = (p1_res == "passed") & known
    return {
        "horizon": label,
        "n_starts": n,
        "n_incomplete": int((~known).sum()),
        "pass_phase1": float(p1_passed.sum() / n),
        "pass_all": float((k == "passed").mean()),
        "fail": float(k.str.startswith("failed").mean()),
        "fail_daily": float((k == "failed_daily_loss").mean()),
        "fail_total": float((k == "failed_total_loss").mean()),
        "timeout": float((k == "timeout").mean()),
        "median_sessions_phase1": float(np.median(p1_days[p1_passed])) if p1_passed.any() else float("nan"),
        "median_sessions_all": float(np.median(days[passed])) if passed.any() else float("nan"),
    }


def summarise_horizons(outcomes: pd.DataFrame, n_phases: int,
                       horizons: tuple[int | None, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    return pd.DataFrame([summarise(outcomes, n_phases, h) for h in horizons])
