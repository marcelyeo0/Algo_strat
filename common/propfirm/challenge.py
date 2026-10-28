"""Challenge simulator: start a fresh challenge at the beginning of every month of a test period.

The input is the daily account path of a strategy run sized on the prop initial capital
(columns ``balance``, ``equity``, ``equity_worst``, ``traded``). For each start, P&L is
rebased so that the account begins at the initial capital; positions open at the start date
are inherited from the strategy run (approximation documented in docs/GUIDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.propfirm.account import PropAccount, Status
from common.propfirm.rules import PropRules

REQUIRED = ("balance", "equity", "equity_worst", "traded")


@dataclass
class PhaseOutcome:
    result: str          # passed | failed_daily_loss | failed_total_loss | timeout | incomplete
    days: int            # trading sessions spent in the phase
    end_date: pd.Timestamp | None
    violation_cause: str | None = None


def _run_phase(path: pd.DataFrame, start_pos: int, rules: PropRules, phase, horizon: int) -> PhaseOutcome:
    initial = rules.initial_capital
    if start_pos > 0:
        base_eq = path["equity"].iloc[start_pos - 1]
        base_bal = path["balance"].iloc[start_pos - 1]
    else:
        base_eq = base_bal = initial
    account = PropAccount(rules, phase)
    start_date = path.index[start_pos]
    n = 0
    for pos in range(start_pos, len(path)):
        row = path.iloc[pos]
        date = path.index[pos]
        if phase.max_calendar_days is not None and (date - start_date).days > phase.max_calendar_days:
            return PhaseOutcome("timeout", n, date)
        status = account.step(
            date,
            balance=initial + row["balance"] - base_bal,
            equity=initial + row["equity"] - base_eq,
            equity_worst=initial + row["equity_worst"] - base_eq,
            traded=bool(row["traded"]),
        )
        n += 1
        if status is Status.PASSED:
            return PhaseOutcome("passed", n, date)
        if status in (Status.FAILED_DAILY, Status.FAILED_TOTAL):
            return PhaseOutcome(status.value, n, date, account.violation.cause)
        if n >= horizon:
            return PhaseOutcome("timeout", n, date)
    return PhaseOutcome("incomplete", n, None)


def simulate_challenges(
    path: pd.DataFrame,
    rules: PropRules,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    horizon_sessions: int = 126,
) -> pd.DataFrame:
    """One row per monthly start date with the outcome of each phase.

    ``horizon_sessions`` caps each phase (FTMO has no time limit; the cap defines "timeout").
    Starts are taken inside [start, end]; a challenge may run past ``end`` if data exists.
    """
    missing = [c for c in REQUIRED if c not in path]
    if missing:
        raise ValueError(f"path is missing columns {missing}")
    path = path.sort_index()
    idx = path.index
    lo = pd.Timestamp(start) if start is not None else idx[0]
    hi = pd.Timestamp(end) if end is not None else idx[-1]
    months = pd.date_range(lo.to_period("M").to_timestamp(), hi, freq="MS")
    rows = []
    for m in months:
        pos = int(idx.searchsorted(max(m, lo)))
        if pos >= len(idx) or idx[pos] > hi:
            continue
        row = {"start": idx[pos]}
        cursor, total_days, final = pos, 0, "passed"
        for k, phase in enumerate(rules.phases, start=1):
            out = _run_phase(path, cursor, rules, phase, horizon_sessions)
            row[f"phase{k}_result"] = out.result
            row[f"phase{k}_days"] = out.days
            total_days += out.days
            if out.result != "passed":
                final = out.result
                break
            cursor = idx.searchsorted(out.end_date, side="right")
            if cursor >= len(idx):
                final = "incomplete" if k < len(rules.phases) else "passed"
                break
        row["result"] = final
        row["total_days"] = total_days
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(outcomes: pd.DataFrame) -> dict[str, float]:
    """Pass / fail / timeout rates over the starts whose outcome is known."""
    if outcomes.empty:
        return {"n_starts": 0}
    known = outcomes[outcomes["result"] != "incomplete"]
    n = len(known)
    if n == 0:
        return {"n_starts": 0, "n_incomplete": len(outcomes)}
    p1 = known["phase1_result"] == "passed"
    passed = known["result"] == "passed"
    failed = known["result"].str.startswith("failed")
    return {
        "n_starts": n,
        "n_incomplete": int((outcomes["result"] == "incomplete").sum()),
        "pass_phase1": float(p1.mean()),
        "pass_all": float(passed.mean()),
        "fail": float(failed.mean()),
        "fail_daily": float((known["result"] == "failed_daily_loss").mean()),
        "fail_total": float((known["result"] == "failed_total_loss").mean()),
        "timeout": float((known["result"] == "timeout").mean()),
        "median_days_phase1": float(np.median(known.loc[p1, "phase1_days"])) if p1.any() else float("nan"),
        "median_days_all": float(np.median(known.loc[passed, "total_days"])) if passed.any() else float("nan"),
    }
