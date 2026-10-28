"""Event-driven daily backtest engine.

Target weights (fraction of capital, signed) are decided at the close of session t. They are
executed in session t+1 only, by one of two modes:

* ``next_open``  - at the open of t+1. Instruments whose source opens are unreliable (share of
  opens equal to the previous close above ``max_stale_open``) are executed at the close of t+1
  instead: a stale "open" is the signal close itself;
* ``next_close`` - at the close of t+1, with an explicit close-execution slippage.

No mode executes at the close that was used to compute the signal. Every fill is recorded with
the timestamp of its signal and of its execution so that this can be tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.backtest.ledger import Bar, Ledger
from common.costs.fx import quote_to_usd
from common.costs.model import CostModel

EXECUTIONS = ("next_open", "next_close")
# Intraday order of price events within a session, used to compare timestamps.
PHASE = {"open": 0, "close": 1}

ExposureFn = Callable[[pd.Timestamp, float, float], float]   # (date, equity, peak equity) -> multiplier


@dataclass
class EngineResult:
    sessions: pd.DataFrame      # balance, equity, equity_worst, traded, costs per session
    fills: pd.DataFrame         # symbol, signal_date, signal_phase, exec_date, exec_phase
    held: pd.DataFrame          # weights held at each close (fraction of capital)


def stale_open_share(bars: pd.DataFrame) -> float:
    """Share of sessions whose open equals the previous close (stale opens)."""
    prev = bars["close"].shift(1)
    same = (bars["open"] - prev).abs() <= 1e-12 * bars["close"].abs()
    return float(same.iloc[1:].mean()) if len(bars) > 1 else 0.0


def _atr_ratio(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    rng = bars["high"] - bars["low"]
    return (rng / rng.rolling(window, min_periods=5).mean()).shift(1)   # known before the session


def run_engine(
    weights: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    costs: CostModel,
    capital: float,
    execution: str = "next_open",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    stops: pd.DataFrame | None = None,
    band: float = 0.0,
    exposure_fn: ExposureFn | None = None,
    max_stale_open: float = 0.10,
) -> EngineResult:
    """Run the weights through the ledger on the sessions in [start, end].

    ``band`` is a no-trade band in weight units: an asset is only rebalanced when its target
    differs from the weight currently held by more than ``band`` (a full exit always trades).
    Sizing uses the fixed ``capital`` (prop accounts size on the initial balance).
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}; same-close execution is not allowed")
    symbols = list(weights.columns)
    frames = {s: panel[s] for s in symbols}
    calendar = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])))
    lo = start if start is not None else weights.index[0]
    hi = end if end is not None else calendar[-1]
    first = max(int(calendar.searchsorted(lo)) - 1, 0)    # the session before `lo` carries the first signal
    calendar = calendar[first:]
    calendar = calendar[calendar <= hi]
    fx = {s: quote_to_usd(costs.instruments[s].quote_ccy, panel, calendar) for s in symbols}
    ratio = {s: _atr_ratio(frames[s]).reindex(calendar) for s in symbols}
    fill_field = {s: "close" if execution == "next_close" or stale_open_share(frames[s]) > max_stale_open else "open"
                  for s in symbols}

    ledger = Ledger(costs, balance=capital)
    last_close = {s: np.nan for s in symbols}
    rows, fills, held_rows = [], [], []
    peak = capital
    for k in range(1, len(calendar)):
        t, d = calendar[k - 1], calendar[k]
        bars = {}
        tradable = set()
        for s in symbols:
            f = frames[s]
            if d in f.index:
                b = f.loc[d]
                last_close[s] = float(b["close"])
                spread = b["spread"] if "spread" in b and pd.notna(b["spread"]) else None
                r = ratio[s].get(d)
                bars[s] = Bar(float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"]),
                              float(fx[s].get(d)), spread, None if pd.isna(r) else float(r))
                tradable.add(s)
            elif not np.isnan(last_close[s]):
                c = last_close[s]
                bars[s] = Bar(c, c, c, c, float(fx[s].get(d)))
            else:
                bars[s] = Bar(np.nan, np.nan, np.nan, np.nan, 1.0)
        targets: dict[str, float] = {}
        target_stops: dict[str, float | None] = {}
        if t in weights.index and lo <= d:
            mult = exposure_fn(t, rows[-1]["equity"] if rows else capital, peak) if exposure_fn else 1.0
            for s in symbols:
                if s not in tradable or s not in frames or t not in frames[s].index:
                    continue
                w = weights.at[t, s]
                w = 0.0 if pd.isna(w) else float(w) * mult
                px_t = float(frames[s].at[t, "close"])
                fx_t = float(fx[s].get(t, fx[s].get(d)))
                pos = ledger.positions.get(s)
                current_w = (pos.units * px_t * fx_t / capital) if pos else 0.0
                if abs(w - current_w) <= band and not (w == 0.0 and current_w != 0.0):
                    continue
                targets[s] = w * capital / (px_t * fx_t)
                if stops is not None and t in stops.index and s in stops.columns and pd.notna(stops.at[t, s]):
                    target_stops[s] = float(stops.at[t, s])
        res = ledger.run_session(d, bars, targets, target_stops, fill=fill_field)
        for s, field in res.fills:
            fills.append({"symbol": s, "signal_date": t, "signal_phase": PHASE["close"],
                          "exec_date": d, "exec_phase": PHASE[field]})
        if d < lo:
            continue
        peak = max(peak, res.equity)
        rows.append({"date": d, "balance": res.balance, "equity": res.equity, "equity_worst": res.equity_worst,
                     "traded": res.traded, "spread_cost": res.spread_cost, "commission": res.commission,
                     "swap": res.swap, "stops_hit": len(res.stops_hit)})
        held_rows.append({"date": d, **{s: (p.units * bars[s].close * bars[s].fx / capital)
                                       for s, p in ledger.positions.items() if p.units}})
    sessions = pd.DataFrame(rows).set_index("date")
    held = pd.DataFrame(held_rows).set_index("date").reindex(columns=symbols).fillna(0.0)
    return EngineResult(sessions, pd.DataFrame(fills), held)
