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

    # Pre-aligned numpy arrays (the per-session loop only touches plain floats).
    n = len(calendar)
    cols = ("open", "high", "low", "close")
    arr = {s: frames[s].reindex(calendar)[list(cols)].to_numpy(float) for s in symbols}
    has_bar = {s: calendar.isin(frames[s].index) for s in symbols}
    spread_arr = {s: (frames[s]["spread"].reindex(calendar).to_numpy(float) if "spread" in frames[s]
                      else np.full(n, np.nan)) for s in symbols}
    ratio_arr = {s: ratio[s].to_numpy(float) for s in symbols}
    fx_arr = {s: fx[s].to_numpy(float) for s in symbols}
    w_al = weights.reindex(calendar)
    is_signal = calendar.isin(weights.index)
    w_arr = {s: w_al[s].to_numpy(float) for s in symbols}
    stop_arr = ({s: stops.reindex(calendar)[s].to_numpy(float) if s in stops.columns else np.full(n, np.nan)
                 for s in symbols} if stops is not None else None)

    ledger = Ledger(costs, balance=capital)
    last_close = {s: np.nan for s in symbols}
    rows, fills, held_rows = [], [], []
    peak = capital
    for k in range(1, n):
        t, d = calendar[k - 1], calendar[k]
        bars = {}
        for s in symbols:
            if has_bar[s][k]:
                o, h, lo_, c = arr[s][k]
                last_close[s] = c
                sp = spread_arr[s][k]
                r = ratio_arr[s][k]
                bars[s] = Bar(o, h, lo_, c, fx_arr[s][k], None if np.isnan(sp) else sp, None if np.isnan(r) else r)
            elif not np.isnan(last_close[s]):
                c = last_close[s]
                bars[s] = Bar(c, c, c, c, fx_arr[s][k])
            else:
                bars[s] = Bar(np.nan, np.nan, np.nan, np.nan, 1.0)
        targets: dict[str, float] = {}
        target_stops: dict[str, float | None] = {}
        if is_signal[k - 1] and lo <= d:
            mult = exposure_fn(t, rows[-1]["equity"] if rows else capital, peak) if exposure_fn else 1.0
            for s in symbols:
                if not (has_bar[s][k] and has_bar[s][k - 1]):
                    continue
                w = w_arr[s][k - 1]
                w = 0.0 if np.isnan(w) else w * mult
                px_t = arr[s][k - 1][3]
                fx_t = fx_arr[s][k - 1]
                pos = ledger.positions.get(s)
                current_w = (pos.units * px_t * fx_t / capital) if pos else 0.0
                if abs(w - current_w) <= band and not (w == 0.0 and current_w != 0.0):
                    continue
                targets[s] = w * capital / (px_t * fx_t)
                if stop_arr is not None and not np.isnan(stop_arr[s][k - 1]):
                    target_stops[s] = float(stop_arr[s][k - 1])
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
