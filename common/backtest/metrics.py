"""Performance metrics of engine runs, in % of the initial prop capital.

Prop accounts size positions on the initial balance, so daily returns are P&L divided by the
initial capital (not by the current equity, which would explode once the account is far below
its starting balance).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from common.backtest.engine import EngineResult, run_engine
from common.costs.model import CostModel

ANNUAL = 252


def daily_returns(sessions: pd.DataFrame, capital: float, gross: bool = False) -> pd.Series:
    """Daily P&L / initial capital; ``gross`` adds back spreads, commissions and swaps."""
    pnl = sessions["equity"].diff()
    pnl.iloc[0] = sessions["equity"].iloc[0] - capital
    if gross:
        pnl = pnl + sessions["spread_cost"] + sessions["commission"] - sessions["swap"]
    return pnl / capital


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANNUAL)) if len(r) > 2 and r.std() > 0 else float("nan")


def summary(res: EngineResult, capital: float) -> dict[str, float]:
    s = res.sessions
    r, g = daily_returns(s, capital), daily_returns(s, capital, gross=True)
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    dd = (s["equity"] - s["equity"].cummax()) / capital
    return {
        "start": s.index[0].date().isoformat(), "end": s.index[-1].date().isoformat(),
        "sharpe_gross": sharpe(g), "sharpe_net": sharpe(r),
        "ann_return": float(r.mean() * ANNUAL), "ann_vol": float(r.std() * np.sqrt(ANNUAL)),
        "max_dd": float(dd.min()), "worst_day": float(r.min()),
        "worst_intraday": float(((s["equity_worst"] - s["equity"].shift(1).fillna(capital)) / capital).min()),
        "turnover": float(res.held.diff().abs().sum(axis=1).sum() / years),
        "cost_spread": float(s["spread_cost"].sum()), "cost_commission": float(s["commission"].sum()),
        "cost_swap": float(-s["swap"].sum()),
    }


def challenge_factory(weights: pd.DataFrame, panel: dict[str, pd.DataFrame], costs: CostModel, capital: float,
                      execution: str, band: float = 0.0, max_sessions: int = 260):
    """Session factory for the challenge simulator: a fresh account at every start.

    At the start date the target weights of the current signal (last weights issued before the
    start) are opened, with every entry cost, then the strategy runs normally.
    """
    calendar = pd.DatetimeIndex(sorted(set().union(*[panel[s].index for s in weights.columns])))

    def factory(start: pd.Timestamp) -> Iterator[dict]:
        pos = int(calendar.searchsorted(start))
        if pos == 0:
            return iter(())
        t = calendar[pos - 1]
        w = weights
        if t not in w.index:
            last = w.loc[:t]
            if last.empty:
                return iter(())
            w = pd.concat([w, last.iloc[[-1]].set_axis([t])]).sort_index()
        end = calendar[min(pos + max_sessions, len(calendar) - 1)]
        res = run_engine(w, panel, costs, capital, execution, start=start, end=end, band=band)
        return (dict(date=d, **row) for d, row in res.sessions.iterrows())

    return factory, calendar
