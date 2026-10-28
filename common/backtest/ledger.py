"""Position ledger and the intraday sequence of one daily session.

Order of events inside session *d* (signals were computed at the close of *d-1*):

1. orders are filled at the open of *d* (spread and commission debited from the balance);
2. protective stops are checked against the bar; a gap through a stop fills at the open;
3. the lowest intraday equity is estimated by marking every position still open at the
   adverse extreme of the bar (stopped positions count at their fill price);
4. at the close, positions are marked to market and overnight swaps are charged on the
   positions held over the session close (three nights on the instrument's triple-swap day).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from common.costs.model import CostModel, stop_fill


@dataclass
class Position:
    units: float = 0.0
    avg_price: float = 0.0
    stop: float | None = None


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    fx: float = 1.0
    spread: float | None = None
    range_atr_ratio: float | None = None


@dataclass
class SessionResult:
    date: pd.Timestamp
    balance: float
    equity: float
    equity_worst: float
    traded: bool
    spread_cost: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    stops_hit: list[str] = field(default_factory=list)


@dataclass
class Ledger:
    costs: CostModel
    balance: float
    positions: dict[str, Position] = field(default_factory=dict)

    def _realise(self, symbol: str, delta: float, price: float, fx: float) -> None:
        pos = self.positions.setdefault(symbol, Position())
        if pos.units == 0 or (pos.units > 0) == (delta > 0):
            new_units = pos.units + delta
            pos.avg_price = (pos.avg_price * pos.units + price * delta) / new_units
            pos.units = new_units
            return
        closing = -delta if abs(delta) <= abs(pos.units) else pos.units
        self.balance += closing * (price - pos.avg_price) * fx
        remaining = pos.units + delta
        if remaining == 0 or (remaining > 0) == (pos.units > 0):
            pos.units = remaining
            if remaining == 0:
                pos.avg_price, pos.stop = 0.0, None
        else:  # position flipped: the excess opens at the fill price
            pos.units, pos.avg_price, pos.stop = remaining, price, None

    def floating(self, prices: dict[str, float], fx: dict[str, float]) -> float:
        return sum(p.units * (prices[s] - p.avg_price) * fx.get(s, 1.0) for s, p in self.positions.items() if p.units)

    def run_session(
        self,
        date: pd.Timestamp,
        bars: dict[str, Bar],
        target_units: dict[str, float] | None = None,
        stops: dict[str, float | None] | None = None,
    ) -> SessionResult:
        res = SessionResult(date=date, balance=0.0, equity=0.0, equity_worst=0.0, traded=False)

        # 1. fills at the open
        for symbol, target in (target_units or {}).items():
            bar = bars[symbol]
            current = self.positions.get(symbol, Position()).units
            delta = target - current
            if abs(delta) < 1e-12:
                continue
            spread = self.costs.spread_cost(symbol, delta, bar.fx, bar.spread, bar.range_atr_ratio)
            commission = self.costs.commission(symbol, delta)
            # Fill at the mid open; the half spread is booked explicitly as a cost.
            self._realise(symbol, delta, bar.open, bar.fx)
            self.balance -= spread + commission
            res.spread_cost += spread
            res.commission += commission
            res.traded = True
        for symbol, level in (stops or {}).items():
            if symbol in self.positions:
                self.positions[symbol].stop = level

        # 2-3. stops and worst intraday equity
        worst_floating = 0.0
        for symbol, pos in list(self.positions.items()):
            if pos.units == 0:
                continue
            bar = bars[symbol]
            side = 1 if pos.units > 0 else -1
            hit = None
            if pos.stop is not None:
                spec = self.costs.instruments[symbol]
                slip = spec.stop_slippage_spreads * self.costs.effective_spread(symbol, bar.spread, bar.range_atr_ratio)
                hit = stop_fill(side, pos.stop, bar.open, bar.high, bar.low, slippage=slip)
            if hit is not None:
                units = pos.units
                spread = self.costs.spread_cost(symbol, units, bar.fx, bar.spread, bar.range_atr_ratio)
                commission = self.costs.commission(symbol, units)
                self._realise(symbol, -units, hit, bar.fx)
                self.balance -= spread + commission
                res.spread_cost += spread
                res.commission += commission
                res.stops_hit.append(symbol)
                res.traded = True
                continue
            adverse = bar.low if side > 0 else bar.high
            worst_floating += pos.units * (adverse - pos.avg_price) * bar.fx
        res.equity_worst = self.balance + worst_floating

        # 4. close: swaps on positions held overnight, then mark to market
        weekday = date.weekday()
        for symbol, pos in self.positions.items():
            if pos.units:
                bar = bars[symbol]
                swap = self.costs.swap(symbol, pos.units, bar.close, weekday, bar.fx)
                self.balance += swap
                res.swap += swap
        res.balance = self.balance
        closes = {s: bars[s].close for s, p in self.positions.items() if p.units}
        fxs = {s: bars[s].fx for s in closes}
        res.equity = self.balance + self.floating(closes, fxs)
        res.equity_worst = min(res.equity_worst, res.equity)
        return res
