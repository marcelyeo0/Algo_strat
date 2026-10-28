"""Trading cost model: spreads, commissions, overnight swaps and stop slippage.

Amounts are returned in account currency (USD). ``fx`` is the quote-currency to USD rate of
the instrument on the relevant day (1.0 for USD-quoted instruments).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from common.utils.config import load_yaml, warn_unverified

DAYS_PER_YEAR = 360.0


@dataclass(frozen=True)
class InstrumentCost:
    symbol: str
    quote_ccy: str
    contract_size: float
    spread: float
    commission_per_lot: float
    swap_long: float          # % of notional per year, negative = debit
    swap_short: float
    triple_swap_day: int      # 0 = Monday
    stop_slippage_spreads: float = 0.5


@dataclass(frozen=True)
class SpreadWidening:
    enabled: bool = True
    range_atr_threshold: float = 2.0
    factor: float = 2.0


class CostModel:
    def __init__(self, instruments: dict[str, InstrumentCost], widening: SpreadWidening, multiplier: float = 1.0):
        self.instruments = instruments
        self.widening = widening
        self.multiplier = multiplier

    @classmethod
    def from_yaml(cls, name: str = "costs.yaml", multiplier: float = 1.0) -> CostModel:
        warn_unverified(name)
        cfg = load_yaml(name)
        defaults: dict[str, Any] = cfg.get("defaults", {})
        widening = SpreadWidening(**defaults.get("spread_widening", {}))
        slip = float(defaults.get("stop_slippage_spreads", 0.5))
        instruments = {
            sym: InstrumentCost(symbol=sym, stop_slippage_spreads=float(spec.pop("stop_slippage_spreads", slip)), **spec)
            for sym, spec in ((s, dict(v)) for s, v in cfg["instruments"].items())
        }
        return cls(instruments, widening, multiplier)

    def scaled(self, multiplier: float) -> CostModel:
        """Same model with every cost multiplied (sensitivity tests at x1.5 and x2)."""
        return CostModel(self.instruments, self.widening, self.multiplier * multiplier)

    def with_instrument(self, symbol: str, **changes: Any) -> CostModel:
        inst = dict(self.instruments)
        inst[symbol] = replace(inst[symbol], **changes)
        return CostModel(inst, self.widening, self.multiplier)

    # --- per-trade costs -------------------------------------------------------------------

    def effective_spread(self, symbol: str, observed: float | None = None, range_atr_ratio: float | None = None) -> float:
        """Full spread in price units, widened on high-volatility days."""
        spec = self.instruments[symbol]
        spread = observed if observed is not None and observed == observed and observed > 0 else spec.spread
        if self.widening.enabled and range_atr_ratio is not None and range_atr_ratio > self.widening.range_atr_threshold:
            spread *= self.widening.factor
        return spread * self.multiplier

    def spread_cost(self, symbol: str, units: float, fx: float = 1.0, observed: float | None = None,
                    range_atr_ratio: float | None = None) -> float:
        """Cost of crossing half the spread for ``units`` traded (either direction)."""
        return abs(units) * 0.5 * self.effective_spread(symbol, observed, range_atr_ratio) * fx

    def commission(self, symbol: str, units: float) -> float:
        spec = self.instruments[symbol]
        return abs(units) / spec.contract_size * spec.commission_per_lot * self.multiplier

    # --- overnight financing ----------------------------------------------------------------

    def swap_nights(self, symbol: str, weekday: int) -> int:
        """Nights charged for a position held over the close of a session on ``weekday``."""
        return 3 if weekday == self.instruments[symbol].triple_swap_day else 1

    def swap(self, symbol: str, units: float, price: float, weekday: int, fx: float = 1.0) -> float:
        """Signed swap (negative = debit) for holding ``units`` over the session close."""
        if units == 0:
            return 0.0
        spec = self.instruments[symbol]
        rate = spec.swap_long if units > 0 else spec.swap_short
        notional = abs(units) * price * fx
        amount = notional * rate / 100.0 / DAYS_PER_YEAR * self.swap_nights(symbol, weekday)
        # Debits scale with the cost multiplier; credits are never inflated.
        return amount * self.multiplier if amount < 0 else amount


def stop_fill(side: int, stop: float, open_: float, high: float, low: float, slippage: float = 0.0) -> float | None:
    """Execution price of a protective stop on a daily bar, or ``None`` if it is not hit.

    ``side`` is the side of the *position* (+1 long, -1 short). When the bar opens beyond the
    stop (gap, including weekend gaps) the fill is the open, i.e. the worse of the two levels.
    ``slippage`` (price units) is added against the position.
    """
    if side > 0:
        if open_ <= stop:
            return open_ - slippage
        if low <= stop:
            return stop - slippage
        return None
    if open_ >= stop:
        return open_ + slippage
    if high >= stop:
        return stop + slippage
    return None
