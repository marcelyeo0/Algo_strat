"""Trading cost model: spreads, commissions, overnight swaps and stop slippage.

Every number comes from the active cost profile (``configs/costs/<profile>.yaml``); nothing is
hard-coded here. Amounts are returned in account currency (USD). ``fx`` is the quote-currency
to USD rate of the instrument on the relevant day (1.0 for USD-quoted instruments).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from common.utils.config import load_yaml, warn_unverified
from common.utils.settings import cost_profile

DAYS_PER_YEAR = 360.0


@dataclass(frozen=True)
class InstrumentCost:
    symbol: str
    quote_ccy: str
    contract_size: float
    spread: float
    commission_per_lot: float
    triple_swap_day: int              # 0 = Monday
    swap_mode: str = "full"           # full | markup
    swap_long: float = 0.0            # % of notional per year, negative = debit (full mode)
    swap_short: float = 0.0
    swap_markup: float = 0.0          # % per year charged on both sides (markup mode)
    stop_slippage_spreads: float = 0.5
    close_execution_slippage_spreads: float = 0.5


@dataclass(frozen=True)
class SpreadWidening:
    enabled: bool = True
    range_atr_threshold: float = 2.0
    factor: float = 2.0


class CostModel:
    def __init__(self, instruments: dict[str, InstrumentCost], widening: SpreadWidening,
                 multiplier: float = 1.0, profile: str = "custom"):
        self.instruments = instruments
        self.widening = widening
        self.multiplier = multiplier
        self.profile = profile

    @classmethod
    def from_profile(cls, profile: str | None = None, multiplier: float = 1.0) -> CostModel:
        """Load the active (or given) cost profile."""
        profile = cost_profile(profile)
        path = f"costs/{profile}.yaml"
        warn_unverified(path)
        cfg = load_yaml(path)
        defaults: dict[str, Any] = cfg.get("defaults", {})
        widening = SpreadWidening(**defaults.get("spread_widening", {}))
        shared = {k: float(defaults[k]) for k in ("stop_slippage_spreads", "close_execution_slippage_spreads") if k in defaults}
        instruments = {sym: InstrumentCost(symbol=sym, **{**shared, **spec}) for sym, spec in cfg["instruments"].items()}
        return cls(instruments, widening, multiplier, profile)

    def scaled(self, multiplier: float) -> CostModel:
        """Same model with every cost multiplied (sensitivity tests at x1.5 and x2)."""
        return CostModel(self.instruments, self.widening, self.multiplier * multiplier, self.profile)

    def with_instrument(self, symbol: str, **changes: Any) -> CostModel:
        inst = dict(self.instruments)
        inst[symbol] = replace(inst[symbol], **changes)
        return CostModel(inst, self.widening, self.multiplier, self.profile)

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

    def close_execution_slippage(self, symbol: str, units: float, fx: float = 1.0, observed: float | None = None,
                                 range_atr_ratio: float | None = None) -> float:
        """Extra slippage of the "close t + slippage" execution mode."""
        spec = self.instruments[symbol]
        return abs(units) * spec.close_execution_slippage_spreads * self.effective_spread(symbol, observed, range_atr_ratio) * fx

    def commission(self, symbol: str, units: float) -> float:
        spec = self.instruments[symbol]
        return abs(units) / spec.contract_size * spec.commission_per_lot * self.multiplier

    def trade_cost_fraction(self, symbol: str, price: float, fx: float = 1.0) -> float:
        """Cost of trading one unit of notional, as a fraction of notional (half spread + commission)."""
        spec = self.instruments[symbol]
        per_unit = 0.5 * self.effective_spread(symbol) * fx + spec.commission_per_lot / spec.contract_size * self.multiplier
        return per_unit / (price * fx)

    # --- overnight financing ----------------------------------------------------------------

    def swap_nights(self, symbol: str, weekday: int) -> int:
        """Nights charged for a position held over the close of a session on ``weekday``."""
        return 3 if weekday == self.instruments[symbol].triple_swap_day else 1

    def swap_rate(self, symbol: str, units: float) -> float:
        """Annual swap rate (%) applied to a position; never positive (no carry credit)."""
        spec = self.instruments[symbol]
        if spec.swap_mode == "markup":
            rate = spec.swap_markup
        else:
            rate = spec.swap_long if units > 0 else spec.swap_short
        return min(rate, 0.0)

    def swap(self, symbol: str, units: float, price: float, weekday: int, fx: float = 1.0) -> float:
        """Signed swap (negative = debit) for holding ``units`` over the session close."""
        if units == 0:
            return 0.0
        notional = abs(units) * price * fx
        rate = self.swap_rate(symbol, units)
        return notional * rate / 100.0 / DAYS_PER_YEAR * self.swap_nights(symbol, weekday) * self.multiplier


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
