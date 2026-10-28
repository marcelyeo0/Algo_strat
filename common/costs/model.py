"""Trading cost model: spreads, commissions, overnight swaps and stop slippage.

Every number comes from the active cost profile (``configs/costs/<profile>.yaml``); nothing is
hard-coded here. Amounts are returned in account currency (USD). ``fx`` is the quote-currency
to USD rate of the instrument on the relevant day (1.0 for USD-quoted instruments).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from common.data.rates import load_policy_rates
from common.utils.config import load_universe, load_yaml, warn_unverified
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
    swap_mode: str = "full"           # rates | index_rates | markup | full
    swap_long: float = 0.0            # % of notional per year, negative = debit (full mode only)
    swap_short: float = 0.0
    swap_markup: float = 0.0          # broker margin, % per year; class default from the profile
    base_ccy: str | None = None       # FX pairs: base currency (rates mode)
    asset_class: str = ""
    stop_slippage_spreads: float = 0.5
    close_execution_slippage_spreads: float = 0.5


@dataclass(frozen=True)
class SpreadWidening:
    enabled: bool = True
    range_atr_threshold: float = 2.0
    factor: float = 2.0


class CostModel:
    def __init__(self, instruments: dict[str, InstrumentCost], widening: SpreadWidening,
                 multiplier: float = 1.0, profile: str = "custom", rates: pd.DataFrame | None = None,
                 credit_mode: str = "zero"):
        self.instruments = instruments
        self.widening = widening
        self.multiplier = multiplier
        self.profile = profile
        self.rates = rates              # daily policy rates (% per year), columns = currencies
        self.credit_mode = credit_mode  # zero: positive swaps are not credited | real: credited

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
        markups = {k: float(v) for k, v in defaults.get("swap_markup_by_class", {}).items()}
        classes = {s: v["asset_class"] for s, v in load_universe().items()}
        instruments = {}
        for sym, spec in cfg["instruments"].items():
            spec = {**shared, **spec}
            asset_class = classes.get(sym, "")
            spec.setdefault("swap_markup", markups.get(asset_class, 0.0))
            instruments[sym] = InstrumentCost(symbol=sym, asset_class=asset_class, **spec)
        needs_rates = any(i.swap_mode in ("rates", "index_rates") for i in instruments.values())
        rates = load_policy_rates() if needs_rates else None
        return cls(instruments, widening, multiplier, profile, rates, defaults.get("swap_credit", "zero"))

    def _clone(self, **changes: Any) -> CostModel:
        kw = dict(instruments=self.instruments, widening=self.widening, multiplier=self.multiplier,
                  profile=self.profile, rates=self.rates, credit_mode=self.credit_mode)
        kw.update(changes)
        return CostModel(**kw)

    def scaled(self, multiplier: float) -> CostModel:
        """Same model with every cost multiplied (sensitivity tests at x1.5 and x2)."""
        return self._clone(multiplier=self.multiplier * multiplier)

    def with_credit_mode(self, mode: str) -> CostModel:
        """``zero`` (default: positive swaps are not credited) or ``real`` (sensitivity: credited)."""
        if mode not in ("zero", "real"):
            raise ValueError("credit mode must be 'zero' or 'real'")
        return self._clone(credit_mode=mode)

    def with_instrument(self, symbol: str, **changes: Any) -> CostModel:
        inst = dict(self.instruments)
        inst[symbol] = replace(inst[symbol], **changes)
        return self._clone(instruments=inst)

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

    def policy_rate(self, ccy: str, date: pd.Timestamp | None) -> float:
        if self.rates is None or date is None:
            raise ValueError("this swap mode needs policy rates and a date")
        r = self.rates
        d = min(max(pd.Timestamp(date), r.index[0]), r.index[-1])
        value = r.at[d, ccy]
        return 0.0 if pd.isna(value) else float(value)

    def swap_rate(self, symbol: str, units: float, date: pd.Timestamp | None = None) -> float:
        """Annual swap rate (%) of a position held on ``date``; negative = debit.

        rates       (FX)      long: (r_base - r_quote) - markup, short: (r_quote - r_base) - markup
        index_rates (indices) long: -(r + markup), short: r - markup, r = rate of the index currency
        markup      (futures-based CFDs, carry already in the price): -markup on both sides
        full        (fixed values of the profile): swap_long / swap_short
        Positive rates are set to 0 unless the credit mode is ``real``.
        """
        spec = self.instruments[symbol]
        m = abs(spec.swap_markup)
        if spec.swap_mode == "markup":
            rate = -m
        elif spec.swap_mode == "rates":
            diff = self.policy_rate(spec.base_ccy, date) - self.policy_rate(spec.quote_ccy, date)
            rate = (diff if units > 0 else -diff) - m
        elif spec.swap_mode == "index_rates":
            r = self.policy_rate(spec.quote_ccy, date)
            rate = -(r + m) if units > 0 else r - m
        else:
            rate = spec.swap_long if units > 0 else spec.swap_short
        return rate if self.credit_mode == "real" else min(rate, 0.0)

    def swap(self, symbol: str, units: float, price: float, weekday: int, fx: float = 1.0,
             date: pd.Timestamp | None = None) -> float:
        """Signed swap (negative = debit) for holding ``units`` over the session close."""
        if units == 0:
            return 0.0
        notional = abs(units) * price * fx
        rate = self.swap_rate(symbol, units, date)
        amount = notional * rate / 100.0 / DAYS_PER_YEAR * self.swap_nights(symbol, weekday)
        # Debits scale with the cost multiplier; credits (real mode) are never inflated.
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
