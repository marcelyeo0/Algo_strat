"""Conversion of quote currencies to the USD account currency, using the universe FX bars."""

from __future__ import annotations

import pandas as pd

# currency -> (universe symbol, True if the symbol is quoted as CCY/USD, False for USD/CCY)
FX_SOURCES = {
    "EUR": ("EURUSD", True),
    "GBP": ("GBPUSD", True),
    "AUD": ("AUDUSD", True),
    "NZD": ("NZDUSD", True),
    "JPY": ("USDJPY", False),
    "CAD": ("USDCAD", False),
    "CHF": ("USDCHF", False),
    "HKD": ("USDHKD", False),     # auxiliary series, not traded
}


def quote_to_usd(ccy: str, panel: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.Series:
    """Daily rate converting one unit of ``ccy`` into USD, forward-filled on ``index``."""
    if ccy == "USD":
        return pd.Series(1.0, index=index)
    symbol, direct = FX_SOURCES[ccy]
    close = panel[symbol]["raw_close"].reindex(index.union(panel[symbol].index)).ffill().reindex(index)
    rate = close if direct else 1.0 / close
    return rate.bfill()
