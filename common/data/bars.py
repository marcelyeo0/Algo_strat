"""Standard daily bar format shared by every source.

Columns of a bars frame (index: session date, tz-naive):
    open, high, low, close   back-adjusted prices (continuous across futures rolls)
    raw_close                close as published by the source
    ret                      close-to-close simple return; NaN on neutralised roll sessions
    is_roll                  True on sessions whose return spans a futures contract change
    spread                   typical bid/ask spread in price units (NaN when unknown)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]


def finalize_bars(raw: pd.DataFrame, roll_mask: pd.Series | None = None) -> pd.DataFrame:
    """Build the standard frame from raw OHLC bars of a single source.

    On roll sessions the close-to-close return compares two different contracts, so it is set
    to NaN. Prices are then rebuilt by back-adjustment from the cleaned returns: the most recent
    contract keeps its published level and earlier prices are rescaled.
    """
    df = raw.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    out = pd.DataFrame(index=df.index)
    out.index.name = "date"
    is_roll = roll_mask.reindex(df.index, fill_value=False) if roll_mask is not None else pd.Series(False, index=df.index)
    ret = df["close"].pct_change()
    ret = ret.mask(is_roll)
    growth = (1.0 + ret.fillna(0.0)).cumprod()
    adj_close = growth * (df["close"].iloc[-1] / growth.iloc[-1])
    factor = adj_close / df["close"]
    for col in PRICE_COLS:
        out[col] = df[col] * factor if col in df else np.nan
    # Some sources (notably Yahoo FX and futures) publish highs/lows that do not bracket the
    # open/close. Repair by widening the range; the quality report counts these bars on the raw data.
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    out["raw_close"] = df["close"]
    out["ret"] = ret
    out["is_roll"] = is_roll.astype(bool)
    out["spread"] = df["spread"] if "spread" in df else np.nan
    return out
