"""Session calendar: aggregation of intraday bars into daily bars closing at the FTMO server close."""

from __future__ import annotations

import pandas as pd


def session_dates(ts_utc: pd.DatetimeIndex, close_tz: str = "America/New_York", close_hour: int = 17) -> pd.DatetimeIndex:
    """Map bar *open* timestamps (UTC) to their trading-session date.

    A session ends at ``close_hour`` in ``close_tz`` (17:00 New York = 00:00 FTMO server time).
    A bar opening at or after the close belongs to the next session, so the Sunday-evening
    open is labelled with Monday's date.
    """
    if ts_utc.tz is None:
        ts_utc = ts_utc.tz_localize("UTC")
    local = ts_utc.tz_convert(close_tz)
    shifted = local + pd.Timedelta(hours=24 - close_hour)
    return pd.DatetimeIndex(shifted.tz_localize(None).normalize(), name="date")


def aggregate_to_sessions(
    bars: pd.DataFrame, close_tz: str = "America/New_York", close_hour: int = 17
) -> pd.DataFrame:
    """Aggregate intraday OHLC(V) bars indexed by UTC open time into daily session bars."""
    if bars.empty:
        return bars
    keys = session_dates(pd.DatetimeIndex(bars.index), close_tz, close_hour)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for col in bars.columns:
        if col not in agg:
            agg[col] = "sum" if col.startswith("volume") else "last"
    daily = bars.groupby(keys).agg(agg)
    # Sessions that start on Saturday/Sunday local time only carry the Sunday open; they are
    # merged into Monday by construction, but drop any residual weekend label.
    return daily[daily.index.day_of_week < 5]
