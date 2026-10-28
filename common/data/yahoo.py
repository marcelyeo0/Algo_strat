"""yfinance fallback source (long history). Daily bars keep Yahoo's native session."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

log = logging.getLogger(__name__)


def download_daily(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Raw OHLCV daily bars from Yahoo Finance, with lower-case columns and a tz-naive index."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=False, multi_level_index=False, threads=False
        )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    return df
