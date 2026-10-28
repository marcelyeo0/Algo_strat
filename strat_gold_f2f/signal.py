"""Forecast-to-Fill signal: smoothed trend-momentum regime probability, ATR and volatility.

Formulas (arXiv 2511.08571):
    y_t       = log P_t
    y~_t      = lambda y~_{t-1} + (1 - lambda) y_t
    z_t       = (dy~_t - mu_train) / sigma_train          (train-window standardisation)
    p_trend_t = (clip(z_t, -3, 3) + 3) / 6
    m_t       = 1{P_t / P_{t-K} > 1}
    p_bull_t  = omega p_trend_t + (1 - omega) m_t,   p_bear_t = 1 - p_bull_t
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252


def smoothed_slope(close: pd.Series, lam: float) -> pd.Series:
    y = np.log(close)
    ema = y.ewm(alpha=1 - lam, adjust=False).mean()     # y~_0 = y_0
    return ema.diff()


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def ewma_vol(ret: pd.Series, span: int) -> pd.Series:
    """Daily EWMA volatility forecast for t+1 made at the close of t."""
    return np.sqrt((ret ** 2).ewm(span=span, min_periods=span).mean())


def regime_signal(bars: pd.DataFrame, lam: float, k: int, omega: float, train_end: pd.Timestamp) -> pd.DataFrame:
    close = bars["close"]
    slope = smoothed_slope(close, lam)
    train = slope.loc[:train_end].dropna()
    mu, sd = float(train.mean()), float(train.std())
    z = (slope - mu) / (sd if sd > 0 else 1.0)
    p_trend = (z.clip(-3, 3) + 3) / 6
    m = (close / close.shift(k) > 1).astype(float).where(close.shift(k).notna())
    p_bull = omega * p_trend + (1 - omega) * m
    return pd.DataFrame({"slope": slope, "z": z, "p_trend": p_trend, "momentum": m, "p_bull": p_bull})
