"""Inputs of the regime filter: basket return, realised volatility and classifier features.

Every feature at date t only uses data available at the close of t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252


def basket_returns(panel: dict[str, pd.DataFrame], symbols: list[str], vol_window: int = 63) -> pd.Series:
    """Inverse-volatility weighted basket of daily returns (weights from data up to t-1)."""
    rets = pd.DataFrame({s: panel[s]["ret"] for s in symbols if s in panel}).sort_index()
    vol = rets.rolling(vol_window, min_periods=20).std().shift(1)
    w = (1.0 / vol).where(rets.notna())
    w = w.div(w.sum(axis=1), axis=0)
    basket = (w * rets).sum(axis=1, min_count=1)
    return basket.dropna().rename("basket")


def hmm_inputs(ret: pd.Series, vol_window: int = 21) -> pd.DataFrame:
    """HMM observations: daily log return and log realised volatility."""
    logret = np.log1p(ret)
    vol = logret.rolling(vol_window, min_periods=vol_window).std() * np.sqrt(ANNUAL)
    return pd.DataFrame({"ret": logret, "log_vol": np.log(vol)}).dropna()


def classifier_features(ret: pd.Series, vol_windows: list[int], return_lags: list[int]) -> pd.DataFrame:
    logret = np.log1p(ret)
    feats = {}
    for lag in return_lags:
        feats[f"ret_{lag}"] = logret.rolling(lag).sum()
    for w in vol_windows:
        feats[f"vol_{w}"] = logret.rolling(w, min_periods=w).std() * np.sqrt(ANNUAL)
    if len(vol_windows) >= 2:
        feats["vol_ratio"] = feats[f"vol_{vol_windows[0]}"] / feats[f"vol_{vol_windows[-1]}"]
    level = logret.cumsum()
    feats["drawdown_63"] = level - level.rolling(63, min_periods=20).max()
    return pd.DataFrame(feats)
