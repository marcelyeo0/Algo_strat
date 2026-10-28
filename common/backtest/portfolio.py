"""Portfolio aggregation of per-asset volatility-scaled positions.

Steps, all using information available at the close of t:

1. equal risk budget across the assets with a valid position (each one is already scaled to
   the same target volatility, so this is inverse-volatility weighting);
2. caps on the share of the risk budget allocated to each asset class (freed risk is not
   redistributed to the other classes);
3. portfolio volatility targeting with a trailing covariance matrix;
4. regime multiplier in [0, 1] (optional);
5. margin constraint from the prop account leverage per asset class.

Output: notional exposure per asset as a fraction of the account capital (signed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252


def equal_risk(pos: pd.DataFrame) -> pd.DataFrame:
    n = pos.notna().sum(axis=1).replace(0, np.nan)
    return pos.div(n, axis=0).fillna(0.0)


def cap_class_risk(w: pd.DataFrame, classes: dict[str, str], caps: dict[str, float]) -> pd.DataFrame:
    w = w.copy()
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    for cls, cap in caps.items():
        cols = [s for s in w if classes.get(s) == cls]
        if not cols:
            continue
        share = w[cols].abs().sum(axis=1) / gross
        factor = (cap / share).clip(upper=1.0).fillna(1.0)
        w[cols] = w[cols].mul(factor, axis=0)
    return w


def target_portfolio_vol(w: pd.DataFrame, returns: pd.DataFrame, target: float, window: int = 126,
                         max_scale: float = 4.0) -> pd.DataFrame:
    """Scale weights so that the ex-ante portfolio volatility equals ``target`` (annualised)."""
    rets = returns.reindex(columns=w.columns).fillna(0.0)
    out = w.copy()
    scales = pd.Series(np.nan, index=w.index)
    dates = w.index
    loc = rets.index.get_indexer(dates)
    values = rets.to_numpy()
    for i, j in enumerate(loc):
        if j < window // 2:
            scales.iloc[i] = 0.0
            continue
        cov = np.cov(values[max(0, j - window + 1):j + 1].T) * ANNUAL
        wi = w.iloc[i].to_numpy()
        vol = float(np.sqrt(max(wi @ cov @ wi, 0.0)))
        scales.iloc[i] = min(target / vol, max_scale) if vol > 0 else 0.0
    return out.mul(scales, axis=0)


def apply_margin(w: pd.DataFrame, classes: dict[str, str], leverage: dict[str, float], max_margin: float = 0.9) -> pd.DataFrame:
    """Scale down when the margin used (sum |w| / leverage) exceeds ``max_margin`` of equity."""
    lev = pd.Series({s: leverage.get(classes.get(s, ""), 1.0) for s in w})
    margin = (w.abs() / lev).sum(axis=1)
    factor = (max_margin / margin).clip(upper=1.0).fillna(1.0)
    return w.mul(factor, axis=0)


def build_portfolio(pos: pd.DataFrame, returns: pd.DataFrame, classes: dict[str, str], caps: dict[str, float],
                    target_vol: float, leverage: dict[str, float], cov_window: int = 126,
                    regime: pd.Series | None = None) -> pd.DataFrame:
    w = equal_risk(pos)
    w = cap_class_risk(w, classes, caps)
    w = target_portfolio_vol(w, returns, target_vol, cov_window)
    if regime is not None:
        w = w.mul(regime.reindex(w.index).ffill().fillna(1.0).clip(0, 1), axis=0)
    return apply_margin(w, classes, leverage)
