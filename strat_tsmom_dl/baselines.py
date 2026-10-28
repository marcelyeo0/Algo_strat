"""Classical baselines (the benchmark the deep model has to beat).

Signals are in [-1, 1] and are turned into positions by volatility scaling:
``position = signal * target_vol / ex_ante_vol``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strat_tsmom_dl.features import ANNUAL, baz_response, ex_ante_vol, macd_indicators, winsorise

MONTH = 21


def tsmom_signal(ret: pd.Series, months: list[int]) -> pd.Series:
    """Average sign of the past 1, 3 and 12-month returns (Moskowitz, Ooi & Pedersen)."""
    logp = np.log1p(ret.fillna(0.0)).cumsum()
    signs = [np.sign(logp - logp.shift(m * MONTH)) for m in months]
    return pd.concat(signs, axis=1).mean(axis=1, skipna=False)


def macd_signal(price: pd.Series, pairs: list[tuple[int, int]]) -> pd.Series:
    """Baz et al. multi-scale MACD: average response over the three time-scale pairs."""
    y = macd_indicators(price, pairs)
    return baz_response(y).mean(axis=1, skipna=False).clip(-1, 1)


def vol_scaled(signal: pd.Series, ret: pd.Series, target: float, span: int) -> pd.Series:
    vol = ex_ante_vol(ret, span) * np.sqrt(ANNUAL)
    return signal * target / vol


def baseline_positions(panel: dict[str, pd.DataFrame], cfg: dict, kind: str) -> pd.DataFrame:
    """Volatility-scaled positions of a baseline for every asset (index: session date)."""
    vc, fc = cfg["volatility"], cfg["features"]
    out = {}
    for sym in cfg["symbols"]:
        bars = panel[sym]
        ret = winsorise(bars["ret"], float(fc.get("winsorise_sigma", 5.0)))
        if kind == "tsmom":
            sig = tsmom_signal(ret, list(cfg["baselines"]["tsmom_months"]))
        elif kind == "macd":
            sig = macd_signal(bars["close"], [tuple(p) for p in fc["macd_pairs"]])
        elif kind == "long_only":
            sig = pd.Series(1.0, index=bars.index)
        else:
            raise ValueError(f"Unknown baseline '{kind}'")
        out[sym] = vol_scaled(sig, ret, float(vc["asset_target"]), int(vc["ewm_span"]))
    return pd.DataFrame(out).sort_index()
