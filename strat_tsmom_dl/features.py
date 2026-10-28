"""Features of the TSMOM / Deep Momentum Network strategy.

* normalised multi-horizon returns (Lim, Zohren & Roberts, 2019);
* MACD indicators of Baz et al. (2015);
* market trend dynamics of Song & Jeon (2025): bull / correction / bear / rebound states from
  the signs of a slow (52-week) and a fast (2-20-week) momentum signal, and the transition
  classes between those states.

Every value at date t uses data up to the close of t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 252
WEEK = 5
EXECUTION_LAG = 1    # sessions between the signal close and the execution (next close)

STATES = ("bull", "correction", "bear", "rebound")
# Nine MTDP classes: ordinary point + the eight transitions where only one signal flips
# (direct bull<->bear and correction<->rebound flips, where both signals change, are excluded).
TRANSITIONS = (
    "OP",
    "bull_to_correction", "bull_to_rebound",
    "bear_to_correction", "bear_to_rebound",
    "correction_to_bull", "correction_to_bear",
    "rebound_to_bull", "rebound_to_bear",
)


def ex_ante_vol(ret: pd.Series, span: int = 60) -> pd.Series:
    """Daily EWMA volatility known at the close of t."""
    return ret.ewm(span=span, min_periods=span // 2).std()


def winsorise(ret: pd.Series, sigma: float = 5.0, halflife: int = 252) -> pd.Series:
    """Cap returns at ``sigma`` EWMA standard deviations (estimated up to t-1)."""
    sd = ret.ewm(halflife=halflife, min_periods=20).std().shift(1)
    return ret.clip(-sigma * sd, sigma * sd).where(sd.notna(), ret)


def normalised_returns(ret: pd.Series, horizons: list[int], vol: pd.Series) -> pd.DataFrame:
    logp = np.log1p(ret.fillna(0.0)).cumsum()
    out = {}
    for h in horizons:
        r_h = np.expm1(logp - logp.shift(h))
        out[f"nret_{h}"] = r_h / (vol * np.sqrt(h))
    return pd.DataFrame(out)


def _halflife(span: int) -> float:
    return np.log(0.5) / np.log(1 - 1 / span)


def macd_indicators(price: pd.Series, pairs: list[tuple[int, int]]) -> pd.DataFrame:
    """Baz et al. normalised MACD: q = (EWMA_S - EWMA_L) / std63(price), y = q / std252(q)."""
    out = {}
    for s, lng in pairs:
        ema_s = price.ewm(halflife=_halflife(s), min_periods=s).mean()
        ema_l = price.ewm(halflife=_halflife(lng), min_periods=lng).mean()
        q = (ema_s - ema_l) / price.rolling(63, min_periods=40).std()
        out[f"macd_{s}_{lng}"] = q / q.rolling(252, min_periods=126).std()
    return pd.DataFrame(out)


def baz_response(y: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Baz et al. response function phi(y) = y exp(-y^2 / 4) / 0.89."""
    return y * np.exp(-(y ** 2) / 4) / 0.89


def trend_states(ret: pd.Series, slow_weeks: int = 52, fast_weeks: int = 8) -> pd.Series:
    """Daily market state from the signs of the slow and fast momentum signals."""
    logp = np.log1p(ret.fillna(0.0)).cumsum()
    slow = logp - logp.shift(slow_weeks * WEEK)
    fast = logp - logp.shift(fast_weeks * WEEK)
    state = pd.Series(np.nan, index=ret.index, dtype=object)
    valid = slow.notna() & fast.notna()
    state[valid & (slow >= 0) & (fast >= 0)] = "bull"
    state[valid & (slow >= 0) & (fast < 0)] = "correction"
    state[valid & (slow < 0) & (fast < 0)] = "bear"
    state[valid & (slow < 0) & (fast >= 0)] = "rebound"
    return state


def transition_labels(states: pd.Series) -> pd.Series:
    """Class of the transition between t and t+1 (uses t+1: training labels only)."""
    nxt = states.shift(-1)
    label = pd.Series("OP", index=states.index, dtype=object)
    changed = states.notna() & nxt.notna() & (states != nxt)
    names = states[changed] + "_to_" + nxt[changed]
    label[changed] = names.where(names.isin(TRANSITIONS), "OP")
    label[states.isna() | nxt.isna()] = np.nan
    return label


def state_one_hot(states: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({f"state_{s}": (states == s).astype(float) for s in STATES}, index=states.index).where(
        states.notna()
    )


def asset_features(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Base feature frame of one asset (normalised returns, MACD, trend state one-hot)."""
    vc, fc = cfg["volatility"], cfg["features"]
    ret = winsorise(bars["ret"], float(fc.get("winsorise_sigma", 5.0)))
    vol = ex_ante_vol(ret, int(vc["ewm_span"]))
    price = bars["close"]
    feats = [
        normalised_returns(ret, list(fc["return_horizons"]), vol),
        macd_indicators(price, [tuple(p) for p in fc["macd_pairs"]]),
    ]
    fast = int(cfg["features"]["mtdp"].get("fast_weeks_default", 8))
    states = trend_states(ret, int(fc["mtdp"]["slow_weeks"]), fast)
    feats.append(state_one_hot(states))
    out = pd.concat(feats, axis=1)
    out["vol"] = vol
    out["ret"] = ret
    # Training target only, never an input: a position decided at the close of t is executed at
    # the close of t+1 at the earliest, so it earns the return of t+2 (no signal-close execution).
    out["next_ret"] = ret.shift(-EXECUTION_LAG - 1)
    out["trend_state"] = states
    return out
