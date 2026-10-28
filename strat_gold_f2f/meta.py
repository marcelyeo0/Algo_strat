"""Meta-labeling of the primary F2F entries (López de Prado, Advances in Financial ML, ch. 3).

Each primary long entry is labelled with the triple-barrier method: the upper barrier is
entry + ``profit_take_atr`` ATR, the lower barrier the hard stop (entry - ``hard_stop_atr``
ATR) and the vertical barrier the timeout. A barrier touched intraday counts on that day; if
both are touched on the same day the lower one is assumed first (conservative). The label is 1
if the upper barrier is hit first, or if the vertical barrier is reached with a positive return.

A LightGBM classifier learns P(label = 1) from features known at the entry close. Training
events must *end* before the train end (purging). The probability only scales or filters the
primary trade; it never changes its direction.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from strat_gold_f2f.signal import ANNUAL, atr, ewma_vol

FEATURES = ("p_bull", "z", "p_trend", "momentum", "vol_ann", "atr_pct", "ret_5", "ret_21", "ret_63")


def entry_features(bars: pd.DataFrame, sig: pd.DataFrame, atr_window: int, ewma_span: int) -> pd.DataFrame:
    logp = np.log(bars["close"])
    a = atr(bars, atr_window)
    feats = sig[["p_bull", "z", "p_trend", "momentum"]].copy()
    feats["vol_ann"] = ewma_vol(bars["ret"].fillna(0.0), ewma_span) * np.sqrt(ANNUAL)
    feats["atr_pct"] = a / bars["close"]
    for h in (5, 21, 63):
        feats[f"ret_{h}"] = logp - logp.shift(h)
    return feats


def triple_barrier(bars: pd.DataFrame, entries: pd.DatetimeIndex, up_atr: float, down_atr: float,
                   timeout: int, atr_window: int) -> pd.DataFrame:
    a = atr(bars, atr_window)
    idx = bars.index
    rows = []
    for d in entries:
        i = idx.get_loc(d)
        entry, width = bars["close"].iloc[i], a.iloc[i]
        if np.isnan(width):
            continue
        up, down = entry + up_atr * width, entry - down_atr * width
        end = min(i + timeout, len(idx) - 1)
        label, t1 = None, idx[end]
        for j in range(i + 1, end + 1):
            if bars["low"].iloc[j] <= down:
                label, t1 = 0, idx[j]
                break
            if bars["high"].iloc[j] >= up:
                label, t1 = 1, idx[j]
                break
        if label is None:
            if end - i < timeout:       # vertical barrier not reached yet (end of data)
                continue
            label = int(bars["close"].iloc[end] > entry)
        rows.append({"entry": d, "t1": t1, "label": label})
    return pd.DataFrame(rows).set_index("entry") if rows else pd.DataFrame(columns=["t1", "label"])


class MetaModel:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.clf: LGBMClassifier | None = None
        self.base_rate = 0.5

    def fit(self, feats: pd.DataFrame, events: pd.DataFrame, train_end: pd.Timestamp) -> MetaModel:
        ev = events[events["t1"] <= train_end]                 # purge events ending after train end
        x = feats.reindex(ev.index)[list(FEATURES)]
        ok = x.notna().all(axis=1)
        x, y = x[ok], ev.loc[ok, "label"].astype(int)
        self.base_rate = float(y.mean()) if len(y) else 0.5
        if len(y) >= 30 and y.nunique() == 2:
            self.clf = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, random_state=self.seed, verbose=-1, n_jobs=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.clf.fit(x, y)
        return self

    def prob(self, feats_row: pd.Series) -> float:
        if self.clf is None or feats_row[list(FEATURES)].isna().any():
            return self.base_rate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self.clf.predict_proba(feats_row[list(FEATURES)].to_frame().T.astype(float))[0, 1])
