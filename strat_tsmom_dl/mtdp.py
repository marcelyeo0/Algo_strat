"""MTDP score (Song & Jeon, 2025): probability of each market-trend transition class.

The paper uses XGBoost; LightGBM (already a project dependency) is used here with the same
multiclass objective, pooled over assets.

Leakage control: the label of date t is the transition between t and t+1, so in-sample
probabilities on the training rows would encode the next-day return. The DMN therefore never
sees in-sample MTDP scores: on the train window it receives *out-of-fold* scores from a purged
K-fold with embargo, and on the test window the scores of a classifier fitted on the whole
train window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from common.tuning.cv import purged_kfold
from strat_tsmom_dl.features import TRANSITIONS, state_one_hot, transition_labels, trend_states

BASE_COLS_PREFIX = ("nret_", "macd_")
SCORE_COLS = [f"mtdp_{c}" for c in TRANSITIONS]


def _design(feats: pd.DataFrame, slow: int, fast: int) -> tuple[pd.DataFrame, pd.Series]:
    states = trend_states(feats["ret"], slow, fast)
    x = pd.concat([feats[[c for c in feats if c.startswith(BASE_COLS_PREFIX)]], state_one_hot(states)], axis=1)
    return x, transition_labels(states)


def _new_clf(seed: int) -> LGBMClassifier:
    return LGBMClassifier(objective="multiclass", n_estimators=200, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, random_state=seed,
                          verbose=-1, n_jobs=1)


def _fit(x: pd.DataFrame, y: pd.Series, seed: int) -> LGBMClassifier:
    clf = _new_clf(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x, y.map({c: i for i, c in enumerate(TRANSITIONS)}).astype(int))
    return clf


def _proba(clf: LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(x), len(TRANSITIONS)))
    if len(x) == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = clf.predict_proba(x)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


def mtdp_fold_scores(asset_feats: dict[str, pd.DataFrame], train_end: pd.Timestamp, slow: int, fast: int,
                     seed: int = 42, n_splits: int = 5, embargo: int = 5) -> dict[str, pd.DataFrame]:
    """MTDP scores per asset: out-of-fold up to ``train_end``, out-of-sample after it."""
    designs = {s: _design(f, slow, fast) for s, f in asset_feats.items()}
    rows = []
    for s, (x, y) in designs.items():
        d = x.join(y.rename("y")).loc[:train_end].dropna().iloc[:-1]   # last label uses a test day
        rows.append(d.assign(symbol=s))
    train = pd.concat(rows)
    xcols = [c for c in train if c not in ("y", "symbol")]
    oof = np.zeros((len(train), len(TRANSITIONS)))
    for tr, te in purged_kfold(pd.DatetimeIndex(train.index), n_splits, horizon=1, embargo=embargo):
        clf = _fit(train.iloc[tr][xcols], train.iloc[tr]["y"], seed)
        oof[te] = _proba(clf, train.iloc[te][xcols])
    oof_df = pd.DataFrame(oof, index=train.index, columns=SCORE_COLS).assign(symbol=train["symbol"].to_numpy())
    final = _fit(train[xcols], train["y"], seed)

    out = {}
    for s, (x, _) in designs.items():
        test_x = x.loc[train_end + pd.Timedelta(days=1):].dropna()
        test = pd.DataFrame(_proba(final, test_x[xcols]), index=test_x.index, columns=SCORE_COLS)
        in_sample = oof_df[oof_df["symbol"] == s].drop(columns="symbol")
        out[s] = pd.concat([in_sample, test]).sort_index()
    return out
