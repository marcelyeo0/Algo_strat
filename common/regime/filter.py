"""Ensemble regime filter (HMM + RandomForest + LightGBM voting) producing an exposure multiplier.

Inspired by the ensemble-HMM voting framework of "A forest of opinions" (DSFE, 2025):

1. a Gaussian HMM is fitted on the train window (number of states chosen by BIC) and run as a
   causal forward filter;
2. a RandomForest (bagging) and a LightGBM (boosting) classifier learn to predict the HMM state
   of the *next* session from features known at t;
3. the one-step-ahead HMM prediction and both classifiers vote (weighted average of the state
   probabilities); the vote is smoothed with a causal EMA and mapped to a score in [0, 1].

The multiplier only scales positions; it never decides a direction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from common.regime.features import classifier_features, hmm_inputs
from common.regime.hmm import FittedHMM, fit_hmm, forward_filter, predict_next, state_scores
from common.tuning.walkforward import Fold
from common.utils.config import load_yaml


def load_regime_config(name: str = "regime.yaml") -> dict[str, Any]:
    return load_yaml(name)


def apply_mode(score: pd.Series, out_cfg: dict[str, Any], mode: str | None = None) -> pd.Series:
    """Map the smoothed score to the exposure multiplier (binary, continuous or hysteresis)."""
    mode = mode or out_cfg.get("mode", "continuous")
    if mode == "binary":
        return (score >= float(out_cfg.get("binary_threshold", 0.5))).astype(float)
    if mode == "continuous":
        floor = float(out_cfg.get("continuous_floor", 0.0))
        return score.clip(0.0, 1.0) * (1 - floor) + floor
    if mode == "hysteresis":
        lo = float(out_cfg["hysteresis"]["off_below"])
        hi = float(out_cfg["hysteresis"]["on_above"])
        state, values = 1.0, []
        for s in score.to_numpy():
            if state == 1.0 and s < lo:
                state = 0.0
            elif state == 0.0 and s > hi:
                state = 1.0
            values.append(state)
        return pd.Series(values, index=score.index)
    raise ValueError(f"Unknown regime mode '{mode}'")


@dataclass
class FittedRegime:
    hmm: FittedHMM
    scores: np.ndarray
    rf: RandomForestClassifier | None
    lgbm: LGBMClassifier | None
    feature_cols: list[str]


class RegimeFilter:
    def __init__(self, cfg: dict[str, Any] | None = None, seed: int = 42):
        self.cfg = cfg or load_regime_config()
        self.seed = seed

    # --- inputs ------------------------------------------------------------------------------

    def _inputs(self, ret: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        fc = self.cfg["features"]
        obs = hmm_inputs(ret, int(fc["hmm_vol_window"]))
        feats = classifier_features(ret, list(fc["vol_windows"]), list(fc["return_lags"]))
        return obs, feats

    def _design(self, feats: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
        return feats.join(filtered, how="inner").dropna()

    # --- fit ---------------------------------------------------------------------------------

    def fit(self, ret: pd.Series, train_end: pd.Timestamp) -> FittedRegime:
        """Fit everything on data up to ``train_end`` only."""
        ret = ret.loc[:train_end]
        obs, feats = self._inputs(ret)
        hc = self.cfg["hmm"]
        hmm = fit_hmm(obs, list(hc["n_states"]), hc.get("covariance_type", "full"),
                      int(hc.get("n_init", 3)), int(hc.get("n_iter", 200)), seed=self.seed)
        scores = state_scores(hmm, obs, ret, hc.get("state_score", "vol_rank"))
        filtered = forward_filter(hmm, obs)
        design = self._design(feats, filtered)
        # Label of row t = filtered HMM state at t+1 (inside the train window).
        label = filtered.to_numpy().argmax(axis=1)
        label = pd.Series(label, index=filtered.index).shift(-1).reindex(design.index)
        ok = label.notna()
        x, y = design[ok], label[ok].astype(int)
        ec = self.cfg["ensemble"]
        rf = lgbm = None
        if y.nunique() >= 2:
            rf = RandomForestClassifier(random_state=self.seed, n_jobs=-1, **ec["random_forest"]).fit(x, y)
            lgbm = LGBMClassifier(random_state=self.seed, verbose=-1, n_jobs=1, **ec["lightgbm"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lgbm.fit(x, y)
        return FittedRegime(hmm, scores, rf, lgbm, list(design.columns))

    # --- apply -------------------------------------------------------------------------------

    def _proba(self, clf, x: pd.DataFrame, k: int) -> np.ndarray:
        out = np.zeros((len(x), k))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = clf.predict_proba(x)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = p[:, j]
        return out

    def score(self, fitted: FittedRegime, ret: pd.Series) -> pd.DataFrame:
        """Causal score and vote for every date of ``ret`` (uses data up to each date only)."""
        obs, feats = self._inputs(ret)
        filtered = forward_filter(fitted.hmm, obs)
        k = fitted.hmm.n_states
        w = self.cfg["ensemble"]["weights"]
        votes = {"hmm": predict_next(fitted.hmm, filtered)}
        design = self._design(feats, filtered)[fitted.feature_cols]
        if fitted.rf is not None:
            votes["random_forest"] = pd.DataFrame(self._proba(fitted.rf, design, k), index=design.index, columns=filtered.columns)
            votes["lightgbm"] = pd.DataFrame(self._proba(fitted.lgbm, design, k), index=design.index, columns=filtered.columns)
        idx = votes["hmm"].index
        for name in votes:
            idx = idx.intersection(votes[name].index)
        total = sum(float(w.get(n, 1.0)) for n in votes)
        prob = sum(votes[n].loc[idx] * float(w.get(n, 1.0)) for n in votes) / total
        hl = float(self.cfg["smoothing"].get("prob_halflife", 0))
        if hl > 0:
            prob = prob.ewm(halflife=hl, adjust=False).mean()
        score = pd.Series(prob.to_numpy() @ fitted.scores, index=idx, name="score")
        return pd.DataFrame({"score": score, "state": prob.to_numpy().argmax(axis=1)}, index=idx)

    def multiplier(self, fitted: FittedRegime, ret: pd.Series, mode: str | None = None) -> pd.Series:
        s = self.score(fitted, ret)["score"]
        return apply_mode(s, self.cfg["output"], mode).rename("regime_multiplier")


def walk_forward_scores(ret: pd.Series, folds: list[Fold], cfg: dict[str, Any] | None = None,
                        seed: int = 42) -> pd.DataFrame:
    """Out-of-sample score and state over the test windows of ``folds``; refitted at every fold.

    For each fold, data after the fold's test end is never passed to the filter.
    """
    flt = RegimeFilter(cfg, seed=seed)
    pieces = []
    for fold in folds:
        fitted = flt.fit(ret, fold.train_end)
        sc = flt.score(fitted, ret.loc[:fold.test_end])
        pieces.append(sc.loc[fold.test_start:fold.test_end].assign(n_states=fitted.hmm.n_states))
    if not pieces:
        return pd.DataFrame(columns=["score", "state", "n_states"])
    return pd.concat(pieces)


def walk_forward_multiplier(ret: pd.Series, folds: list[Fold], cfg: dict[str, Any] | None = None,
                            mode: str | None = None, seed: int = 42,
                            scores: pd.DataFrame | None = None) -> pd.Series:
    """Out-of-sample exposure multiplier; pass precomputed ``scores`` to compare modes cheaply."""
    cfg = cfg or load_regime_config()
    scores = scores if scores is not None else walk_forward_scores(ret, folds, cfg, seed)
    if scores.empty:
        return pd.Series(dtype=float, name="regime_multiplier")
    return apply_mode(scores["score"], cfg["output"], mode).rename("regime_multiplier")
