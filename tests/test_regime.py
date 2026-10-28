import numpy as np
import pandas as pd
import pytest

from common.regime.features import hmm_inputs
from common.regime.filter import RegimeFilter, apply_mode, walk_forward_multiplier
from common.regime.hmm import fit_hmm, forward_filter
from common.tuning.walkforward import make_folds

FAST_CFG = {
    "features": {"vol_windows": [5, 21, 63], "hmm_vol_window": 21, "return_lags": [1, 5, 21]},
    "hmm": {"n_states": [2, 3], "covariance_type": "full", "n_init": 1, "n_iter": 50, "state_score": "vol_rank"},
    "smoothing": {"prob_halflife": 3},
    "ensemble": {"weights": {"hmm": 1.0, "random_forest": 1.0, "lightgbm": 1.0},
                 "random_forest": {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 20},
                 "lightgbm": {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20}},
    "output": {"mode": "continuous", "binary_threshold": 0.5, "continuous_floor": 0.0,
               "hysteresis": {"off_below": 0.35, "on_above": 0.6}},
}


def regime_returns(n=2500, seed=0) -> pd.Series:
    """Alternating calm / turbulent regimes."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.025)
    return pd.Series(rng.normal(0.0003, vol), index=pd.bdate_range("2005-01-03", periods=n))


def test_forward_filter_is_causal():
    ret = regime_returns()
    obs = hmm_inputs(ret)
    fitted = fit_hmm(obs.iloc[:1500], [2], n_init=1, n_iter=50)
    full = forward_filter(fitted, obs)
    cut = obs.index[1800]
    truncated = forward_filter(fitted, obs.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], truncated)


def test_hmm_separates_calm_and_turbulent_states():
    ret = regime_returns()
    filt = RegimeFilter(FAST_CFG)
    fitted = filt.fit(ret, ret.index[1999])
    score = filt.score(fitted, ret)["score"]
    calm = (np.arange(len(ret)) // 250) % 2 == 0
    calm = pd.Series(calm, index=ret.index).reindex(score.index)
    assert score[calm].mean() > score[~calm].mean() + 0.3


def test_multiplier_in_unit_interval_and_modes():
    s = pd.Series([0.9, 0.5, 0.3, 0.4, 0.7, 0.2])
    out = FAST_CFG["output"]
    assert apply_mode(s, out, "binary").tolist() == [1, 1, 0, 0, 1, 0]
    assert apply_mode(s, out, "continuous").between(0, 1).all()
    assert apply_mode(s, out, "hysteresis").tolist() == [1, 1, 0, 0, 1, 0]
    # Hysteresis keeps the previous state between the two thresholds.
    assert apply_mode(pd.Series([0.9, 0.3, 0.5, 0.55, 0.65]), out, "hysteresis").tolist() == [1, 0, 0, 0, 1]


def test_walk_forward_multiplier_does_not_use_future_data():
    ret = regime_returns()
    folds = make_folds(ret.index, train_years=4, test_months=12)
    full = walk_forward_multiplier(ret, folds[:2], FAST_CFG)
    # Corrupt everything after the end of the second test window: results must not change.
    corrupted = ret.copy()
    corrupted.loc[folds[1].test_end + pd.Timedelta(days=1):] *= 50
    again = walk_forward_multiplier(corrupted, folds[:2], FAST_CFG)
    pd.testing.assert_series_equal(full, again)
    assert full.between(0, 1).all()
    assert full.index.min() >= folds[0].test_start


def test_fit_ignores_data_after_train_end():
    ret = regime_returns()
    end = ret.index[1500]
    a = RegimeFilter(FAST_CFG).fit(ret, end)
    corrupted = ret.copy()
    corrupted.loc[end + pd.Timedelta(days=1):] = 0.5
    b = RegimeFilter(FAST_CFG).fit(corrupted, end)
    assert np.allclose(a.hmm.model.means_, b.hmm.model.means_)
    assert a.hmm.n_states == b.hmm.n_states


def test_make_folds_are_contiguous_and_bounded():
    idx = pd.bdate_range("2000-01-03", "2021-12-31")
    folds = make_folds(idx, train_years=10, test_months=6, expanding=False, last_date=pd.Timestamp("2021-12-31"))
    assert folds[0].test_start == pd.Timestamp("2010-01-03")
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start == a.test_end + pd.Timedelta(days=1)
        assert a.train_end < a.test_start
    assert folds[-1].test_end <= pd.Timestamp("2021-12-31")
    assert (folds[3].test_start - folds[3].train_start).days == pytest.approx(3653, abs=2)
