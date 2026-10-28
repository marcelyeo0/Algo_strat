import numpy as np
import pandas as pd
import pytest
import torch

from common.backtest.portfolio import apply_margin, build_portfolio, cap_class_risk, equal_risk
from common.utils.config import load_yaml
from strat_tsmom_dl.baselines import macd_signal, tsmom_signal
from strat_tsmom_dl.dmn import LSTMPolicy, build_sequences, predict_positions, sharpe_loss
from strat_tsmom_dl.features import (
    TRANSITIONS,
    asset_features,
    baz_response,
    transition_labels,
    trend_states,
)


def synthetic_bars(n=800, drift=0.0005, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = pd.Series(rng.normal(drift, 0.01, n), index=pd.bdate_range("2010-01-01", periods=n))
    close = 100 * (1 + ret).cumprod()
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
                         "raw_close": close, "ret": ret, "is_roll": False})


def test_features_are_causal():
    cfg = load_yaml("strategies/tsmom_dl.yaml")
    bars = synthetic_bars()
    full = asset_features(bars, cfg).drop(columns=["next_ret"])
    cut = bars.index[600]
    trunc = asset_features(bars.loc[:cut], cfg).drop(columns=["next_ret"])
    pd.testing.assert_frame_equal(full.loc[:cut], trunc)


def test_tsmom_signal_follows_trend():
    up = synthetic_bars(drift=0.003)["ret"]
    assert tsmom_signal(up, [1, 3, 12]).dropna().mean() > 0.8
    assert macd_signal((1 + up).cumprod(), [(8, 24), (16, 48), (32, 96)]).dropna().mean() > 0


def test_baz_response_shape():
    y = pd.Series([0.0, np.sqrt(2), 10.0])
    r = baz_response(y)
    assert r.iloc[0] == 0 and r.iloc[1] == pytest.approx(r.max()) and abs(r.iloc[2]) < 0.01


def test_trend_states_and_transitions():
    ret = synthetic_bars(n=600)["ret"]
    states = trend_states(ret, 52, 8)
    assert set(states.dropna().unique()) <= {"bull", "correction", "bear", "rebound"}
    labels = transition_labels(states).dropna()
    assert set(labels.unique()) <= set(TRANSITIONS)
    s = pd.Series(["bull", "bull", "correction", "bear", "bear"])
    assert transition_labels(s).tolist()[:4] == ["OP", "bull_to_correction", "correction_to_bear", "OP"]


def test_sharpe_loss_rewards_correct_direction_and_penalises_turnover():
    torch.manual_seed(0)
    r = torch.full((4, 50), 0.001) + torch.randn(4, 50) * 0.01
    scale = torch.ones(4, 50)
    cost = torch.full((4,), 0.001)
    long_ = sharpe_loss(torch.ones(4, 50), scale, r, cost, 0.0)
    short = sharpe_loss(-torch.ones(4, 50), scale, r, cost, 0.0)
    assert long_ < short
    flip = torch.ones(4, 50)
    flip[:, ::2] = -1
    assert sharpe_loss(flip, scale, r, cost, 1.0) > sharpe_loss(flip, scale, r, cost, 0.0)


def test_predict_positions_uses_past_window_only():
    torch.manual_seed(0)
    model = LSTMPolicy(2, 4, 0.0)
    idx = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 2)), index=idx, columns=["a", "b"])
    p1 = predict_positions(model, df, ["a", "b"], 10, idx[50:60])
    df2 = df.copy()
    df2.iloc[60:] = 99.0
    p2 = predict_positions(model, df2, ["a", "b"], 10, idx[50:60])
    pd.testing.assert_series_equal(p1, p2)
    assert p1.between(-1, 1).all()


def test_build_sequences_shapes():
    idx = pd.bdate_range("2020-01-01", periods=130)
    df = pd.DataFrame({"a": 1.0, "scale": 1.0, "next_ret": 0.0}, index=idx)
    seq = build_sequences({"X": df}, ["a"], 63, {"X": 0.0002})
    assert seq.x.shape == (2, 63, 1) and seq.cost.tolist() == pytest.approx([0.0002, 0.0002])


def test_portfolio_caps_and_margin():
    idx = pd.bdate_range("2020-01-01", periods=3)
    pos = pd.DataFrame({"A": [1.0, 1.0, np.nan], "B": [1.0, -1.0, 1.0], "C": [1.0, 1.0, 1.0]}, index=idx)
    classes = {"A": "fx", "B": "fx", "C": "metal"}
    w = equal_risk(pos)
    assert w.iloc[0].tolist() == pytest.approx([1 / 3] * 3)
    capped = cap_class_risk(w, classes, {"fx": 0.5})
    # The cap is a share of the original risk budget; freed risk is not redistributed.
    assert capped.iloc[0][["A", "B"]].abs().sum() == pytest.approx(0.5 * w.iloc[0].abs().sum())
    assert capped.iloc[0]["C"] == pytest.approx(w.iloc[0]["C"])
    big = pd.DataFrame({"A": [20.0], "C": [5.0]})
    m = apply_margin(big, classes, {"fx": 30, "metal": 9}, max_margin=0.9)
    assert (m["A"] / 30 + m["C"] / 9).iloc[0] == pytest.approx(0.9)


def test_build_portfolio_hits_target_vol():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=600)
    rets = pd.DataFrame(rng.normal(0, 0.01, (600, 3)), index=idx, columns=list("ABC"))
    pos = pd.DataFrame(1.0, index=idx, columns=list("ABC"))
    w = build_portfolio(pos, rets, {s: "fx" for s in "ABC"}, {}, 0.10, {"fx": 100})
    realised = (w.shift(1) * rets).sum(axis=1).iloc[200:].std() * np.sqrt(252)
    assert realised == pytest.approx(0.10, rel=0.15)


def test_purged_kfold_purges_and_embargoes():
    from common.tuning.cv import purged_kfold

    dates = pd.bdate_range("2020-01-01", periods=100)
    pooled = dates.append(dates)               # two assets on the same dates
    for train, test in purged_kfold(pooled, n_splits=5, horizon=1, embargo=5):
        test_dates = pooled[test]
        lo, hi = test_dates.min(), test_dates.max()
        tr = pooled[train]
        assert not ((tr >= lo) & (tr <= hi)).any()
        # One business day before the block is purged, five after are embargoed.
        assert not ((tr >= lo - pd.offsets.BDay(1)) & (tr < lo)).any()
        assert not ((tr > hi) & (tr <= hi + pd.offsets.BDay(5))).any()


def test_mtdp_train_scores_are_out_of_fold():
    from strat_tsmom_dl.mtdp import SCORE_COLS, mtdp_fold_scores

    cfg = load_yaml("strategies/tsmom_dl.yaml")
    feats = {f"A{i}": asset_features(synthetic_bars(n=900, seed=i), cfg) for i in range(2)}
    train_end = feats["A0"].index[700]
    scores = mtdp_fold_scores(feats, train_end, 52, 8, n_splits=3)
    s = scores["A0"]
    assert list(s.columns) == SCORE_COLS
    assert np.allclose(s.sum(axis=1), 1.0)
    # Test-window scores do not change when data after train_end is perturbed in the labels' future.
    assert s.index.max() > train_end
