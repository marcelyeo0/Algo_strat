import numpy as np
import pandas as pd
import pytest

from common.tuning.walkforward import Fold
from common.utils.config import load_yaml
from strat_gold_f2f.meta import MetaModel, entry_features, triple_barrier
from strat_gold_f2f.signal import regime_signal
from strat_gold_f2f.sizing import impact_adjusted_kelly, position_weight
from strat_gold_f2f.strategy import run_rules, strategy_returns


def bars_from_close(close: np.ndarray, start="2010-01-01", spread=0.002) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(close))
    c = pd.Series(close, index=idx)
    return pd.DataFrame({"open": c.shift(1).fillna(c), "high": c * (1 + spread), "low": c * (1 - spread),
                         "close": c, "raw_close": c, "ret": c.pct_change(), "is_roll": False})


def random_walk(n=1500, drift=0.0003, vol=0.01, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1000 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def test_kelly_matches_closed_form_without_impact():
    mu, sigma = 0.0005, 0.01
    assert impact_adjusted_kelly(mu, sigma, 1.0, 0.0, 0.0) == pytest.approx(mu / sigma ** 2)
    assert impact_adjusted_kelly(0.0001, sigma, 1.0, 0.0002, 0.02) == 0.0     # mu <= n k
    assert impact_adjusted_kelly(mu, sigma, 1.0, 0.0, 0.02) < mu / sigma ** 2  # impact shrinks f*


def test_kelly_root_solves_first_order_condition():
    mu, sigma, n, k, g = 0.0006, 0.012, 1.0, 0.00007, 0.02
    x = np.sqrt(impact_adjusted_kelly(mu, sigma, n, k, g))
    assert 2 * sigma ** 2 * x ** 2 + 3 * g * n ** 1.5 * x - 2 * (mu - n * k) == pytest.approx(0.0, abs=1e-12)


def test_position_weight_caps_and_confidence():
    cfg = load_yaml("strategies/gold_f2f.yaml")["sizing"]
    assert position_weight(0.5, 0.01, 5.0, cfg) == 0.0
    assert position_weight(1.0, 1e-6, 100.0, cfg) == pytest.approx(cfg["w_max"])


def test_signal_standardisation_uses_train_only():
    bars = bars_from_close(random_walk())
    end = bars.index[1000]
    a = regime_signal(bars, 0.97, 50, 0.6, end)
    corrupted = bars.copy()
    corrupted.loc[end + pd.Timedelta(days=1):, "close"] *= 3
    b = regime_signal(corrupted, 0.97, 50, 0.6, end)
    pd.testing.assert_frame_equal(a.loc[:end], b.loc[:end])
    assert a["p_bull"].dropna().between(0, 1).all()


def test_rules_hard_stop_and_timeout():
    cfg = load_yaml("strategies/gold_f2f.yaml")
    up = np.concatenate([np.linspace(1000, 1300, 300), np.full(5, 1300.0), np.linspace(1300, 1000, 60)])
    bars = bars_from_close(up, spread=0.0005)
    sig = regime_signal(bars, 0.97, 50, 0.6, bars.index[250])
    rules = run_rules(bars, sig, cfg, kelly=None)
    reasons = set(rules["exit_reason"]) - {""}
    assert reasons & {"timeout", "trailing_stop", "hard_stop", "derisk"}
    assert "timeout" in reasons                    # steady uptrend: trades end after 30 sessions
    assert (rules["weight"] >= 0).all()            # long-only
    assert rules.loc[rules["weight"] > 0, "stop"].notna().all()


def test_t0_uses_same_close_and_t1_does_not():
    idx = pd.bdate_range("2020-01-01", periods=5)
    ret = pd.Series([0.0, 0.10, -0.05, 0.02, 0.0], index=idx)
    w = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=idx)     # decided at each close
    t0 = strategy_returns(w, ret, 0, 0.0, 0.0)
    t1 = strategy_returns(w, ret, 1, 0.0, 0.0)
    assert t0.iloc[1] == pytest.approx(0.10)                 # earns the very return used to decide
    assert t1.iloc[1] == 0.0 and t1.iloc[2] == pytest.approx(-0.05)


def test_triple_barrier_labels():
    close = np.array([100.0] * 20 + [101, 102, 104, 106, 108] + [108.0] * 10)
    bars = bars_from_close(close, spread=0.001)
    entry = bars.index[[19]]
    ev = triple_barrier(bars, entry, up_atr=2.0, down_atr=2.0, timeout=10, atr_window=5)
    assert ev["label"].iloc[0] == 1
    down = bars_from_close(np.array([100.0] * 20 + [99, 97, 95, 93] + [93.0] * 10), spread=0.001)
    ev2 = triple_barrier(down, down.index[[19]], 2.0, 2.0, 10, 5)
    assert ev2["label"].iloc[0] == 0


def test_meta_model_purges_events_ending_after_train_end():
    bars = bars_from_close(random_walk(seed=3))
    sig = regime_signal(bars, 0.97, 50, 0.6, bars.index[1000])
    feats = entry_features(bars, sig, 14, 20)
    entries = bars.index[100:1400:15]
    events = triple_barrier(bars, entries, 2.0, 2.0, 30, 14)
    end = bars.index[1000]
    m = MetaModel().fit(feats, events, end)
    used = events[events["t1"] <= end]
    assert m.base_rate == pytest.approx(used["label"].mean())
    p = m.prob(feats.loc[bars.index[1200]])
    assert 0.0 <= p <= 1.0


def test_fold_object_bounds():
    f = Fold(pd.Timestamp("2005-01-01"), pd.Timestamp("2014-12-31"), pd.Timestamp("2015-01-01"), pd.Timestamp("2015-06-30"))
    assert f.train_end < f.test_start
