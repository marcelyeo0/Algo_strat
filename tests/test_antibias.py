"""Anti-bias checks on the execution engine: no fill at or before the signal timestamp."""

import warnings

import numpy as np
import pandas as pd
import pytest

from common.backtest.engine import EXECUTIONS, run_engine
from common.costs.model import CostModel


def _panel(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    out = {}
    for i, s in enumerate(["EURUSD", "US500", "XAUUSD"]):
        c = (1.1, 4000.0, 1800.0)[i] * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        o = c * np.exp(rng.normal(0, 0.002, n))
        df = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) * 1.003,
                           "low": np.minimum(o, c) * 0.997, "spread": np.nan}, index=idx)
        df["raw_close"] = df["close"]
        df["ret"] = df["close"].pct_change()
        out[s] = df
    return out


def _costs():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CostModel.from_profile("estimates")


def _weights(panel, seed=1):
    rng = np.random.default_rng(seed)
    idx = panel["EURUSD"].index
    return pd.DataFrame(rng.normal(0, 0.5, (len(idx), 3)), index=idx, columns=list(panel))


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_every_fill_is_strictly_after_its_signal(execution):
    panel = _panel()
    res = run_engine(_weights(panel), panel, _costs(), 100_000, execution)
    f = res.fills
    assert len(f) > 0
    exec_key = list(zip(f["exec_date"], f["exec_phase"], strict=True))
    sig_key = list(zip(f["signal_date"], f["signal_phase"], strict=True))
    assert all(e > s for e, s in zip(exec_key, sig_key, strict=True))
    assert (f["exec_date"] > f["signal_date"]).all()


def test_same_close_execution_is_rejected():
    panel = _panel()
    with pytest.raises(ValueError):
        run_engine(_weights(panel), panel, _costs(), 100_000, "same_close")


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_pnl_up_to_t_does_not_depend_on_signal_at_t(execution):
    panel = _panel()
    w = _weights(panel)
    base = run_engine(w, panel, _costs(), 100_000, execution).sessions
    t = w.index[60]
    w2 = w.copy()
    w2.loc[t:] = -5 * w2.loc[t:]
    alt = run_engine(w2, panel, _costs(), 100_000, execution).sessions
    pd.testing.assert_frame_equal(base.loc[:t], alt.loc[:t])
    assert not base.loc[t:].equals(alt.loc[t:])


@pytest.mark.parametrize("execution", EXECUTIONS)
def test_future_prices_do_not_change_past_sessions(execution):
    panel = _panel()
    w = _weights(panel)
    base = run_engine(w, panel, _costs(), 100_000, execution).sessions
    t = w.index[80]
    shocked = {s: df.copy() for s, df in panel.items()}
    for df in shocked.values():
        df.loc[df.index > t, ["open", "high", "low", "close"]] *= 1.5
    alt = run_engine(w, shocked, _costs(), 100_000, execution).sessions
    pd.testing.assert_frame_equal(base.loc[:t], alt.loc[:t])


def test_stale_opens_are_executed_at_the_next_close():
    """A stale open equals the signal close: such instruments must not be filled at the open."""
    panel = _panel()
    stale = panel["US500"].copy()
    stale["open"] = stale["close"].shift(1).fillna(stale["close"])
    panel["US500"] = stale
    res = run_engine(_weights(panel), panel, _costs(), 100_000, "next_open")
    f = res.fills
    assert (f.loc[f["symbol"] == "US500", "exec_phase"] == 1).all()
    assert (f.loc[f["symbol"] == "EURUSD", "exec_phase"] == 0).all()
