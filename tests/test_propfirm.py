import warnings

import numpy as np
import pandas as pd
import pytest

from common.backtest.ledger import Bar, Ledger
from common.costs.model import CostModel, InstrumentCost, SpreadWidening, stop_fill
from common.propfirm.account import PropAccount, Status
from common.propfirm.challenge import simulate_challenges, summarise, summarise_horizons
from common.propfirm.rules import Phase, PropRules, load_prop_rules
from common.utils.config import UnverifiedConfigWarning

INITIAL = 100_000.0


def rules(**kw) -> PropRules:
    base = dict(
        name="test", initial_capital=INITIAL, daily_loss_pct=0.05, total_loss_pct=0.10,
        phases=(Phase("phase1", 0.10, min_trading_days=4), Phase("phase2", 0.05, min_trading_days=4)),
    )
    base.update(kw)
    return PropRules(**base)


def day(i: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01") + pd.offsets.BDay(i)


# --- daily loss -------------------------------------------------------------------------------

def test_daily_loss_exactly_at_floor_is_a_breach():
    acc = PropAccount(rules())
    assert acc.step(day(0), INITIAL, INITIAL, 95_000.0, True) is Status.FAILED_DAILY
    assert acc.violation.cause == "daily_loss"
    assert acc.violation.date == day(0)


def test_daily_loss_just_above_floor_survives():
    acc = PropAccount(rules())
    assert acc.step(day(0), INITIAL, 95_100.0, 95_000.01, True) is Status.ACTIVE


def test_daily_reference_uses_max_of_balance_and_equity():
    # Floating profit: equity above balance -> the reference is the equity.
    acc = PropAccount(rules())
    acc.step(day(0), INITIAL, 103_000.0, 100_000.0, True)
    assert acc.daily_floor() == pytest.approx(98_000.0)
    assert acc.step(day(1), INITIAL, 99_000.0, 98_000.0, False) is Status.FAILED_DAILY

    # Floating loss: balance above equity -> the reference is the balance.
    acc = PropAccount(rules())
    acc.step(day(0), 101_000.0, 99_000.0, 99_000.0, True)
    assert acc.daily_floor() == pytest.approx(96_000.0)


def test_floating_equity_counts_intraday():
    acc = PropAccount(rules())
    # Closed balance unchanged, but the worst intraday equity crosses the floor.
    assert acc.step(day(0), INITIAL, 99_000.0, 94_500.0, False) is Status.FAILED_DAILY


def test_daily_reset_reference_comes_from_previous_bar_close():
    """The CE(S)T midnight reset is modelled with the previous bar's close (approximation)."""
    acc = PropAccount(rules())
    acc.step(day(0), 98_000.0, 98_000.0, 97_000.0, True)        # -2% day, survives
    # New session: the reference resets to the previous close (98k), not to the initial capital.
    assert acc.daily_reference() == pytest.approx(98_000.0)
    assert acc.daily_floor() == pytest.approx(93_000.0)
    assert acc.step(day(1), 98_000.0, 98_000.0, 93_500.0, False) is Status.ACTIVE
    # Two consecutive -4.5% days are allowed by the daily rule (each day resets) ...
    acc2 = PropAccount(rules())
    assert acc2.step(day(0), 95_500.0, 95_500.0, 95_500.0, True) is Status.ACTIVE
    assert acc2.step(day(1), 91_200.0, 91_200.0, 91_200.0, True) is Status.ACTIVE
    # ... until the static total floor (90k) is hit.
    assert acc2.step(day(2), 90_000.0, 90_000.0, 90_000.0, True) is Status.FAILED_TOTAL


def test_reference_does_not_move_with_intraday_profit_of_the_same_day():
    acc = PropAccount(rules())
    acc.step(day(0), INITIAL, INITIAL, INITIAL, True)
    floor_before = acc.daily_floor()
    acc.step(day(1), 104_000.0, 104_000.0, 99_000.0, True)
    # The floor in force during day(1) was the one computed from day(0)'s close.
    assert acc.history[-1].daily_floor == floor_before


# --- total loss ---------------------------------------------------------------------------------

def test_static_total_loss():
    acc = PropAccount(rules(daily_loss_pct=0.5))
    assert acc.step(day(0), 90_000.0, 90_000.0, 90_000.0, True) is Status.FAILED_TOTAL


def test_trailing_eod_total_loss_locks_at_initial():
    r = rules(daily_loss_pct=0.03, total_loss_type="trailing_eod", total_loss_pct=0.10, trail_basis="balance")
    acc = PropAccount(r)
    acc.step(day(0), 105_000.0, 105_000.0, 104_000.0, True)
    assert acc.total_floor() == pytest.approx(95_000.0)
    acc.step(day(1), 108_000.0, 108_000.0, 107_000.0, True)
    acc.step(day(2), 111_000.0, 111_000.0, 110_000.0, True)
    assert acc.total_floor() == pytest.approx(INITIAL)  # locked at the initial capital


# --- targets ------------------------------------------------------------------------------------

def test_target_requires_min_trading_days():
    acc = PropAccount(rules(), phase=Phase("phase1", 0.10, min_trading_days=4))
    assert acc.step(day(0), 111_000.0, 111_000.0, 110_000.0, True) is Status.ACTIVE
    assert acc.step(day(1), 111_000.0, 111_000.0, 111_000.0, True) is Status.ACTIVE
    assert acc.step(day(2), 111_000.0, 111_000.0, 111_000.0, False) is Status.ACTIVE
    assert acc.step(day(3), 111_000.0, 111_000.0, 111_000.0, True) is Status.ACTIVE
    assert acc.step(day(4), 111_000.0, 111_000.0, 111_000.0, True) is Status.PASSED


def test_best_day_rule_blocks_pass():
    r = rules(best_day_rule_pct=0.5)
    acc = PropAccount(r, phase=Phase("phase1", 0.10))
    assert acc.step(day(0), 110_500.0, 110_500.0, 110_000.0, True) is Status.ACTIVE  # one day = 100% of profit
    acc.step(day(1), 116_000.0, 116_000.0, 110_500.0, True)
    assert acc.status is Status.ACTIVE      # best day 10.5k > 50% of 16k
    acc.step(day(2), 122_000.0, 122_000.0, 116_000.0, True)
    assert acc.status is Status.PASSED      # best day 10.5k <= 50% of 22k


# --- costs, stops, swaps -------------------------------------------------------------------------

def cost_model() -> CostModel:
    inst = {
        "US500": InstrumentCost("US500", "USD", 1, 0.5, 0.0, triple_swap_day=4, swap_long=-6.0, swap_short=-2.0, stop_slippage_spreads=0.0),
        "EURUSD": InstrumentCost("EURUSD", "USD", 100_000, 0.0001, 2.5, triple_swap_day=2, swap_long=-2.0, swap_short=-2.0),
    }
    return CostModel(inst, SpreadWidening(enabled=True, range_atr_threshold=2.0, factor=2.0))


def test_stop_fill_gap_uses_worse_of_stop_and_open():
    assert stop_fill(+1, stop=99.0, open_=97.0, high=98.0, low=96.0) == 97.0   # gap down through a long stop
    assert stop_fill(+1, stop=99.0, open_=100.0, high=101.0, low=98.5) == 99.0
    assert stop_fill(+1, stop=99.0, open_=100.0, high=101.0, low=99.5) is None
    assert stop_fill(-1, stop=101.0, open_=103.0, high=104.0, low=102.0) == 103.0
    assert stop_fill(-1, stop=101.0, open_=103.0, high=104.0, low=102.0, slippage=0.1) == pytest.approx(103.1)


def test_triple_swap_on_instrument_day():
    cm = cost_model()
    tue = cm.swap("EURUSD", 100_000, 1.10, weekday=1)
    wed = cm.swap("EURUSD", 100_000, 1.10, weekday=2)
    assert tue < 0
    assert wed == pytest.approx(3 * tue)
    assert tue == pytest.approx(-110_000 * 0.02 / 360)
    # Index CFDs: triple swap on Friday, not Wednesday.
    assert cm.swap("US500", 10, 5000, weekday=2) == pytest.approx(cm.swap("US500", 10, 5000, weekday=1))
    assert cm.swap("US500", 10, 5000, weekday=4) == pytest.approx(3 * cm.swap("US500", 10, 5000, weekday=1))


def test_cost_multiplier_and_spread_widening():
    cm = cost_model()
    assert cm.effective_spread("US500") == pytest.approx(0.5)
    assert cm.effective_spread("US500", range_atr_ratio=3.0) == pytest.approx(1.0)
    assert cm.scaled(2.0).effective_spread("US500") == pytest.approx(1.0)
    assert cm.scaled(2.0).commission("EURUSD", 100_000) == pytest.approx(5.0)


def test_ledger_charges_triple_swap_through_session():
    cm = cost_model()
    led = Ledger(cm, balance=INITIAL)
    bar = Bar(open=1.10, high=1.11, low=1.09, close=1.10)
    led.run_session(pd.Timestamp("2024-01-02"), {"EURUSD": bar}, {"EURUSD": 100_000})   # Tuesday
    tue = led.run_session(pd.Timestamp("2024-01-02"), {"EURUSD": bar}).swap
    wed = led.run_session(pd.Timestamp("2024-01-03"), {"EURUSD": bar}).swap            # Wednesday
    assert wed == pytest.approx(3 * tue)


def test_weekend_gap_through_stop_breaches_daily_loss():
    """A Friday position gaps down on Monday below its stop and through the daily floor."""
    cm = cost_model()
    led = Ledger(cm, balance=INITIAL)
    acc = PropAccount(rules())
    fri, mon = pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-11")
    r = led.run_session(fri, {"US500": Bar(5000, 5010, 4995, 5000)}, {"US500": 100}, stops={"US500": 4960})
    assert acc.step(fri, r.balance, r.equity, r.equity_worst, r.traded) is Status.ACTIVE
    # Monday opens at 4930: the stop at 4960 fills at the open, loss ~7k > 5k daily limit.
    r = led.run_session(mon, {"US500": Bar(4930, 4950, 4900, 4940)})
    assert r.stops_hit == ["US500"]
    assert led.positions["US500"].units == 0
    assert r.equity == pytest.approx(r.balance)
    assert INITIAL - r.equity > 5_000
    assert acc.step(mon, r.balance, r.equity, r.equity_worst, r.traded) is Status.FAILED_DAILY
    assert acc.violation.date == mon


def test_worst_equity_marks_open_positions_at_adverse_extreme():
    cm = cost_model()
    led = Ledger(cm, balance=INITIAL)
    r = led.run_session(pd.Timestamp("2024-03-05"), {"US500": Bar(5000, 5050, 4950, 5020)}, {"US500": 10})
    spread_cost = 10 * 0.25
    assert r.equity_worst == pytest.approx(INITIAL - spread_cost - 10 * 50)
    assert r.equity > r.equity_worst


def test_ledger_realises_pnl_on_reduction_and_flip():
    cm = cost_model().scaled(0.0)
    led = Ledger(cm, balance=INITIAL)
    led.run_session(day(0), {"US500": Bar(100, 100, 100, 100)}, {"US500": 10})
    led.run_session(day(1), {"US500": Bar(110, 110, 110, 110)}, {"US500": 4})
    assert led.balance == pytest.approx(INITIAL + 60)
    led.run_session(day(2), {"US500": Bar(120, 120, 120, 120)}, {"US500": -5})
    assert led.balance == pytest.approx(INITIAL + 60 + 4 * 20)
    assert led.positions["US500"].units == -5 and led.positions["US500"].avg_price == 120


# --- challenge simulator ------------------------------------------------------------------------

def _factory(daily_ret: np.ndarray, start="2020-01-01", entry_cost=0.0):
    """Fresh account at every start: P&L accrues only from the start session, minus entry costs."""
    idx = pd.bdate_range(start, periods=len(daily_ret))

    def factory(first: pd.Timestamp):
        pos = idx.get_loc(first)
        equity = INITIAL - entry_cost
        for i in range(pos, len(idx)):
            equity += INITIAL * daily_ret[i]
            yield {"date": idx[i], "balance": equity, "equity": equity, "equity_worst": equity, "traded": True}

    return factory, idx


def test_challenge_steady_growth_passes_both_phases():
    factory, idx = _factory(np.full(400, 0.005))
    out = simulate_challenges(factory, idx, rules(), start="2020-01-01", end="2020-06-30")
    assert (out["phase1_result"] == "passed").all() and (out["phase2_result"] == "passed").all()
    assert (out["phase1_days"] == 20).all()      # 10% at 0.5% per session
    assert (out["phase2_days"] == 10).all()      # fresh account for phase 2
    s = summarise(out, n_phases=2, horizon=126)
    assert s["pass_all"] == 1.0 and s["median_sessions_phase1"] == 20 and s["median_sessions_all"] == 30


def test_challenge_horizons_and_unlimited():
    factory, idx = _factory(np.full(700, 0.0011))   # ~91 sessions for phase 1
    out = simulate_challenges(factory, idx, rules(), start="2020-01-01", end="2020-01-31")
    table = summarise_horizons(out, n_phases=2)
    by_h = table.set_index("horizon")
    assert by_h.loc["63", "timeout"] == 1.0
    assert by_h.loc["126", "pass_phase1"] == 1.0
    assert by_h.loc["unlimited", "pass_all"] == 1.0


def test_challenge_crash_fails():
    ret = np.zeros(400)
    ret[30] = -0.06
    factory, idx = _factory(ret)
    out = simulate_challenges(factory, idx, rules(), start="2020-01-01", end="2020-01-31")
    assert out.iloc[0]["phase1_result"] == "failed_daily_loss"
    assert summarise(out, 2, None)["fail_daily"] == 1.0


def test_challenge_starts_on_fresh_account_with_entry_costs():
    # Losses before a start never leak into it; entry costs are paid at every start.
    ret = np.full(400, 0.001)
    ret[:40] = -0.002
    factory, idx = _factory(ret, entry_cost=500.0)
    out = simulate_challenges(factory, idx, rules(), start="2020-03-01", end="2020-04-30")
    assert (out["phase1_result"] == "passed").all()
    assert (out["phase1_days"] == 105).all()     # (10% + 0.5% entry cost) / 0.1% per session


# --- configuration --------------------------------------------------------------------------------

def test_presets_load_and_warn_on_unverified_values():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = load_prop_rules("ftmo_2step_swing")
    assert any(issubclass(w.category, UnverifiedConfigWarning) for w in caught)
    assert r.daily_loss_pct == 0.05 and r.total_loss_pct == 0.10
    assert [p.profit_target_pct for p in r.phases] == [0.10, 0.05]
    for preset in ("ftmo_1step", "fundednext_stellar_2step"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert load_prop_rules(preset).initial_capital == INITIAL


def test_cost_yaml_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = CostModel.from_profile("estimates")
    assert cm.profile == "estimates"
    assert cm.instruments["XAUUSD"].contract_size == 100
    assert cm.instruments["US500"].triple_swap_day == 4


def test_futures_based_cfds_only_pay_the_broker_markup():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = CostModel.from_profile("estimates")
    long_ = cm.swap("XAUUSD", 100, 2000.0, weekday=1)
    short = cm.swap("XAUUSD", -100, 2000.0, weekday=1)
    assert long_ == pytest.approx(short) == pytest.approx(-200_000 * 0.02 / 360)


def test_positive_swaps_are_never_credited():
    cm = cost_model().with_instrument("EURUSD", swap_short=1.5)
    assert cm.swap("EURUSD", -100_000, 1.1, weekday=1) == 0.0


def test_cost_profile_selection_and_holdout_guard(monkeypatch):
    from common.utils import settings as st
    assert st.cost_profile() == "estimates"
    assert st.cost_profile("estimates") == "estimates"
    with pytest.raises(FileNotFoundError):
        st.cost_profile("mt5")          # not generated yet
    with pytest.raises(ValueError):
        st.cost_profile("cheap")
    assert not st.holdout_allowed("estimates")
    assert "provisoires" in st.profile_banner("estimates")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert load_prop_rules("ftmo_2step_swing").leverage["fx"] == 30
