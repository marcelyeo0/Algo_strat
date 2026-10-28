import pandas as pd
import pytest

from common.costs.model import CostModel, InstrumentCost, SpreadWidening
from common.costs.mt5_specs import implied_markup

DAY = pd.Timestamp("2015-06-02")      # a Tuesday: one night of swap


def model(rates: dict[str, float], markup: float = 0.0, credit: str = "zero") -> CostModel:
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    rates_df = pd.DataFrame({c: v for c, v in rates.items()}, index=idx)
    inst = {
        "EURUSD": InstrumentCost("EURUSD", "USD", 100_000, 0.0001, 0.0, triple_swap_day=2, swap_mode="rates",
                                 base_ccy="EUR", swap_markup=markup, asset_class="fx"),
        "US500": InstrumentCost("US500", "USD", 1, 0.5, 0.0, triple_swap_day=4, swap_mode="index_rates",
                                swap_markup=markup, asset_class="index"),
        "XAUUSD": InstrumentCost("XAUUSD", "USD", 100, 0.3, 0.0, triple_swap_day=2, swap_mode="markup",
                                 swap_markup=markup, asset_class="metal"),
    }
    return CostModel(inst, SpreadWidening(), rates=rates_df, credit_mode=credit)


@pytest.mark.parametrize("symbol,units,price", [("EURUSD", 100_000, 1.1), ("EURUSD", -100_000, 1.1),
                                                ("US500", 10, 4000.0), ("US500", -10, 4000.0),
                                                ("XAUUSD", 100, 1800.0), ("XAUUSD", -100, 1800.0)])
@pytest.mark.parametrize("credit", ["zero", "real"])
def test_no_swap_with_zero_rates_and_zero_markup(symbol, units, price, credit):
    cm = model({"EUR": 0.0, "USD": 0.0}, markup=0.0, credit=credit)
    assert cm.swap(symbol, units, price, DAY.weekday(), 1.0, DAY) == 0.0


def test_fx_swap_follows_rate_differential_and_markup():
    cm = model({"EUR": 1.0, "USD": 4.0}, markup=0.5)
    # Long EURUSD: (1 - 4) - 0.5 = -3.5 % per year; short: (4 - 1) - 0.5 = +2.5 % -> not credited.
    assert cm.swap_rate("EURUSD", 1, DAY) == pytest.approx(-3.5)
    assert cm.swap_rate("EURUSD", -1, DAY) == 0.0
    assert cm.with_credit_mode("real").swap_rate("EURUSD", -1, DAY) == pytest.approx(2.5)
    notional = 100_000 * 1.1
    assert cm.swap("EURUSD", 100_000, 1.1, DAY.weekday(), 1.0, DAY) == pytest.approx(-notional * 0.035 / 360)


def test_index_swap_long_pays_rate_plus_markup_short_receives_rate_minus_markup():
    cm = model({"USD": 3.0}, markup=2.5, credit="real")
    assert cm.swap_rate("US500", 1, DAY) == pytest.approx(-5.5)
    assert cm.swap_rate("US500", -1, DAY) == pytest.approx(0.5)
    assert model({"USD": 1.0}, markup=2.5).swap_rate("US500", -1, DAY) == pytest.approx(-1.5)


def test_futures_based_swap_is_markup_only_and_independent_of_rates():
    a = model({"USD": 0.0}, markup=2.0).swap_rate("XAUUSD", 1, DAY)
    b = model({"USD": 5.0}, markup=2.0).swap_rate("XAUUSD", -1, DAY)
    assert a == b == pytest.approx(-2.0)


def test_rates_vary_over_time():
    idx = pd.date_range("2010-01-01", "2020-12-31", freq="D")
    rates = pd.DataFrame({"EUR": 0.0, "USD": 0.0}, index=idx)
    rates.loc["2018-01-01":, "USD"] = 2.0
    cm = model({"EUR": 0.0, "USD": 0.0}).with_instrument("EURUSD")   # clone keeps the rates
    cm.rates = rates
    assert cm.swap_rate("EURUSD", 1, pd.Timestamp("2015-01-05")) == 0.0
    assert cm.swap_rate("EURUSD", 1, pd.Timestamp("2019-01-07")) == pytest.approx(-2.0)


def test_implied_markup_from_mt5_swaps():
    # FX: long swap -4%, short +1% with r_b=1, r_q=4 -> m_long = -3 + 4 = 1, m_short = 3 - 1 = 2 -> 2 (prudent)
    assert implied_markup("rates", -4.0, 1.0, 1.0, 4.0) == pytest.approx(2.0)
    # Index: long -6% with r=3 -> m = 3; short +0.5 -> m = 2.5 -> 3
    assert implied_markup("index_rates", -6.0, 0.5, 0.0, 3.0) == pytest.approx(3.0)
