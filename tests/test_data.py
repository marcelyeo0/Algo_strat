import lzma
import struct

import numpy as np
import pandas as pd
import pytest

from common.data.bars import finalize_bars
from common.data.calendar import aggregate_to_sessions, session_dates
from common.data.dukascopy import decode_hour_candles
from common.data.quality import check_bars
from common.data.rolls import brent_last_trade, cl_last_trade, first_notice_day, roll_flags

# --- roll calendars ---------------------------------------------------------------------------

def test_cl_last_trade_known_dates():
    # Jan-2024 WTI: 25 Dec 2023 is a holiday -> previous business day 22 Dec -> minus 3 = 19 Dec.
    assert cl_last_trade(2024, 1) == pd.Timestamp("2023-12-19")
    # Jul-2023 WTI: 25 Jun 2023 is a Sunday -> 23 Jun -> minus 3 = 20 Jun.
    assert cl_last_trade(2023, 7) == pd.Timestamp("2023-06-20")


def test_brent_last_trade_rules():
    assert brent_last_trade(2024, 1) == pd.Timestamp("2023-11-30")
    # Pre-2016 rule: business day preceding the 15th day before the delivery month.
    assert brent_last_trade(2015, 6) == pd.Timestamp("2015-05-15")


def test_first_notice_day_gold():
    assert first_notice_day(2024, 2) == pd.Timestamp("2024-01-31")


def test_roll_flags_mark_first_session_after_switch():
    idx = pd.bdate_range("2023-12-01", "2024-01-31")
    contract = {"rule": "cl_last_trade", "months": list(range(1, 13)), "neutralise_window_bdays": 0}
    flags = roll_flags(idx, contract)
    assert flags[pd.Timestamp("2023-12-20")]          # first session on the Feb-2024 contract
    assert not flags[pd.Timestamp("2023-12-19")]
    assert flags.sum() == 2                            # Dec-19 and Jan-19 switches


# --- session calendar --------------------------------------------------------------------------

def test_session_dates_follow_new_york_close():
    ts = pd.DatetimeIndex(
        ["2024-03-15 20:00", "2024-03-17 21:00", "2024-01-15 21:00", "2024-01-15 22:00"], tz="UTC"
    )
    got = session_dates(ts)
    assert list(got.strftime("%Y-%m-%d")) == ["2024-03-15", "2024-03-18", "2024-01-15", "2024-01-16"]


def test_aggregate_to_sessions_ohlc():
    idx = pd.date_range("2024-01-15 22:00", periods=24, freq="h", tz="UTC")  # one EST session
    prices = np.arange(24, dtype=float) + 100
    bars = pd.DataFrame({"open": prices, "high": prices + 1, "low": prices - 1, "close": prices + 0.5,
                         "volume": 1.0}, index=idx)
    daily = aggregate_to_sessions(bars)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["open"] == 100 and row["close"] == 123.5 and row["high"] == 124 and row["low"] == 99
    assert row["volume"] == 24


# --- Dukascopy decoding -----------------------------------------------------------------------

def test_decode_hour_candles_roundtrip():
    month = pd.Timestamp("2024-01-01")
    records = [(3600 * h, 110000 + h, 110005 + h, 109990 + h, 110010 + h, 1.5) for h in range(3)]
    raw = b"".join(struct.pack(">5if", *r) for r in records)
    raw = lzma.decompress(lzma.compress(raw, format=lzma.FORMAT_ALONE))
    df = decode_hour_candles(raw, month, 0.00001)
    assert len(df) == 3
    assert df.index[1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")
    assert df["open"].iloc[0] == pytest.approx(1.10000)
    assert df["high"].iloc[0] == pytest.approx(1.10010)


# --- bars and quality ---------------------------------------------------------------------------

def _bars_with_roll():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = np.array([100, 101, 102, 103, 110, 111, 112, 113, 114, 115], dtype=float)  # +7 contract gap on day 4
    raw = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[4] = True
    return raw, mask


def test_finalize_bars_neutralises_roll_and_back_adjusts():
    raw, mask = _bars_with_roll()
    bars = finalize_bars(raw, mask)
    assert np.isnan(bars["ret"].iloc[4])
    assert bars["close"].iloc[-1] == pytest.approx(115.0)      # latest contract keeps its level
    assert bars["close"].iloc[4] == pytest.approx(bars["close"].iloc[3])
    # Returns away from the roll are unchanged.
    assert bars["ret"].iloc[6] == pytest.approx(112 / 111 - 1)


def test_quality_flags_but_keeps_outliers():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    close[200:] *= 1.3  # genuine 30% jump, not a roll
    raw = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close}, index=idx)
    bars = finalize_bars(raw)
    rep = check_bars(bars, "TEST", "synthetic")
    assert rep.outliers >= 1
    assert len(bars) == 300                                    # nothing removed
    assert bars["ret"].iloc[200] == pytest.approx(close[200] / close[199] - 1)
