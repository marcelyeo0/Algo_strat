"""Roll calendars of the futures used as price proxies (yfinance continuous ``=F`` series).

Roll days are derived from the exchange expiry rules, never from price jumps: a statistical
threshold would also delete genuine market moves. The threshold is only used in the quality
report to flag anomalies (see ``common.data.quality``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def _is_bday(day: pd.Timestamp) -> bool:
    return BDAY.is_on_offset(day)


def _prev_bday(day: pd.Timestamp) -> pd.Timestamp:
    """Business day strictly before ``day``."""
    return day - BDAY


def _last_bday_of_month(year: int, month: int) -> pd.Timestamp:
    end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return end if _is_bday(end) else _prev_bday(end)


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + delta
    return idx // 12, idx % 12 + 1


def cl_last_trade(year: int, month: int) -> pd.Timestamp:
    """NYMEX WTI last trading day for the contract delivered in ``year-month``."""
    py, pm = _shift_month(year, month, -1)
    ref = pd.Timestamp(year=py, month=pm, day=25)
    if not _is_bday(ref):
        ref = _prev_bday(ref)
    return ref - 3 * BDAY


def brent_last_trade(year: int, month: int) -> pd.Timestamp:
    """ICE Brent last trading day for the contract delivered in ``year-month``."""
    if (year, month) >= (2016, 3):
        py, pm = _shift_month(year, month, -2)
        return _last_bday_of_month(py, pm)
    ref = pd.Timestamp(year=year, month=month, day=1) - pd.Timedelta(days=15)
    return _prev_bday(ref)


def ng_last_trade(year: int, month: int) -> pd.Timestamp:
    """NYMEX natural gas: 3 business days before the first calendar day of the delivery month."""
    first = pd.Timestamp(year=year, month=month, day=1)
    return first - 3 * BDAY


def first_notice_day(year: int, month: int) -> pd.Timestamp:
    """COMEX first notice day: last business day of the month preceding delivery."""
    py, pm = _shift_month(year, month, -1)
    return _last_bday_of_month(py, pm)


def last_session_on_old_contract(contract: dict[str, Any], year: int, month: int) -> pd.Timestamp:
    rule = contract["rule"]
    if rule == "cl_last_trade":
        return cl_last_trade(year, month)
    if rule == "brent_last_trade":
        return brent_last_trade(year, month)
    if rule == "ng_last_trade":
        return ng_last_trade(year, month)
    if rule == "before_first_notice":
        return first_notice_day(year, month) - int(contract.get("offset_bdays", 0)) * BDAY
    raise ValueError(f"Unknown roll rule: {rule}")


def switch_dates(contract: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Last session on each expiring contract between ``start`` and ``end``."""
    dates = []
    for year in range(start.year - 1, end.year + 2):
        for month in contract["months"]:
            d = last_session_on_old_contract(contract, year, month)
            if start <= d <= end:
                dates.append(d)
    return sorted(dates)


def roll_flags(index: pd.DatetimeIndex, contract: dict[str, Any]) -> pd.Series:
    """Boolean series, True on the sessions whose return spans a contract change.

    The flagged session is the first session in ``index`` strictly after the last session of
    the old contract, plus ``neutralise_window_bdays`` following sessions.
    """
    flags = pd.Series(False, index=index)
    if len(index) == 0:
        return flags
    window = int(contract.get("neutralise_window_bdays", 0))
    positions = index.searchsorted(switch_dates(contract, index[0], index[-1]), side="right")
    for pos in positions:
        for k in range(window + 1):
            if pos + k < len(index):
                flags.iloc[pos + k] = True
    return flags
