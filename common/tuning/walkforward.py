"""Walk-forward fold generation and the fixed research periods."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Strict time split (see docs/GUIDE.md). The holdout is locked and only used by `algo holdout`.
DEV_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def period_bounds(period: str) -> tuple[pd.Timestamp | None, pd.Timestamp]:
    """(first test date, last date allowed) of a research period."""
    if period == "dev":
        return None, DEV_END
    if period == "validation":
        return VALIDATION_START, VALIDATION_END
    if period == "holdout":
        return HOLDOUT_START, pd.Timestamp.max
    raise ValueError(f"Unknown period '{period}'")


def make_folds(
    index: pd.DatetimeIndex,
    train_years: float,
    test_months: int,
    expanding: bool = True,
    first_test: pd.Timestamp | None = None,
    last_date: pd.Timestamp | None = None,
) -> list[Fold]:
    """Consecutive, non-overlapping test windows, each preceded by its training window.

    The first test window starts ``train_years`` after the first date (or at ``first_test``).
    Test windows stop at ``last_date``; nothing after it is ever touched.
    """
    index = pd.DatetimeIndex(index).sort_values()
    last_date = min(last_date, index[-1]) if last_date is not None else index[-1]
    data_start = index[0]
    test_start = first_test or (data_start + pd.DateOffset(months=int(round(train_years * 12))))
    folds = []
    while test_start <= last_date:
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), last_date)
        train_end = test_start - pd.Timedelta(days=1)
        train_start = data_start if expanding else max(data_start, test_start - pd.DateOffset(months=int(round(train_years * 12))))
        if index[(index >= train_start) & (index <= train_end)].size > 0:
            folds.append(Fold(train_start, train_end, test_start, test_end))
        test_start = test_end + pd.Timedelta(days=1)
    return folds
