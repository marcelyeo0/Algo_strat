"""Purged K-fold cross-validation with embargo (López de Prado, Advances in Financial ML, ch. 7).

Observations are indexed by date; each label spans [t, t + horizon]. Training rows whose label
window overlaps the test fold are purged, and an embargo removes rows right after the test fold.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def purged_kfold(dates: pd.DatetimeIndex, n_splits: int = 5, horizon: int = 1,
                 embargo: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_positions, test_positions) over contiguous date blocks.

    ``dates`` may contain repeated dates (several assets pooled); folds are built on the unique
    sorted dates so that one date never sits on both sides.
    """
    dates = pd.DatetimeIndex(dates)
    unique = dates.unique().sort_values()
    blocks = np.array_split(np.arange(len(unique)), n_splits)
    pos_of = unique.get_indexer(dates)
    for block in blocks:
        if len(block) == 0:
            continue
        lo, hi = block[0], block[-1]
        test = (pos_of >= lo) & (pos_of <= hi)
        # Purge labels overlapping the test block, embargo after it.
        purge = (pos_of >= lo - horizon) & (pos_of <= hi + embargo)
        train = ~purge
        yield np.flatnonzero(train), np.flatnonzero(test)
