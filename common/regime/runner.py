"""Regime scores for a strategy basket, cached on disk (fitting is the expensive part)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common.regime.features import basket_returns
from common.regime.filter import load_regime_config, walk_forward_scores
from common.tuning.walkforward import Fold
from common.utils.meta import config_hash
from common.utils.paths import data_dir


def regime_scores(panel: dict[str, pd.DataFrame], symbols: list[str], folds: list[Fold],
                  cfg: dict[str, Any] | None = None, seed: int = 42, use_cache: bool = True) -> pd.DataFrame:
    """Walk-forward out-of-sample regime scores of the inverse-vol basket of ``symbols``."""
    cfg = cfg or load_regime_config()
    ret = basket_returns(panel, symbols)
    key = config_hash({
        "cfg": {k: v for k, v in cfg.items() if k != "output"},   # output mode does not change scores
        "symbols": sorted(symbols), "seed": seed,
        "folds": [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds],
        "last": str(ret.index[-1]), "n": len(ret),
    })
    path = data_dir() / "cache" / "regime" / f"{key}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    scores = walk_forward_scores(ret, folds, cfg, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(path)
    return scores
