"""Walk-forward position generation of strategy 1 (baselines and Deep Momentum Network).

Positions are volatility-scaled exposures per asset (fraction of capital at the asset target
volatility), decided at the close of t. Portfolio aggregation is done in
``common.backtest.portfolio``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.costs.model import CostModel
from common.tuning.walkforward import Fold
from strat_tsmom_dl.baselines import baseline_positions
from strat_tsmom_dl.dmn import build_sequences, predict_positions, train_dmn
from strat_tsmom_dl.features import ANNUAL, asset_features
from strat_tsmom_dl.mtdp import mtdp_fold_scores


def asset_cost_fractions(panel: dict[str, pd.DataFrame], symbols: list[str], costs: CostModel) -> dict[str, float]:
    """Per-asset cost of trading one unit of notional, at the median price (turnover penalty)."""
    return {s: costs.trade_cost_fraction(s, float(panel[s]["raw_close"].median())) for s in symbols}


def feature_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    cols = [c for c in frame if c.startswith(("nret_", "macd_"))]
    if variant == "lstm_mtdp":
        cols += [c for c in frame if c.startswith("mtdp_")]
    return cols


def dmn_fold(panel: dict[str, pd.DataFrame], cfg: dict[str, Any], fold: Fold, cost_frac: dict[str, float],
             seed: int, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict]:
    """Train on the fold's train window, return vol-scaled positions on its test window."""
    p = {**cfg["dmn"], **(params or {})}
    variant = p["variant"]
    target = float(cfg["volatility"]["asset_target"])
    mt = cfg["features"]["mtdp"]
    fast = int(p.get("fast_weeks", mt.get("fast_weeks_default", 8)))
    feats = {s: asset_features(panel[s], {**cfg, "features": {**cfg["features"], "mtdp": {**mt, "fast_weeks_default": fast}}})
             .loc[:fold.test_end] for s in cfg["symbols"]}
    if variant == "lstm_mtdp":
        scores = mtdp_fold_scores(feats, fold.train_end, int(mt["slow_weeks"]), fast, seed)
        feats = {s: f.join(scores[s]) for s, f in feats.items()}
    cols = feature_columns(next(iter(feats.values())), variant)

    # Scaler fitted on pooled train rows only.
    train_rows = pd.concat([f.loc[:fold.train_end, cols] for f in feats.values()]).dropna()
    mu, sd = train_rows.mean(), train_rows.std().replace(0, 1.0)
    frames = {}
    for s, f in feats.items():
        g = f.copy()
        g[cols] = (g[cols] - mu) / sd
        g["scale"] = target / (g["vol"] * np.sqrt(ANNUAL))
        frames[s] = g.dropna(subset=[*cols, "scale"])

    train_dates = pd.DatetimeIndex(sorted(set().union(*[f.loc[:fold.train_end].index for f in frames.values()])))
    cut = train_dates[int(len(train_dates) * (1 - float(p["validation_share"])))]
    length = int(p["sequence_length"])

    def window(lo, hi):
        out = {}
        for s, f in frames.items():
            w = f.loc[lo:hi].iloc[:-1] if hi == fold.train_end else f.loc[lo:hi]   # last train target is a test day
            w = w.dropna(subset=["next_ret"])
            if len(w) >= length:
                out[s] = w
        return out

    tr = build_sequences(window(None, cut - pd.Timedelta(days=1)), cols, length, cost_frac)
    va = build_sequences(window(cut, fold.train_end), cols, min(length, 21), cost_frac)
    model, info = train_dmn(tr, va, int(p["hidden_size"]), float(p["dropout"]), float(p["learning_rate"]),
                            int(p["batch_size"]), float(p["max_grad_norm"]), int(p["max_epochs"]),
                            int(p["patience"]), float(p["turnover_penalty"]), seed)
    out = {}
    for s, f in frames.items():
        dates = f.loc[fold.test_start:fold.test_end].index
        x = predict_positions(model, f, cols, length, dates)
        out[s] = x * f.loc[x.index, "scale"]
    return pd.DataFrame(out).sort_index(), info


def dmn_positions(panel: dict[str, pd.DataFrame], cfg: dict[str, Any], folds: list[Fold], costs: CostModel,
                  seed: int = 42, params: dict[str, Any] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    cost_frac = asset_cost_fractions(panel, cfg["symbols"], costs)
    pieces, infos = [], []
    for fold in folds:
        pos, info = dmn_fold(panel, cfg, fold, cost_frac, seed, params)
        pieces.append(pos)
        infos.append({"test_start": fold.test_start, **info})
    return pd.concat(pieces).sort_index(), infos


def baseline_test_positions(panel: dict[str, pd.DataFrame], cfg: dict[str, Any], folds: list[Fold], kind: str) -> pd.DataFrame:
    """Baselines have no fitted parameters; they are simply sliced to the test windows."""
    pos = baseline_positions(panel, cfg, kind)
    return pd.concat([pos.loc[f.test_start:f.test_end] for f in folds]).sort_index()
