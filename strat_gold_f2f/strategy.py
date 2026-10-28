"""Forecast-to-Fill strategy: trade state machine, walk-forward, paper replication.

The state machine runs on daily closes. At the close of t, with the regime probability and
the ATR known at t:

* flat  -> enter long if p_bull >= threshold and the smoothed slope is positive;
* long  -> exit on the hard stop (entry - a ATR), the trailing stop (peak - b ATR), the
           timeout, or regime de-risking (p_bear > 0.5); otherwise the weight is re-sized.

The weight decided at the close of t is traded either at the next open (prop engine), at the
close of t+1 (paper "T+1" baseline), or - for the paper's headline figures only - applied to
the return of t itself ("T+0"), which uses the closing price at which the signal is computed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from common.tuning.walkforward import Fold, make_folds
from strat_gold_f2f.signal import ANNUAL, atr, ewma_vol, regime_signal
from strat_gold_f2f.sizing import impact_adjusted_kelly, position_weight

ProbFn = Callable[[pd.Timestamp], float]


def run_rules(bars: pd.DataFrame, sig: pd.DataFrame, cfg: dict[str, Any], kelly: float | None,
              meta_prob: ProbFn | None = None, meta_mode: str = "size", meta_threshold: float = 0.5) -> pd.DataFrame:
    """Daily weight decided at each close. ``kelly=None`` gives unit weights (used to estimate mu, sigma)."""
    sc, ec, xc = cfg["signal"], cfg["exits"], cfg["sizing"]
    close = bars["close"].to_numpy()
    a = atr(bars, int(ec["atr_window"])).to_numpy()
    vol = ewma_vol(bars["ret"].fillna(0.0), int(xc["ewma_span"])).to_numpy()
    p_bull = sig["p_bull"].to_numpy()
    slope = sig["slope"].to_numpy()
    n = len(bars)
    weight = np.zeros(n)
    stop = np.full(n, np.nan)
    entry_flag = np.zeros(n, bool)
    exit_reason = np.array([""] * n, dtype=object)
    in_trade, entry, peak, days, size_mult = False, 0.0, 0.0, 0, 1.0
    thr, derisk = float(sc["entry_threshold"]), float(sc["derisk_p_bear"])
    for t in range(n):
        if np.isnan(p_bull[t]) or np.isnan(a[t]) or np.isnan(vol[t]):
            continue
        if in_trade:
            days += 1
            peak = max(peak, close[t])
            hard = entry - float(ec["hard_stop_atr"]) * a[t]
            trail = peak - float(ec["trailing_stop_atr"]) * a[t]
            reason = ""
            if close[t] <= hard:
                reason = "hard_stop"
            elif close[t] <= trail:
                reason = "trailing_stop"
            elif days >= int(ec["timeout_sessions"]):
                reason = "timeout"
            elif 1 - p_bull[t] > derisk and sc.get("derisk_action", "close") == "close":
                reason = "derisk"
            if reason:
                in_trade = False
                exit_reason[t] = reason
                continue
        elif p_bull[t] >= thr and slope[t] > 0:
            mult = 1.0
            if meta_prob is not None:
                p = meta_prob(bars.index[t])
                if meta_mode == "filter" and p < meta_threshold:
                    continue
                mult = p if meta_mode == "size" else 1.0
            in_trade, entry, peak, days, size_mult = True, close[t], close[t], 0, mult
            entry_flag[t] = True
        if in_trade:
            w = 1.0 if kelly is None else position_weight(p_bull[t], vol[t], kelly, xc)
            if kelly is not None and 1 - p_bull[t] > derisk and sc.get("derisk_action") == "halve":
                w *= 0.5
            weight[t] = w * size_mult
            stop[t] = max(entry - float(ec["hard_stop_atr"]) * a[t], peak - float(ec["trailing_stop_atr"]) * a[t])
    return pd.DataFrame({"weight": weight, "stop": stop, "entry": entry_flag, "exit_reason": exit_reason}, index=bars.index)


def strategy_returns(weight: pd.Series, ret: pd.Series, lag: int, k_bps: float, gamma: float,
                     vol: pd.Series | None = None) -> pd.Series:
    """Paper-style daily net returns: w_{t-lag} r_t - k |dw| - gamma sigma_1d |dw|^{3/2}.

    lag = 0 is the paper's T+0 headline (same-close execution, look-ahead); lag = 1 is T+1.
    """
    w = weight.shift(lag).fillna(0.0)
    dw = w.diff().abs().fillna(w.abs())
    sigma = vol.reindex(w.index).fillna(0.0) if vol is not None else 0.0
    cost = k_bps * 1e-4 * dw + gamma * sigma * dw ** 1.5
    return (w * ret.fillna(0.0) - cost).rename("ret")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANNUAL)) if len(r) > 2 and r.std() > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def kelly_from_train(bars: pd.DataFrame, sig: pd.DataFrame, cfg: dict, train_end: pd.Timestamp) -> tuple[float, float, float]:
    """mu, sigma of unit-weight trade returns on the train window and the resulting f*."""
    unit = run_rules(bars.loc[:train_end], sig.loc[:train_end], cfg, kelly=None)["weight"]
    r = (unit.shift(1) * bars["ret"].loc[:train_end]).loc[unit.shift(1) > 0].dropna()
    mu, sd = (float(r.mean()), float(r.std())) if len(r) > 20 else (0.0, 0.0)
    xc = cfg["sizing"]
    f = impact_adjusted_kelly(mu, sd, float(xc["turnover_n"]), float(xc["linear_cost_bps"]) * 1e-4, float(xc["impact_gamma"]))
    return mu, sd, f


def with_params(cfg: dict, params: dict) -> dict:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    for key, value in params.items():
        for section in ("signal", "exits", "sizing"):
            if key in out[section]:
                out[section][key] = value
    return out


def fold_weights(bars: pd.DataFrame, cfg: dict, fold: Fold, meta_prob: ProbFn | None = None) -> tuple[pd.DataFrame, dict]:
    """Weights of the fold's test window with the parameters of ``cfg`` (fitted stats on train only)."""
    window = bars.loc[fold.train_start:fold.test_end]
    sc = cfg["signal"]
    sig = regime_signal(window, float(sc["ema_lambda"]), int(sc["momentum_k"]), float(sc["omega"]), fold.train_end)
    mu, sd, f = kelly_from_train(window, sig, cfg, fold.train_end)
    mc = cfg.get("meta_labeling", {})
    rules = run_rules(window, sig, cfg, kelly=f, meta_prob=meta_prob,
                      meta_mode=mc.get("mode", "size"), meta_threshold=float(mc.get("filter_threshold", 0.5)))
    out = rules.join(sig).loc[fold.test_start:fold.test_end]
    return out, {"mu": mu, "sigma": sd, "kelly": f}


def select_on_train(bars: pd.DataFrame, cfg: dict, fold: Fold) -> tuple[dict, float]:
    """Paper replication: pick the grid combination with the best train Sharpe at T+1."""
    grid = cfg["replication"]["grid"]
    keys = list(grid)
    xc = cfg["sizing"]
    window = bars.loc[fold.train_start:fold.train_end]
    vol = ewma_vol(window["ret"].fillna(0.0), int(xc["ewma_span"]))
    best, best_s = None, -np.inf
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        c = with_params(cfg, params)
        sc = c["signal"]
        sig = regime_signal(window, float(sc["ema_lambda"]), int(sc["momentum_k"]), float(sc["omega"]), fold.train_end)
        _, _, f = kelly_from_train(window, sig, c, fold.train_end)
        w = run_rules(window, sig, c, kelly=f)["weight"]
        s = sharpe(strategy_returns(w, window["ret"], 1, float(xc["linear_cost_bps"]), float(xc["impact_gamma"]), vol))
        if np.isfinite(s) and s > best_s:
            best, best_s = params, s
    return best or {}, best_s


def replicate_paper(bars: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """Walk-forward replication (10y rolling train, 6m test, 2015 -> 2025-10) at T+0 and T+1."""
    rc, wc, xc = cfg["replication"], cfg["walk_forward"], cfg["sizing"]
    folds = make_folds(bars.index, float(wc["train_years"]), int(wc["test_months"]), bool(wc["expanding"]),
                       first_test=pd.Timestamp(rc["first_test"]), last_date=pd.Timestamp(rc["last_date"]))
    pieces, chosen = [], []
    for fold in folds:
        params, train_s = select_on_train(bars, cfg, fold)
        w, info = fold_weights(bars, with_params(cfg, params), fold)
        pieces.append(w)
        chosen.append({"test_start": fold.test_start.date(), **params, "train_sharpe": round(train_s, 3), **info})
    weights = pd.concat(pieces)
    ret = bars["ret"].reindex(weights.index)
    vol = ewma_vol(bars["ret"].fillna(0.0), int(xc["ewma_span"])).reindex(weights.index)
    out = {"folds": pd.DataFrame(chosen), "weights": weights}
    for lag, name in ((0, "T+0"), (1, "T+1")):
        r = strategy_returns(weights["weight"], ret, lag, float(xc["linear_cost_bps"]), float(xc["impact_gamma"]), vol)
        out[name] = {
            "returns": r, "sharpe": sharpe(r), "max_drawdown": max_drawdown(r),
            "ann_return": float(r.mean() * ANNUAL), "ann_vol": float(r.std() * np.sqrt(ANNUAL)),
            "exposure": float((weights["weight"] > 0).mean()), "avg_weight": float(weights["weight"].mean()),
            "n_days": int(len(r)), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        }
    return out


# --- CFD adaptation ------------------------------------------------------------------------------

def cfd_config(cfg: dict, cost_fraction: float) -> dict:
    """Paper sizing with CFD costs: no impact term, round-trip linear cost from the cost profile."""
    out = with_params(cfg, {})
    out["sizing"]["impact_gamma"] = float(cfg.get("cfd", {}).get("impact_gamma", 0.0))
    out["sizing"]["linear_cost_bps"] = 2 * cost_fraction * 1e4
    return out


def meta_prob_fn(bars: pd.DataFrame, cfg: dict, fold: Fold, seed: int = 42) -> ProbFn:
    """Fit the meta-model on the primary entries of the train window and return P(win | entry)."""
    from strat_gold_f2f.meta import MetaModel, entry_features, triple_barrier

    window = bars.loc[fold.train_start:fold.test_end]
    sc, ec, mc, xc = cfg["signal"], cfg["exits"], cfg["meta_labeling"], cfg["sizing"]
    sig = regime_signal(window, float(sc["ema_lambda"]), int(sc["momentum_k"]), float(sc["omega"]), fold.train_end)
    feats = entry_features(window, sig, int(ec["atr_window"]), int(xc["ewma_span"]))
    _, _, f = kelly_from_train(window, sig, cfg, fold.train_end)
    primary = run_rules(window.loc[:fold.train_end], sig.loc[:fold.train_end], cfg, kelly=f)
    entries = primary.index[primary["entry"]]
    events = triple_barrier(window.loc[:fold.train_end], entries, float(mc["profit_take_atr"]),
                            float(ec["hard_stop_atr"]), int(ec["timeout_sessions"]), int(ec["atr_window"]))
    model = MetaModel(seed).fit(feats, events, fold.train_end)
    return lambda d: model.prob(feats.loc[d])


def cfd_positions(bars: pd.DataFrame, cfg: dict, folds: list[Fold], cost_fraction: float,
                  use_meta: bool = False, seed: int = 42, params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward weights (fraction of capital) and protective stop levels on the test windows."""
    c = cfd_config(with_params(cfg, params or {}), cost_fraction)
    pieces, infos = [], []
    for fold in folds:
        prob = meta_prob_fn(bars, c, fold, seed) if use_meta else None
        w, info = fold_weights(bars, c, fold, meta_prob=prob)
        pieces.append(w)
        infos.append({"test_start": fold.test_start, **info})
    return pd.concat(pieces), pd.DataFrame(infos)
