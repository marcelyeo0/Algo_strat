"""Gaussian HMM with causal (forward-only) filtering.

hmmlearn's ``predict`` / ``predict_proba`` use the full sequence (Viterbi or forward-backward
smoothing), so the state at t would depend on observations after t. In live trading only the
forward filter is available; :func:`forward_filter` implements it explicitly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass
class FittedHMM:
    model: GaussianHMM
    mean: np.ndarray            # scaler fitted on the train window only
    std: np.ndarray
    n_states: int
    bic: float

    def scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def covariances(self) -> np.ndarray:
        cov = self.model.covars_
        if cov.ndim == 2:  # diag stored as (n, d)
            cov = np.stack([np.diag(c) for c in cov])
        return cov


def fit_hmm(train: pd.DataFrame, n_states: list[int], covariance_type: str = "full", n_init: int = 3,
            n_iter: int = 200, seed: int = 42) -> FittedHMM:
    """Fit candidates on the train window and keep the lowest BIC."""
    x = train.to_numpy(dtype=float)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    xs = (x - mean) / std
    best: FittedHMM | None = None
    for k in n_states:
        cand, cand_ll = None, -np.inf
        for r in range(n_init):
            m = GaussianHMM(n_components=k, covariance_type=covariance_type, n_iter=n_iter,
                            random_state=seed + 101 * r + k, min_covar=1e-4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(xs)
            ll = m.score(xs)
            if np.isfinite(ll) and ll > cand_ll:
                cand, cand_ll = m, ll
        if cand is None:
            continue
        bic = cand.bic(xs)
        if best is None or bic < best.bic:
            best = FittedHMM(cand, mean, std, k, bic)
    if best is None:
        raise RuntimeError("HMM fit failed for every candidate number of states")
    return best


def emission_logprob(fitted: FittedHMM, xs: np.ndarray) -> np.ndarray:
    covs = fitted.covariances()
    return np.column_stack([
        multivariate_normal(mean=fitted.model.means_[k], cov=covs[k], allow_singular=True).logpdf(xs)
        for k in range(fitted.n_states)
    ]).reshape(len(xs), fitted.n_states)


def forward_filter(fitted: FittedHMM, obs: pd.DataFrame) -> pd.DataFrame:
    """Filtered probabilities P(state_t | x_1..x_t) for every date of ``obs`` (causal)."""
    xs = fitted.scale(obs.to_numpy(dtype=float))
    log_e = emission_logprob(fitted, xs)
    log_a = np.log(np.clip(fitted.model.transmat_, 1e-300, None))
    log_alpha = np.log(np.clip(fitted.model.startprob_, 1e-300, None)) + log_e[0]
    log_alpha -= logsumexp(log_alpha)
    out = np.empty_like(log_e)
    out[0] = log_alpha
    for t in range(1, len(xs)):
        log_alpha = logsumexp(log_alpha[:, None] + log_a, axis=0) + log_e[t]
        log_alpha -= logsumexp(log_alpha)
        out[t] = log_alpha
    return pd.DataFrame(np.exp(out), index=obs.index, columns=[f"p{k}" for k in range(fitted.n_states)])


def predict_next(fitted: FittedHMM, filtered: pd.DataFrame) -> pd.DataFrame:
    """One-step-ahead state probabilities P(state_{t+1} | x_1..x_t)."""
    return pd.DataFrame(filtered.to_numpy() @ fitted.model.transmat_, index=filtered.index, columns=filtered.columns)


def state_scores(fitted: FittedHMM, train_obs: pd.DataFrame, train_ret: pd.Series, method: str = "vol_rank") -> np.ndarray:
    """Exposure score in [0, 1] of each state, computed on the train window only."""
    k = fitted.n_states
    if k == 1:
        return np.ones(1)
    if method == "vol_rank":
        # Mean log-vol of each state in original units.
        vol_col = list(train_obs.columns).index("log_vol")
        key = fitted.model.means_[:, vol_col] * fitted.std[vol_col] + fitted.mean[vol_col]
        order = np.argsort(key)                # calm first
    elif method == "sharpe_rank":
        probs = forward_filter(fitted, train_obs)
        state = probs.to_numpy().argmax(axis=1)
        r = train_ret.reindex(train_obs.index).to_numpy()
        sharpe = np.array([np.nanmean(r[state == s]) / (np.nanstd(r[state == s]) + 1e-12) if (state == s).any() else -np.inf
                           for s in range(k)])
        order = np.argsort(-sharpe)            # best first
    else:
        raise ValueError(f"Unknown state_score method '{method}'")
    scores = np.empty(k)
    scores[order] = np.linspace(1.0, 0.0, k)
    return scores
