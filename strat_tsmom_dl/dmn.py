"""Deep Momentum Network: LSTM trained directly on the (negative) Sharpe ratio.

Following Lim, Zohren & Roberts (2019): the network outputs a position X_t in (-1, 1) for each
asset and date, the strategy return is X_t * sigma_target / sigma_t * r_{t+1}, and the loss is
the negative annualised Sharpe ratio of these returns over the batch (pooled over assets and
time steps). A turnover penalty subtracts the per-asset cost (bps from the cost profile) times
the change in the volatility-scaled position, so that trading costs are part of the objective.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from strat_tsmom_dl.features import ANNUAL


class LSTMPolicy(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return torch.tanh(self.head(self.dropout(h))).squeeze(-1)


def sharpe_loss(pos: torch.Tensor, scale: torch.Tensor, next_ret: torch.Tensor, cost: torch.Tensor,
                penalty: float) -> torch.Tensor:
    """Negative annualised Sharpe of net strategy returns. Shapes: (batch, time); cost: (batch,)."""
    w = pos * scale
    gross = w * next_ret
    turnover = torch.zeros_like(w)
    turnover[:, 1:] = (w[:, 1:] - w[:, :-1]).abs()
    net = gross - penalty * cost[:, None] * turnover
    return -net.mean() / (net.std() + 1e-9) * np.sqrt(ANNUAL)


@dataclass
class SequenceData:
    x: np.ndarray          # (n, L, F)
    scale: np.ndarray      # (n, L) sigma_target / sigma_t
    next_ret: np.ndarray   # (n, L)
    cost: np.ndarray       # (n,)


def build_sequences(frames: dict[str, pd.DataFrame], cols: list[str], length: int, costs: dict[str, float]) -> SequenceData:
    """Non-overlapping windows of ``length`` sessions per asset, aligned on the most recent date."""
    xs, ss, rs, cs = [], [], [], []
    for sym, df in frames.items():
        n = len(df) // length
        if n == 0:
            continue
        d = df.iloc[len(df) - n * length:]
        xs.append(d[cols].to_numpy(np.float32).reshape(n, length, len(cols)))
        ss.append(d["scale"].to_numpy(np.float32).reshape(n, length))
        rs.append(d["next_ret"].to_numpy(np.float32).reshape(n, length))
        cs.append(np.full(n, costs.get(sym, 0.0), np.float32))
    if not xs:
        raise ValueError("Not enough data to build training sequences")
    return SequenceData(np.concatenate(xs), np.concatenate(ss), np.concatenate(rs), np.concatenate(cs))


def train_dmn(train: SequenceData, valid: SequenceData, hidden: int, dropout: float, lr: float, batch_size: int,
              max_grad_norm: float, max_epochs: int, patience: int, penalty: float, seed: int) -> tuple[LSTMPolicy, dict]:
    """Train with Adam and early stopping on the validation loss; returns the best model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LSTMPolicy(train.x.shape[-1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = {k: torch.from_numpy(getattr(train, k)) for k in ("x", "scale", "next_ret", "cost")}
    v = {k: torch.from_numpy(getattr(valid, k)) for k in ("x", "scale", "next_ret", "cost")}
    best_loss, best_state, wait, history = np.inf, None, 0, []
    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(len(train.x))
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            opt.zero_grad()
            loss = sharpe_loss(model(t["x"][b]), t["scale"][b], t["next_ret"][b], t["cost"][b], penalty)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(sharpe_loss(model(v["x"]), v["scale"], v["next_ret"], v["cost"], penalty))
        history.append(vloss)
        if vloss < best_loss - 1e-4:
            best_loss, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_valid_loss": best_loss, "epochs": len(history)}


def predict_positions(model: LSTMPolicy, df: pd.DataFrame, cols: list[str], length: int, dates: pd.DatetimeIndex) -> pd.Series:
    """Position X_t for each date in ``dates`` from the window of the last ``length`` sessions."""
    arr = df[cols].to_numpy(np.float32)
    loc = df.index.get_indexer(dates)
    keep = loc >= length - 1
    windows = np.stack([arr[i - length + 1:i + 1] for i in loc[keep]]) if keep.any() else np.empty((0, length, len(cols)), np.float32)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(windows))[:, -1].numpy() if len(windows) else np.empty(0)
    return pd.Series(out, index=dates[keep])
