"""Volatility targeting, confidence shaping and impact-adjusted fractional Kelly (arXiv 2511.08571).

Growth with linear cost k and impact gamma at turnover n:
    g(f) ~ mu f - 1/2 sigma^2 f^2 - n k f - gamma (n f)^{3/2}
With x = sqrt(f), the first-order condition is 2 sigma^2 x^2 + 3 gamma n^{3/2} x - 2 (mu - n k) = 0:
    x* = (-3 gamma n^{3/2} + sqrt(9 gamma^2 n^3 + 16 sigma^2 (mu - n k))) / (4 sigma^2),  f* = x*^2
f* = 0 when mu <= n k. The fractional Kelly is lambda_Kelly f*.
"""

from __future__ import annotations

import numpy as np


def impact_adjusted_kelly(mu: float, sigma: float, n: float, k: float, gamma: float) -> float:
    if sigma <= 0 or mu <= n * k:
        return 0.0
    a = 3 * gamma * n ** 1.5
    disc = 9 * gamma ** 2 * n ** 3 + 16 * sigma ** 2 * (mu - n * k)
    x = (-a + np.sqrt(disc)) / (4 * sigma ** 2)
    return float(max(x, 0.0) ** 2)


def position_weight(p_bull: float, sigma_hat: float, kelly: float, cfg: dict) -> float:
    """w_t = f~ * w_conf, capped at W_max (long-only)."""
    sigma_star = float(cfg["target_vol"]) / np.sqrt(252)
    w_max = float(cfg["w_max"])
    w_vol = min(w_max, sigma_star / sigma_hat) if sigma_hat > 0 else 0.0
    w_conf = w_vol * max(p_bull - 0.5, 0.0) / 0.5
    f_tilde = float(cfg["kelly_fraction"]) * kelly
    if f_tilde < 1e-6:
        return min(w_max, float(cfg["tiny_kelly_baseline"]) * w_vol)
    return min(w_max, f_tilde * w_conf)
