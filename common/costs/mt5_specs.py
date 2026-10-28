"""Reconcile the cost configuration with the symbol specifications exported from MT5."""

from __future__ import annotations

import pandas as pd

from common.costs.model import DAYS_PER_YEAR, CostModel

# MT5 SYMBOL_SWAP_MODE values used by FTMO (see the MetaTrader 5 documentation).
SWAP_MODE_POINTS = 1
SWAP_MODE_INTEREST_CURRENT = 5


def swap_to_annual_pct(swap: float, mode: int, point: float, price: float) -> float | None:
    """Convert an MT5 swap value into % of notional per year (``None`` for unsupported modes)."""
    if mode == SWAP_MODE_POINTS:
        return swap * point / price * DAYS_PER_YEAR * 100.0
    if mode == SWAP_MODE_INTEREST_CURRENT:
        return swap
    return None


def apply_mt5_specs(model: CostModel, specs: pd.DataFrame, prices: dict[str, float]) -> CostModel:
    """Return a cost model whose contract sizes, spreads, swaps and triple-swap days come from MT5."""
    for symbol, row in specs.iterrows():
        if symbol not in model.instruments:
            continue
        changes = {
            "contract_size": float(row["trade_contract_size"]),
            "spread": float(row["spread"]) * float(row["point"]),
            "triple_swap_day": (int(row["swap_rollover3days"]) - 1) % 7,  # MT5: 0 = Sunday
        }
        price = prices.get(symbol)
        if price:
            for side in ("long", "short"):
                pct = swap_to_annual_pct(float(row[f"swap_{side}"]), int(row["swap_mode"]), float(row["point"]), price)
                if pct is not None:
                    changes[f"swap_{side}"] = pct
        model = model.with_instrument(symbol, **changes)
    return model
