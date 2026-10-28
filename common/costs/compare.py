"""``algo costs compare``: gap between the estimates and the MT5 cost profiles."""

from __future__ import annotations

import warnings

import pandas as pd

from common.costs.model import CostModel
from common.utils.config import resolve_config_path
from common.utils.paths import results_dir

FIELDS = ("spread", "commission_per_lot", "swap_long", "swap_short", "swap_markup", "contract_size", "triple_swap_day")


def compare_profiles(estimates: CostModel, mt5: CostModel) -> pd.DataFrame:
    rows = []
    for symbol, est in estimates.instruments.items():
        real = mt5.instruments.get(symbol)
        if real is None:
            continue
        row = {"symbol": symbol}
        for f in FIELDS:
            e, r = getattr(est, f), getattr(real, f)
            row[f"{f}_est"], row[f"{f}_mt5"] = e, r
            if f == "spread" and r:
                row["spread_ratio_est_over_mt5"] = e / r
        rows.append(row)
    return pd.DataFrame(rows)


def run_costs_compare(strat: str | None = None) -> pd.DataFrame:
    if not resolve_config_path("costs/mt5.yaml").exists():
        raise SystemExit("configs/costs/mt5.yaml does not exist yet: export MT5 specs and run "
                         "`uv run python -m common.costs.mt5_specs` first.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est, real = CostModel.from_profile("estimates"), CostModel.from_profile("mt5")
    table = compare_profiles(est, real)
    if strat is not None:
        from common.backtest.runner import validation_sharpe

        table.attrs["validation_sharpe"] = {p: validation_sharpe(strat, cost_profile=p) for p in ("estimates", "mt5")}
        print(f"Validation Sharpe ({strat}): {table.attrs['validation_sharpe']}")
    out = results_dir() / "costs_compare.csv"
    table.to_csv(out, index=False, lineterminator="\n")
    print(table.to_string(index=False))
    return table
