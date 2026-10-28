"""Export rates and symbol specifications from a MetaTrader 5 terminal (FTMO demo account).

Runs on the Windows host, inside the same uv environment, with the optional ``mt5`` extra:

    uv sync --extra mt5
    uv run python -m common.data.mt5_export --out mt5_exports

The terminal must be running and logged in. No credentials are read or stored by this script:
it attaches to the terminal session that is already open.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.utils.config import load_universe
from common.utils.paths import repo_root

SPEC_FIELDS = (
    "name", "digits", "point", "spread", "trade_contract_size", "trade_tick_value", "trade_tick_size",
    "volume_min", "volume_step", "volume_max", "swap_mode", "swap_long", "swap_short",
    "swap_rollover3days", "currency_base", "currency_profit", "currency_margin", "margin_initial",
)


def export(out_dir: Path, start: datetime, timeframes: tuple[str, ...] = ("D1", "H1")) -> None:
    import MetaTrader5 as mt5  # noqa: N813 - official package name

    if not mt5.initialize():
        raise RuntimeError(f"MetaTrader5 initialize() failed: {mt5.last_error()}")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = []
        end = datetime.now(UTC)
        for symbol, spec in load_universe(include_auxiliary=True).items():
            name = spec.get("mt5")
            info = mt5.symbol_info(name) if name else None
            if info is None:
                print(f"[skip] {symbol}: symbol '{name}' not found on this server")
                continue
            mt5.symbol_select(name, True)
            specs.append({"symbol": symbol, **{f: getattr(info, f, None) for f in SPEC_FIELDS}})
            for tf in timeframes:
                rates = mt5.copy_rates_range(name, getattr(mt5, f"TIMEFRAME_{tf}"), start, end)
                if rates is None or len(rates) == 0:
                    print(f"[warn] {symbol} {tf}: no rates")
                    continue
                df = pd.DataFrame(rates)
                # MT5 times are expressed in server time (seconds since epoch, server clock).
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.to_csv(out_dir / f"{symbol}_{tf}.csv", index=False, lineterminator="\n")
                print(f"[ok] {symbol} {tf}: {len(df)} bars")
        pd.DataFrame(specs).to_csv(out_dir / "symbol_specs.csv", index=False, lineterminator="\n")
    finally:
        mt5.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=repo_root() / "mt5_exports")
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()
    export(args.out, datetime.fromisoformat(args.start).replace(tzinfo=UTC))


if __name__ == "__main__":
    main()
