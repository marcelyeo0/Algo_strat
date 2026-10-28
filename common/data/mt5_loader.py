"""Loader for the CSV files produced by ``common.data.mt5_export``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.utils.paths import repo_root


def default_dir() -> Path:
    return repo_root() / "mt5_exports"


def load_rates(symbol: str, timeframe: str = "D1", directory: Path | None = None) -> pd.DataFrame:
    """MT5 rates as an OHLC frame.

    D1 bars from the FTMO server already close at 00:00 server time, so their date is the
    session date. The MT5 ``spread`` column is in points; it is converted to price units when
    the symbol specs are available.
    """
    directory = directory or default_dir()
    path = directory / f"{symbol}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
    df.index.name = "date"
    if timeframe == "D1":
        df.index = df.index.normalize()
    specs = load_specs(directory)
    if specs is not None and symbol in specs.index and "spread" in df:
        df["spread"] = df["spread"] * float(specs.loc[symbol, "point"])
    return df


def load_specs(directory: Path | None = None) -> pd.DataFrame | None:
    """Symbol specifications (contract size, swaps, triple-swap day, typical spread...)."""
    path = (directory or default_dir()) / "symbol_specs.csv"
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("symbol")


def available(symbol: str, timeframe: str = "D1", directory: Path | None = None) -> bool:
    return ((directory or default_dir()) / f"{symbol}_{timeframe}.csv").exists()
