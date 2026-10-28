"""Data pipeline: source selection, cleaning, Parquet cache and quality reports."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common.data import dukascopy, mt5_loader, yahoo
from common.data.bars import finalize_bars
from common.data.quality import calendar_coverage, check_bars, render_markdown
from common.data.rolls import roll_flags
from common.utils.config import load_universe, load_yaml
from common.utils.paths import data_dir

log = logging.getLogger(__name__)


def bars_path(symbol: str) -> Path:
    return data_dir() / "bars" / f"{symbol}.parquet"


def load_bars(symbol: str) -> pd.DataFrame:
    """Read cached standard bars of one instrument (run ``uv run algo data`` first)."""
    path = bars_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No cached bars for {symbol}: run `uv run algo data` first ({path}).")
    return pd.read_parquet(path)


def load_panel(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    symbols = symbols or list(load_universe(include_auxiliary=True))
    return {s: load_bars(s) for s in symbols if bars_path(s).exists()}


def _from_yfinance(spec: dict, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    yf_spec = spec["yfinance"]
    raw = yahoo.download_daily(yf_spec["ticker"], start=start)
    mask = None
    if yf_spec.get("basis") == "futures":
        contract = load_yaml("rolls.yaml")["contracts"][yf_spec["roll_contract"]]
        mask = roll_flags(pd.DatetimeIndex(raw.index.unique().sort_values()), contract)
    return raw, finalize_bars(raw, mask)


def _from_dukascopy(client: dukascopy.DukascopyClient, spec: dict, start: str, session: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = spec["dukascopy"]
    raw = dukascopy.load_daily(
        client, d["instrument"], float(d["point"]), pd.Timestamp(start), pd.Timestamp.now(),
        session["close_tz"], int(session["close_hour"]),
    )
    if raw.empty:
        raise dukascopy.DukascopyUnavailable(f"no Dukascopy data for {d['instrument']}")
    return raw, finalize_bars(raw)


def _from_mt5(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = mt5_loader.load_rates(symbol, "D1")
    return raw, finalize_bars(raw)


def run_data_pipeline(
    symbols: str | None = None, source: str = "auto", start: str | None = None, refresh: bool = False
) -> dict[str, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    universe = load_universe(include_auxiliary=True)
    cfg = load_yaml("universe.yaml")
    session = cfg["session"]
    start = start or cfg.get("default_start", "2000-01-01")
    wanted = [s.strip() for s in symbols.split(",")] if symbols else list(universe)
    unknown = set(wanted) - set(universe)
    if unknown:
        raise ValueError(f"Unknown symbols: {sorted(unknown)}")

    client = dukascopy.DukascopyClient(cache_dir=data_dir() / "raw" / "dukascopy")
    dukascopy_ok = source in ("auto", "dukascopy")
    if dukascopy_ok:
        try:
            client.probe(universe[wanted[0]]["dukascopy"]["instrument"])
        except dukascopy.DukascopyUnavailable as exc:
            if source == "dukascopy":
                raise
            log.warning("Dukascopy unavailable (%s): falling back to yfinance for this run.", exc)
            dukascopy_ok = False

    results: dict[str, pd.DataFrame] = {}
    reports = []
    (data_dir() / "bars").mkdir(parents=True, exist_ok=True)
    qc_dir = data_dir() / "quality"
    qc_dir.mkdir(parents=True, exist_ok=True)

    for symbol in wanted:
        spec = universe[symbol]
        path = bars_path(symbol)
        raw_path = data_dir() / "raw" / "source_bars" / f"{symbol}.parquet"
        if path.exists() and not refresh:
            bars = pd.read_parquet(path)
            raw = pd.read_parquet(raw_path) if raw_path.exists() else None
            results[symbol] = bars
            reports.append(check_bars(bars, symbol, str(bars["source"].iloc[0]), raw=raw))
            continue
        used = None
        raw = bars = None
        if source == "mt5":
            raw, bars = _from_mt5(symbol)
            used = "mt5"
        if bars is None and dukascopy_ok:
            try:
                raw, bars = _from_dukascopy(client, spec, start, session)
                used = "dukascopy"
            except (dukascopy.DukascopyUnavailable, ValueError) as exc:
                log.warning("%s: Dukascopy failed (%s), using yfinance.", symbol, exc)
        if bars is None:
            raw, bars = _from_yfinance(spec, start)
            used = f"yfinance:{spec['yfinance']['ticker']}"
        bars["source"] = used
        bars.to_parquet(path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(raw_path)
        results[symbol] = bars
        reports.append(check_bars(bars, symbol, used, raw=raw))
        log.info("%s: %d bars from %s", symbol, len(bars), used)

    coverage = calendar_coverage(results)
    rows = []
    for rep in reports:
        rep.calendar_coverage = coverage.get(rep.symbol, float("nan"))
        (qc_dir / f"{rep.symbol}.md").write_text(render_markdown(rep), encoding="utf-8")
        rows.append(rep.to_row())
    summary = pd.DataFrame(rows)
    summary.to_csv(qc_dir / "summary.csv", index=False, lineterminator="\n")
    print(summary[["symbol", "source", "first_date", "last_date", "n_bars", "missing_weekdays",
                   "ohlc_inconsistent", "outliers", "roll_sessions"]].to_string(index=False))
    return results
