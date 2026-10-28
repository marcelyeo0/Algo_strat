"""Dukascopy historical data feed: hourly bid/ask candles re-aggregated to server-close daily bars.

Monthly files ``<instrument>/<year>/<month-1>/<SIDE>_candles_hour_1.bi5`` are LZMA-compressed
arrays of 24-byte big-endian records: seconds from the start of the month (int32), open,
close, low, high (int32, in units of the instrument's point) and volume (float32).

The feed is rate limited. The client throttles requests, honours ``Retry-After`` on HTTP 429
and raises :class:`DukascopyUnavailable` when the server keeps refusing, so that the pipeline
can fall back to another source.
"""

from __future__ import annotations

import logging
import lzma
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common.data.calendar import aggregate_to_sessions, session_dates

log = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">5if")


class DukascopyUnavailable(RuntimeError):
    """Raised when the feed is unreachable or keeps rate limiting us."""


def decode_hour_candles(raw: bytes, month_start: pd.Timestamp, point: float) -> pd.DataFrame:
    """Decode an (already decompressed) hourly candle file into a UTC-indexed OHLCV frame."""
    if len(raw) % RECORD.size:
        raise ValueError("Corrupted bi5 payload: size is not a multiple of the record size.")
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    arr = np.frombuffer(raw, dtype=np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                                             ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    index = pd.DatetimeIndex(month_start.tz_localize("UTC") + pd.to_timedelta(arr["t"].astype(np.int64), unit="s"))
    df = pd.DataFrame(
        {
            "open": arr["o"].astype(float) * point,
            "high": arr["h"].astype(float) * point,
            "low": arr["l"].astype(float) * point,
            "close": arr["c"].astype(float) * point,
            "volume": arr["v"].astype(float),
        },
        index=index,
    )
    # Dukascopy writes flat zero-volume candles for closed-market hours.
    df = df[df["volume"] > 0]
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.mean() > 0.01:
        raise ValueError("Decoded candles are inconsistent (high/low), check the record layout or point.")
    return df


class DukascopyClient:
    def __init__(self, cache_dir: Path, min_interval_s: float = 1.0, max_retries: int = 4, timeout_s: float = 30.0,
                 offline: bool = False):
        self.cache_dir = cache_dir
        self.offline = offline
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._last_request = 0.0
        self.session = requests.Session()

    def _get(self, url: str) -> bytes:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout_s)
            except requests.RequestException as exc:
                raise DukascopyUnavailable(str(exc)) from exc
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                return b""
            if resp.status_code in (429, 503) and attempt < self.max_retries:
                delay = float(resp.headers.get("Retry-After", 2 ** (attempt + 2)))
                log.warning("Dukascopy HTTP %s, retrying in %.0fs", resp.status_code, delay)
                time.sleep(delay)
                continue
            raise DukascopyUnavailable(f"HTTP {resp.status_code} for {url}")
        raise DukascopyUnavailable(f"Too many retries for {url}")

    def month_file(self, instrument: str, year: int, month: int, side: str) -> bytes:
        """Return the decompressed hourly file of one month, using the local raw cache."""
        cache = self.cache_dir / instrument / f"{year}-{month:02d}_{side}.bin"
        if cache.exists():
            return cache.read_bytes()
        if self.offline:
            raise DukascopyUnavailable(f"{instrument} {year}-{month:02d} {side} not in the local cache (offline mode)")
        url = f"{BASE_URL}/{instrument}/{year}/{month - 1:02d}/{side}_candles_hour_1.bi5"
        payload = self._get(url)
        raw = lzma.decompress(payload) if payload else b""
        current = pd.Timestamp.now(tz="UTC")
        # Do not cache the current (incomplete) month.
        if (year, month) < (current.year, current.month):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        return raw

    def probe(self, instrument: str) -> None:
        """Cheap availability check; raises DukascopyUnavailable on failure."""
        last = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=2)
        self.month_file(instrument, last.year, last.month, "BID")

    def hourly(self, instrument: str, point: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Hourly mid OHLC plus bid/ask closes between ``start`` and ``end``."""
        frames = []
        for month_start in pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS"):
            sides = {}
            for side in ("BID", "ASK"):
                raw = self.month_file(instrument, month_start.year, month_start.month, side)
                sides[side] = decode_hour_candles(raw, month_start, point)
            bid, ask = sides["BID"], sides["ASK"]
            if bid.empty or ask.empty:
                continue
            idx = bid.index.intersection(ask.index)
            bid, ask = bid.loc[idx], ask.loc[idx]
            mid = (bid[["open", "high", "low", "close"]] + ask[["open", "high", "low", "close"]]) / 2
            mid["bid_close"] = bid["close"]
            mid["ask_close"] = ask["close"]
            mid["spread"] = ask["close"] - bid["close"]
            mid["volume"] = bid["volume"]
            frames.append(mid)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).sort_index()


def load_daily(
    client: DukascopyClient,
    instrument: str,
    point: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    close_tz: str,
    close_hour: int,
) -> pd.DataFrame:
    """Daily session bars (mid OHLC, bid/ask closes, median hourly spread)."""
    hourly = client.hourly(instrument, point, start, end)
    if hourly.empty:
        return hourly
    daily = aggregate_to_sessions(hourly.drop(columns=["spread"]), close_tz, close_hour)
    keys = session_dates(pd.DatetimeIndex(hourly.index), close_tz, close_hour)
    spread_med = hourly["spread"].groupby(keys).median()
    daily["spread"] = spread_med.reindex(daily.index)
    return daily


def slow_download(instruments: list[str], start: pd.Timestamp, cache_dir: Path, min_interval_s: float = 5.0,
                  max_retries: int = 6) -> dict[str, str]:
    """Download every monthly bid/ask file not yet cached, slowly, and stop cleanly when refused.

    Progress is the local cache itself: re-running the command resumes where it stopped.
    Returns ``{instrument: "complete" | "stopped at YYYY-MM"}``.
    """
    client = DukascopyClient(cache_dir, min_interval_s=min_interval_s, max_retries=max_retries)
    now = pd.Timestamp.now()
    months = pd.date_range(start.to_period("M").to_timestamp(), now, freq="MS")
    status: dict[str, str] = {}
    for instrument in instruments:
        status[instrument] = "complete"
        for m in months:
            try:
                for side in ("BID", "ASK"):
                    client.month_file(instrument, m.year, m.month, side)
            except DukascopyUnavailable as exc:
                status[instrument] = f"stopped at {m:%Y-%m} ({exc})"
                log.warning("%s: %s - re-run the command later to resume.", instrument, status[instrument])
                return {**status, **{i: "not started" for i in instruments if i not in status}}
        log.info("%s: all months cached", instrument)
    return status
