"""Historical central bank policy rates (BIS, data set WS_CBPOL), used by the swap model.

Source: Bank for International Settlements, "Central bank policy rates", public bulk download
https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip (daily series, % per year).
"""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests

from common.utils.paths import data_dir

log = logging.getLogger(__name__)

BIS_URL = "https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip"
# Currency -> BIS reference area (XM = euro area)
AREAS = {"USD": "US", "EUR": "XM", "GBP": "GB", "JPY": "JP", "CHF": "CH", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
         "HKD": "HK"}


def rates_path():
    return data_dir() / "rates" / "policy_rates.parquet"


def download_policy_rates(timeout: float = 300.0) -> pd.DataFrame:
    """Download the BIS bulk file and keep the daily series of the project currencies."""
    resp = requests.get(BIS_URL, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            raw = pd.read_csv(fh, usecols=[3, 4, 5, 6], header=0, names=["freq", "area", "date", "rate"], dtype=str)
    raw = raw[raw["freq"].str.startswith("D")]
    raw["area"] = raw["area"].str.split(":").str[0].str.strip()
    raw = raw[raw["area"].isin(AREAS.values())]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["rate"] = pd.to_numeric(raw["rate"], errors="coerce")
    wide = raw.dropna().pivot_table(index="date", columns="area", values="rate", aggfunc="last").sort_index()
    inverse = {v: k for k, v in AREAS.items()}
    wide = wide.rename(columns=inverse)
    wide = wide.reindex(pd.date_range(wide.index[0], wide.index[-1], freq="D")).ffill()
    path = rates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(path)
    log.info("Policy rates saved: %s, %s -> %s", list(wide.columns), wide.index[0].date(), wide.index[-1].date())
    return wide


def load_policy_rates() -> pd.DataFrame:
    """Daily policy rates in % per year, columns = currencies, forward-filled on calendar days."""
    path = rates_path()
    if not path.exists():
        return download_policy_rates()
    return pd.read_parquet(path)
