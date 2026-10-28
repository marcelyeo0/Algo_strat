"""Comparison of Dukascopy and yfinance daily bars on their common period."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.data import dukascopy
from common.data.bars import finalize_bars
from common.data.pipeline import load_bars
from common.utils.config import load_universe, load_yaml
from common.utils.paths import data_dir


def compare_series(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Return correlation, price gaps and missing days between two standard bars frames."""
    lo, hi = max(a.index[0], b.index[0]), min(a.index[-1], b.index[-1])
    a, b = a.loc[lo:hi], b.loc[lo:hi]
    common = a.index.intersection(b.index)
    ra, rb = a["ret"].reindex(common), b["ret"].reindex(common)
    ok = ra.notna() & rb.notna()
    gap = (a["raw_close"].reindex(common) / b["raw_close"].reindex(common) - 1).abs()
    return {
        "start": lo.strftime("%Y-%m-%d"),
        "end": hi.strftime("%Y-%m-%d"),
        "common_days": len(common),
        "only_in_a": len(a.index.difference(b.index)),
        "only_in_b": len(b.index.difference(a.index)),
        "ret_corr": float(np.corrcoef(ra[ok], rb[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
        "close_gap_mean_pct": float(gap.mean() * 100),
        "close_gap_p99_pct": float(gap.quantile(0.99) * 100),
        "close_gap_max_pct": float(gap.max() * 100),
    }


def run_compare(symbols: list[str] | None = None) -> pd.DataFrame:
    """Build Dukascopy bars from the local cache only (no network) and compare them with yfinance."""
    universe = load_universe()
    session = load_yaml("universe.yaml")["session"]
    client = dukascopy.DukascopyClient(data_dir() / "raw" / "dukascopy", offline=True)
    rows = []
    for symbol in symbols or list(universe):
        spec = universe[symbol]["dukascopy"]
        folder = client.cache_dir / spec["instrument"]
        files = sorted(folder.glob("*_BID.bin")) if folder.exists() else []
        if not files:
            rows.append({"symbol": symbol, "status": "no Dukascopy cache"})
            continue
        first = pd.Timestamp(files[0].stem.split("_")[0] + "-01")
        last = pd.Timestamp(files[-1].stem.split("_")[0] + "-01") + pd.offsets.MonthEnd(0)
        try:
            raw = dukascopy.load_daily(client, spec["instrument"], float(spec["point"]), first, last,
                                       session["close_tz"], int(session["close_hour"]))
        except dukascopy.DukascopyUnavailable as exc:
            rows.append({"symbol": symbol, "status": f"incomplete cache ({exc})"})
            continue
        duka = finalize_bars(raw)
        yf = load_bars(symbol)
        rows.append({"symbol": symbol, "status": "ok", **compare_series(duka, yf)})
    out = pd.DataFrame(rows)
    qc = data_dir() / "quality"
    qc.mkdir(parents=True, exist_ok=True)
    out.to_csv(qc / "dukascopy_vs_yfinance.csv", index=False, lineterminator="\n")
    lines = [
        "# Comparaison Dukascopy / yfinance",
        "",
        "Période commune par instrument, rendements close-to-close (hors jours de roll neutralisés).",
        "« only_in_a » : jours présents seulement chez Dukascopy ; « only_in_b » : seulement chez yfinance.",
        "Les clôtures Dukascopy sont alignées sur 17:00 New York, celles de yfinance sur la session native :",
        "une corrélation inférieure à 1 et des écarts de prix sont donc attendus, surtout sur le FX.",
        "",
        "```",
        out.to_string(index=False),
        "```",
    ]
    (qc / "dukascopy_vs_yfinance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
