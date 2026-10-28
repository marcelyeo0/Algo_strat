"""Data quality checks and per-instrument report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    symbol: str
    source: str
    first_date: str
    last_date: str
    n_bars: int
    missing_weekdays: int
    longest_gap_days: int
    duplicates: int
    non_positive_prices: int
    ohlc_inconsistent: int
    stale_runs: int
    outliers: int
    outliers_near_roll: int
    roll_sessions: int
    calendar_coverage: float = float("nan")
    outlier_dates: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("outlier_dates")
        return row


def check_bars(
    bars: pd.DataFrame,
    symbol: str,
    source: str,
    k_sigma: float = 6.0,
    vol_window: int = 63,
    stale_run: int = 5,
    raw: pd.DataFrame | None = None,
) -> QualityReport:
    """Run the quality checks on a standard bars frame.

    ``raw`` (the source frame before cleaning) is used to count duplicates. Outliers are
    returns larger than ``k_sigma`` times the trailing volatility; they are only *flagged*.
    """
    idx = bars.index
    weekdays = pd.bdate_range(idx[0], idx[-1]) if len(idx) else pd.DatetimeIndex([])
    missing = weekdays.difference(idx)
    gaps = pd.Series(idx[1:] - idx[:-1]).dt.days if len(idx) > 1 else pd.Series(dtype=float)
    src = raw if raw is not None else bars
    duplicates = int(src.index.duplicated().sum())
    non_positive = int((src[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    inconsistent = int(
        ((src["high"] < src[["open", "close"]].max(axis=1) * (1 - 1e-9))
         | (src["low"] > src[["open", "close"]].min(axis=1) * (1 + 1e-9))).sum()
    )
    same = bars["raw_close"].diff().eq(0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).sum()
    stale = int((run_len >= stale_run).sum())

    logret = np.log1p(bars["ret"])
    sigma = logret.rolling(vol_window, min_periods=20).std().shift(1)
    jumps = logret.abs() > k_sigma * sigma
    roll_near = bars["is_roll"].rolling(7, center=True, min_periods=1).max().astype(bool)
    outlier_dates = [d.strftime("%Y-%m-%d") for d in idx[jumps.fillna(False).to_numpy()]]

    return QualityReport(
        symbol=symbol,
        source=source,
        first_date=idx[0].strftime("%Y-%m-%d") if len(idx) else "",
        last_date=idx[-1].strftime("%Y-%m-%d") if len(idx) else "",
        n_bars=len(idx),
        missing_weekdays=len(missing),
        longest_gap_days=int(gaps.max()) if len(gaps) else 0,
        duplicates=duplicates,
        non_positive_prices=non_positive,
        ohlc_inconsistent=inconsistent,
        stale_runs=stale,
        outliers=int(jumps.sum()),
        outliers_near_roll=int((jumps & roll_near).sum()),
        roll_sessions=int(bars["is_roll"].sum()),
        outlier_dates=outlier_dates,
    )


def calendar_coverage(all_bars: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Share of the weekdays of the common period on which each instrument has a bar."""
    if not all_bars:
        return {}
    start = max(b.index[0] for b in all_bars.values())
    end = min(b.index[-1] for b in all_bars.values())
    days = pd.bdate_range(start, end)
    return {s: float(b.index.isin(days).sum() / max(len(days), 1)) for s, b in all_bars.items()}


def render_markdown(rep: QualityReport) -> str:
    lines = [
        f"# Rapport qualité — {rep.symbol}",
        "",
        f"- Source : `{rep.source}`",
        f"- Période : {rep.first_date} → {rep.last_date} ({rep.n_bars} barres)",
        f"- Jours ouvrés manquants : {rep.missing_weekdays} (plus long trou : {rep.longest_gap_days} jours calendaires)",
        f"- Doublons dans la source : {rep.duplicates} (dernier conservé)",
        f"- Prix nuls ou négatifs dans la source : {rep.non_positive_prices} (barres supprimées)",
        f"- Barres OHLC incohérentes dans la source : {rep.ohlc_inconsistent} (high/low élargis pour encadrer open et close)",
        f"- Séquences de clôtures figées (≥ 5 jours) : {rep.stale_runs}",
        f"- Sessions de roll neutralisées (calendrier d'échéance) : {rep.roll_sessions}",
        f"- Rendements > k·σ signalés : {rep.outliers}, dont {rep.outliers_near_roll} à moins de 3 séances d'un roll",
        f"- Couverture du calendrier commun : {rep.calendar_coverage:.1%}",
        "",
        "Les rendements signalés ne sont pas supprimés : ils sont listés pour contrôle manuel.",
    ]
    if rep.outlier_dates:
        lines += ["", "Dates signalées : " + ", ".join(rep.outlier_dates[:100])]
    return "\n".join(lines) + "\n"
