"""Command-line entry point: ``uv run algo <command>`` (replaces a Makefile on Windows)."""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated

import typer

from common.utils.paths import repo_root

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Algo_strat research pipeline.")
costs_app = typer.Typer(no_args_is_help=True, help="Cost profiles.")
app.add_typer(costs_app, name="costs")

STRATEGIES = ("tsmom_dl", "gold_f2f")

StratOpt = Annotated[str, typer.Option("--strat", "-s", help=f"Strategy: one of {', '.join(STRATEGIES)}.")]
CostProfileOpt = Annotated[
    str | None, typer.Option("--cost-profile", help="estimates | mt5 (default: cost_profile in configs/settings.yaml).")
]
ExecOpt = Annotated[str, typer.Option("--execution", help="next_open | close_slippage | both")]


def _check_strat(strat: str) -> str:
    if strat not in STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy '{strat}'. Expected one of {STRATEGIES}.")
    return strat


def _profile(override: str | None) -> str:
    from common.utils.settings import cost_profile, profile_banner

    profile = cost_profile(override)
    typer.echo(profile_banner(profile))
    return profile


@app.command()
def data(
    symbols: Annotated[str | None, typer.Option(help="Comma-separated subset of the universe.")] = None,
    source: Annotated[str, typer.Option(help="auto | dukascopy | yfinance | mt5")] = "auto",
    start: Annotated[str | None, typer.Option(help="First date (YYYY-MM-DD).")] = None,
    refresh: Annotated[bool, typer.Option(help="Ignore the Parquet cache and re-download.")] = False,
) -> None:
    """Download, clean and cache daily bars, then write the quality report."""
    from common.data.pipeline import run_data_pipeline

    run_data_pipeline(symbols=symbols, source=source, start=start, refresh=refresh)


@app.command("dukascopy-download")
def dukascopy_download(
    symbols: Annotated[str | None, typer.Option(help="Comma-separated subset of the universe.")] = None,
    start: Annotated[str, typer.Option(help="First month (YYYY-MM-DD).")] = "2003-01-01",
    interval: Annotated[float, typer.Option(help="Seconds between two requests.")] = 5.0,
) -> None:
    """Slow, resumable download of the Dukascopy hourly files (re-run to resume)."""
    import logging

    import pandas as pd

    from common.data.dukascopy import slow_download
    from common.utils.config import load_universe
    from common.utils.paths import data_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    universe = load_universe()
    wanted = [s.strip() for s in symbols.split(",")] if symbols else list(universe)
    instruments = [universe[s]["dukascopy"]["instrument"] for s in wanted]
    status = slow_download(instruments, pd.Timestamp(start), data_dir() / "raw" / "dukascopy", min_interval_s=interval)
    for inst, st in status.items():
        typer.echo(f"{inst}: {st}")


@app.command("data-compare")
def data_compare(
    symbols: Annotated[str | None, typer.Option(help="Comma-separated subset of the universe.")] = None,
) -> None:
    """Compare cached Dukascopy data with yfinance on the common period."""
    from common.data.compare import run_compare

    out = run_compare([s.strip() for s in symbols.split(",")] if symbols else None)
    typer.echo(out.to_string(index=False))


@app.command()
def test(
    args: Annotated[list[str] | None, typer.Argument(help="Extra arguments passed to pytest.")] = None,
) -> None:
    """Run the test suite."""
    code = subprocess.call([sys.executable, "-m", "pytest", *(args or [])], cwd=repo_root())
    raise typer.Exit(code)


@app.command()
def backtest(
    strat: StratOpt,
    config: Annotated[str | None, typer.Option(help="Strategy config YAML (default: configs/strategies/<strat>.yaml).")] = None,
    preset: Annotated[str, typer.Option(help="Prop-firm preset in configs/propfirm/.")] = "ftmo_2step_swing",
    period: Annotated[str, typer.Option(help="dev | validation (the holdout has its own command).")] = "validation",
    cost_profile: CostProfileOpt = None,
    execution: ExecOpt = "both",
) -> None:
    """Run a walk-forward backtest with and without the regime filter."""
    from common.backtest.runner import run_backtest

    run_backtest(strat=_check_strat(strat), config=config, preset=preset, period=period,
                 cost_profile=_profile(cost_profile), execution=execution)


@app.command()
def tune(
    strat: StratOpt,
    config: Annotated[str | None, typer.Option(help="Strategy config YAML.")] = None,
    preset: Annotated[str, typer.Option(help="Prop-firm preset.")] = "ftmo_2step_swing",
    cost_profile: CostProfileOpt = None,
) -> None:
    """Optuna tuning on the development period, with the trial budget fixed in the config."""
    from common.tuning.runner import run_tuning

    run_tuning(strat=_check_strat(strat), config=config, preset=preset, cost_profile=_profile(cost_profile))


@app.command()
def report(
    strat: StratOpt,
    run_id: Annotated[str | None, typer.Option(help="Run to report on (default: latest run of the strategy).")] = None,
) -> None:
    """Build the HTML/CSV report of a run and refresh results/SUMMARY.md."""
    from common.reporting.runner import run_report

    run_report(strat=_check_strat(strat), run_id=run_id)


@app.command()
def holdout(
    strat: StratOpt,
    config: Annotated[str, typer.Option(help="FINAL strategy config YAML (frozen).")],
    preset: Annotated[str, typer.Option(help="Prop-firm preset.")] = "ftmo_2step_swing",
    cost_profile: CostProfileOpt = None,
    i_confirm: Annotated[bool, typer.Option("--i-confirm", help="Explicit confirmation: runs once per config.")] = False,
) -> None:
    """Evaluate a frozen configuration on the locked holdout (2024-01-01 onwards). Runs once per config."""
    from common.utils.settings import holdout_allowed

    profile = _profile(cost_profile)
    if not holdout_allowed(profile):
        typer.echo(f"Refused: the holdout is only played once, with real costs (profile '{profile}' is not allowed).")
        raise typer.Exit(2)
    if not i_confirm:
        typer.echo("The holdout is locked. Re-run with --i-confirm once the configuration is final.")
        raise typer.Exit(1)
    from common.backtest.holdout import run_holdout

    run_holdout(strat=_check_strat(strat), config=config, preset=preset, cost_profile=profile)


@costs_app.command("compare")
def costs_compare(
    strat: Annotated[str | None, typer.Option("--strat", "-s", help="Also report the validation Sharpe impact.")] = None,
) -> None:
    """Compare the estimates profile with the MT5 profile, per instrument."""
    from common.costs.compare import run_costs_compare

    run_costs_compare(strat=_check_strat(strat) if strat else None)


if __name__ == "__main__":
    app()
