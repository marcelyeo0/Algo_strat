"""Command-line entry point: ``uv run algo <command>`` (replaces a Makefile on Windows)."""

from __future__ import annotations

import subprocess
import sys
from typing import Annotated

import typer

from common.utils.paths import repo_root

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Algo_strat research pipeline.")

STRATEGIES = ("tsmom_dl", "gold_f2f")

StratOpt = Annotated[str, typer.Option("--strat", "-s", help=f"Strategy: one of {', '.join(STRATEGIES)}.")]


def _check_strat(strat: str) -> str:
    if strat not in STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy '{strat}'. Expected one of {STRATEGIES}.")
    return strat


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
) -> None:
    """Run a walk-forward backtest with and without the regime filter."""
    from common.backtest.runner import run_backtest

    run_backtest(strat=_check_strat(strat), config=config, preset=preset, period=period)


@app.command()
def tune(
    strat: StratOpt,
    config: Annotated[str | None, typer.Option(help="Strategy config YAML.")] = None,
    preset: Annotated[str, typer.Option(help="Prop-firm preset.")] = "ftmo_2step_swing",
) -> None:
    """Optuna tuning on the development period, with the trial budget fixed in the config."""
    from common.tuning.runner import run_tuning

    run_tuning(strat=_check_strat(strat), config=config, preset=preset)


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
    i_confirm: Annotated[bool, typer.Option("--i-confirm", help="Explicit confirmation: runs once per config.")] = False,
) -> None:
    """Evaluate a frozen configuration on the locked holdout (2024-01-01 onwards). Runs once per config."""
    if not i_confirm:
        typer.echo("The holdout is locked. Re-run with --i-confirm once the configuration is final.")
        raise typer.Exit(1)
    from common.backtest.holdout import run_holdout

    run_holdout(strat=_check_strat(strat), config=config, preset=preset)


if __name__ == "__main__":
    app()
