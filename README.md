# Algo_strat

Recherche, backtest et tuning de deux stratégies de trading swing en barres daily, destinées à des comptes prop firm (FTMO 2-Step Swing par défaut, FTMO 1-Step, FundedNext Stellar 2-Step) :

- **`strat_tsmom_dl/`** — Time Series Momentum multi-actifs, baselines classiques et Deep Momentum Network (LSTM entraîné sur le Sharpe).
- **`strat_gold_f2f/`** — trend-momentum sur l'or, inspiré de « Forecast-to-Fill » (arXiv 2511.08571), avec meta-labeling.

Les deux stratégies partagent un filtre de régime (HMM + ensemble), un moteur de règles prop firm, un modèle de coûts et un moteur de backtest walk-forward (`common/`). Les signaux sont calculés après la clôture daily et validés manuellement une fois par jour ; aucune surveillance intraday n'est requise.

La documentation complète se trouve dans [`docs/GUIDE.md`](docs/GUIDE.md) et les sources dans [`docs/sources/references.md`](docs/sources/references.md).

## Installation (Windows natif, sans Docker)

Prérequis : [uv](https://docs.astral.sh/uv/) et git. uv installe lui-même Python 3.12 (version figée dans `.python-version`).

```powershell
uv sync                 # crée .venv à partir de uv.lock
uv sync --extra mt5     # en plus : package MetaTrader5 (Windows uniquement), pour l'export MT5
```

La reproductibilité repose sur `uv.lock`, `.python-version`, les seeds fixées et le fichier `results/<run_id>/meta.json` écrit par chaque run (hash de commit, configuration, seeds, versions des packages).

## Commandes

Toutes les étapes passent par la CLI `algo` (remplace le Makefile) :

| Commande | Rôle |
|---|---|
| `uv run algo data [--symbols EURUSD,XAUUSD] [--source auto]` | télécharge, nettoie et met en cache les données, écrit le rapport qualité |
| `uv run algo test` | lance la suite de tests |
| `uv run algo backtest --strat tsmom_dl` | backtest walk-forward (avec et sans filtre de régime) |
| `uv run algo tune --strat gold_f2f` | tuning Optuna avec budget d'essais fixe |
| `uv run algo report --strat tsmom_dl` | rapport HTML/CSV et mise à jour de `results/SUMMARY.md` |
| `uv run algo holdout --strat gold_f2f --config <yaml> --i-confirm` | évaluation unique sur le holdout verrouillé (2024 →) |

Stratégies disponibles : `tsmom_dl`, `gold_f2f`.

## Arborescence

```
common/          code partagé : data, costs, propfirm, regime, backtest, tuning, reporting, cli.py
strat_tsmom_dl/  stratégie 1 (features, modèle, signal)
strat_gold_f2f/  stratégie 4 (features, modèle, signal)
configs/         YAML : univers, coûts, règles prop, stratégies
docs/            GUIDE.md et sources
tests/           tests unitaires et anti-biais
```

Les dossiers `data/` et `results/raw/` sont générés localement et ne sont pas versionnés.
