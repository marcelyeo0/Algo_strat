# Synthèse des résultats

> **Profil de coûts : estimates — Résultats provisoires — coûts estimés**

Ce fichier est régénéré par `uv run algo report` à chaque run final. Aucun run final n'a encore été produit : le tuning (Phase 6) n'a pas commencé.

## Statut des preuves

| Stratégie | Holdout 2024-01-01 → | Preuve valable |
|---|---|---|
| `tsmom_dl` | verrouillé, jamais exécuté | holdout 2024+ (une seule exécution, avec le profil de coûts `mt5`) |
| `gold_f2f` | **contaminé** : deux diagnostics agrégés de la réplication du papier ont été vus sur 2015–10/2025 (voir `docs/GUIDE.md`, hypothèse 22) | uniquement un forward test en démo sur des données nouvelles, à partir de novembre 2026 |

## Nombre d'essais déjà consommés (comptés dans le DSR)

Il s'agit d'évaluations exploratoires sur la période de développement, avant le tuning :

- `gold_f2f` : 4 évaluations CFD (XAUUSD et XAGUSD, avec et sans meta-labeling), refaites ensuite avec l'exécution corrigée ;
- `tsmom_dl` : baselines TSMOM, MACD et long-only, ainsi que leurs variantes de rebalancement et de bande (le détail figure dans `docs/GUIDE.md`) ; 1 fold DMN de test.

## Baselines corrigées — développement 2010-01-01 → 2021-12-31

Profil `estimates` (résultats provisoires). Moteur prop event-driven, capital de 100 000 USD fixe, exécution à la clôture t+1 sauf dans la colonne « open t+1 » (avec repli sur la clôture t+1 pour les opens figés). Les TSMOM sont rebalancées chaque semaine, avec une bande de 0,02 et une volatilité cible du portefeuille de 8 %. F2F est en mode CFD (volatilité cible de 10 %, paramètres par défaut). Le filtre de régime est en mode continu, en walk-forward avec 8 ans d'entraînement en expanding et 12 mois de test. Le challenge est simulé avec le preset FTMO 2-Step Swing : un départ par mois sur compte neuf, chaque phase plafonnée à 126 sessions. Ces 24 évaluations (12 variantes × 2 modes d'exécution) sont exploratoires et comptées dans le nombre d'essais du DSR.

| Stratégie | Régime | Sharpe brut | Sharpe net (clôture t+1) | Sharpe net (open t+1) | Turnover/an | Max DD | Pire jour (clôture) | Pire jour (intraday estimé) | Passage complet à 126 sessions | Phase 1 à 126 | Échecs | Timeouts | Départs | Coûts kUSD spread/comm./swap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TSMOM 1/3/12 | off | 0.07 | -0.64 | -0.58 | 39.6 | -72.6% | -4.20% | -6.40% | 1% | 4% | 30% | 69% | 138 | 18.1 / 0.7 / 56.6 |
| TSMOM 1/3/12 | on | 0.20 | -0.53 | -0.47 | 32.0 | -49.6% | -4.18% | -5.15% | 1% | 4% | 15% | 84% | 138 | 14.2 / 0.6 / 47.1 |
| TSMOM 12m | off | 0.34 | -0.33 | -0.34 | 20.8 | -55.4% | -4.10% | -6.65% | 1% | 9% | 35% | 64% | 139 | 8.7 / 0.4 / 64.1 |
| TSMOM 12m | on | 0.48 | -0.21 | -0.20 | 17.6 | -32.0% | -4.09% | -5.23% | 3% | 7% | 16% | 81% | 139 | 7.4 / 0.3 / 52.8 |
| MACD (Baz) | off | 0.08 | -0.53 | -0.46 | 19.1 | -56.6% | -3.78% | -6.68% | 1% | 6% | 25% | 73% | 139 | 8.3 / 0.3 / 54.7 |
| MACD (Baz) | on | 0.32 | -0.32 | -0.27 | 16.0 | -31.7% | -3.76% | -4.80% | 2% | 5% | 9% | 88% | 138 | 6.9 / 0.3 / 45.8 |
| Long-only | off | 0.44 | -0.11 | -0.09 | 7.6 | -49.6% | -5.15% | -8.16% | 1% | 7% | 43% | 55% | 139 | 1.8 / 0.1 / 60.2 |
| Long-only | on | 0.35 | -0.24 | -0.23 | 7.5 | -38.8% | -3.67% | -4.69% | 1% | 4% | 32% | 68% | 139 | 2.2 / 0.1 / 49.1 |
| F2F CFD XAUUSD | off | 0.49 | 0.30 | 0.25 | 23.9 | -32.1% | -3.22% | -4.90% | 3% | 12% | 1% | 96% | 138 | 7.6 / 0.5 / 8.1 |
| F2F CFD XAUUSD | on | 0.66 | 0.48 | 0.51 | 19.0 | -20.8% | -2.49% | -3.30% | 1% | 7% | 0% | 99% | 138 | 6.2 / 0.4 / 6.4 |
| F2F CFD XAGUSD | off | 0.40 | -0.20 | -0.00 | 13.6 | -42.6% | -3.89% | -3.89% | 5% | 15% | 13% | 82% | 138 | 48.1 / 0.6 / 3.9 |
| F2F CFD XAGUSD | on | 0.41 | -0.24 | 0.02 | 9.3 | -30.0% | -3.43% | -3.43% | 1% | 1% | 1% | 98% | 138 | 34.7 / 0.4 / 2.5 |

## Baselines — univers étendu (33 instruments), modèle de swap variable — développement 2010-01-01 → 2021-12-31

Profil `estimates` (résultats provisoires). Mêmes conventions que le tableau précédent. Les swaps suivent les taux directeurs historiques de la BIS, diminués des marges provisoires (FX 1,5 %, indices 2,5 %, métaux et énergie 2 %). La colonne « crédit réel » crédite les swaps positifs (sensibilité). Les variantes F2F ne dépendent pas de l'univers. Ces 36 évaluations (12 variantes × 3 configurations d'exécution et de crédit) sont exploratoires et comptées dans le nombre d'essais du DSR.

| Stratégie | Régime | Sharpe brut | Sharpe net (clôture t+1) | Sharpe net (open t+1) | Sharpe net (crédit réel) | Turnover/an | Max DD | Pire jour (clôture) | Pire jour (intraday estimé) | Passage complet à 126 | Phase 1 à 126 | Échecs | Timeouts | Départs | Coûts kUSD spread / comm. | Swap kUSD crédit 0 / crédit réel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TSMOM 1/3/12 | off | 0.02 | -0.55 | -0.48 | -0.54 | 45.6 | -67.1% | -3.63% | -6.00% | 0% | 1% | 29% | 71% | 138 | 21.4 / 0.8 | 44.1 / 42.9 |
| TSMOM 1/3/12 | on | 0.07 | -0.54 | -0.46 | -0.53 | 36.7 | -53.4% | -3.62% | -4.74% | 0% | 5% | 14% | 86% | 138 | 16.8 / 0.6 | 36.0 / 35.1 |
| TSMOM 12m | off | 0.37 | -0.15 | -0.09 | -0.14 | 24.4 | -53.9% | -3.27% | -5.71% | 1% | 7% | 36% | 63% | 138 | 10.2 / 0.4 | 50.4 / 48.9 |
| TSMOM 12m | on | 0.38 | -0.16 | -0.08 | -0.15 | 20.2 | -40.0% | -3.26% | -5.38% | 1% | 7% | 23% | 75% | 138 | 8.6 / 0.4 | 40.6 / 39.5 |
| MACD (Baz) | off | 0.01 | -0.48 | -0.41 | -0.47 | 20.2 | -55.0% | -3.50% | -6.19% | 0% | 2% | 22% | 78% | 139 | 8.7 / 0.3 | 41.2 / 40.0 |
| MACD (Baz) | on | 0.18 | -0.32 | -0.28 | -0.31 | 16.5 | -32.1% | -3.23% | -5.36% | 0% | 5% | 9% | 91% | 138 | 7.1 / 0.3 | 33.8 / 32.9 |
| Long-only | off | 0.20 | -0.12 | -0.10 | -0.10 | 7.0 | -49.9% | -9.38% | -15.73% | 3% | 6% | 51% | 46% | 139 | 1.5 / 0.1 | 36.4 / 35.0 |
| Long-only | on | 0.20 | -0.14 | -0.12 | -0.13 | 6.4 | -36.3% | -6.41% | -10.80% | 1% | 4% | 40% | 59% | 139 | 1.6 / 0.1 | 28.9 / 27.8 |
| F2F CFD XAUUSD | off | 0.49 | 0.30 | 0.25 | 0.30 | 23.9 | -32.1% | -3.22% | -4.90% | 3% | 12% | 1% | 96% | 138 | 7.6 / 0.5 | 8.1 / 8.1 |
| F2F CFD XAUUSD | on | 0.66 | 0.48 | 0.51 | 0.48 | 19.0 | -20.8% | -2.49% | -3.30% | 1% | 7% | 0% | 99% | 138 | 6.2 / 0.4 | 6.4 / 6.4 |
| F2F CFD XAGUSD | off | 0.40 | -0.20 | -0.00 | -0.20 | 13.6 | -42.6% | -3.89% | -3.89% | 5% | 15% | 13% | 82% | 138 | 48.1 / 0.6 | 3.9 / 3.9 |
| F2F CFD XAGUSD | on | 0.41 | -0.24 | 0.02 | -0.24 | 9.3 | -30.0% | -3.43% | -3.43% | 1% | 1% | 1% | 98% | 138 | 34.7 / 0.4 | 2.5 / 2.5 |

**Application de la règle de budget DMN (hypothèse 29 du GUIDE).** La meilleure baseline TSMOM avec bande et rebalancement hebdomadaire est la TSMOM 12 mois sans régime : Sharpe net de −0,15 (clôture t+1), −0,09 en open t+1, −0,14 en crédit réel. Le seuil de 0 n'est pas atteint : **le budget du tuning DMN reste fixé à 50 essais, avec un prior défavorable.**
