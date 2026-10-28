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
