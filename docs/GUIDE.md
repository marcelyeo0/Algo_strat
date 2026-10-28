# Guide du projet Algo_strat

> Ce guide est complété au fil des phases du projet. La version complète (architecture, stratégies, protocole de validation, lecture des rapports) est rédigée en Phase 8.

## Environnement

Le projet tourne directement sous Windows, sans Docker. Python 3.12 et toutes les dépendances sont gérés par uv :

```powershell
uv sync                 # environnement standard
uv sync --extra mt5     # ajoute le package MetaTrader5 (Windows uniquement)
uv run algo --help      # liste des commandes
```

## Hypothèses

Cette section recense toutes les hypothèses prises lorsque une information manquait ou était ambiguë. Lorsque plusieurs choix étaient possibles, c'est le plus conservateur qui a été retenu.

1. **Décalage entre la clôture daily et le reset de la perte journalière.** Les barres daily sont clôturées à 00:00 heure serveur FTMO (heure d'Europe de l'Est, EET/EEST), ce qui correspond à environ 17:00 à New York. La perte journalière FTMO, elle, se réinitialise à minuit heure de Prague (CE(S)T), soit environ une heure plus tôt. Avec des barres daily, on ne connaît pas l'equity à 23:00 heure serveur. On approxime donc la référence de la perte journalière du jour *d* par le maximum de la balance et de l'equity **à la clôture de la barre d−1**. Les variations de prix survenues pendant la dernière heure de la barre d−1 sont ainsi attribuées au jour d−1 au lieu du jour *d*.
2. **Rolls des futures (source yfinance).** Les séries continues `=F` de yfinance présentent des sauts artificiels aux changements de contrat. Les dates de roll sont reconstruites à partir des calendriers d'échéance officiels (règles CME/ICE propres à chaque contrat), et seuls ces jours-là ont leur rendement neutralisé. Un seuil statistique (k écarts-types) sert uniquement au contrôle qualité, pour signaler des anomalies, jamais pour supprimer des données : il effacerait de vrais mouvements de marché.
