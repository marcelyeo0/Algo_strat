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
2. **Rolls des futures (source yfinance).** Les séries continues `=F` de yfinance présentent des sauts artificiels aux changements de contrat. Les dates de roll sont reconstruites à partir des calendriers d'échéance officiels (règles CME/ICE propres à chaque contrat, décrites dans `configs/rolls.yaml`), et seuls ces jours-là ont leur rendement neutralisé (mis à NaN, donc sans P&L). Les prix sont ensuite reconstruits par back-adjustment à partir des rendements nettoyés. Un seuil statistique (k écarts-types) sert uniquement au contrôle qualité, pour signaler des anomalies, jamais pour supprimer des données : il effacerait de vrais mouvements de marché. Le moment exact où Yahoo bascule d'un contrat au suivant n'est pas documenté. Une analyse des rendements autour des dates calculées n'a pas permis de l'identifier : pour l'or et l'argent, le saut de roll (le carry sur deux mois, de l'ordre de 0,1 à 0,5 %) est petit devant la volatilité quotidienne (environ 1 %). L'erreur de datation éventuelle a donc un impact limité, mais elle existe.
3. **Source des données.** L'ordre de priorité prévu est Dukascopy, puis yfinance, puis MT5. Lors du premier téléchargement (28/10/2026), le flux Dukascopy a répondu HTTP 429 (limitation de débit) de façon persistante. Le loader Dukascopy est implémenté (limitation à une requête par seconde, respect de `Retry-After`, cache des fichiers mensuels), mais la pipeline bascule automatiquement sur yfinance quand Dukascopy est indisponible. Toutes les données actuelles proviennent donc de yfinance. Relancer `uv run algo data --source dukascopy --refresh` quand le flux sera accessible.
4. **Alignement des clôtures avec yfinance.** L'alignement sur la clôture serveur FTMO (17:00 New York) n'est possible qu'à partir de barres intraday (Dukascopy H1, ou export MT5). Les barres daily de yfinance gardent leur session native (clôture de la bourse pour les indices et les futures, journée UTC/Londres pour le FX). Cela introduit un décalage de quelques heures entre instruments, sans effet de fuite d'information puisque les signaux restent calculés à la clôture et exécutés à l'ouverture suivante.
5. **Indices : prix cash plutôt que futures.** Pour US500, US100, US30, GER40, UK100 et JP225, on utilise l'indice cash (`^GSPC`, `^NDX`, `^DJI`, `^GDAXI`, `^FTSE`, `^N225`). C'est le sous-jacent direct des CFD « .cash » de FTMO, l'historique est long et il n'y a pas de roll à gérer. Le financement et les dividendes sont portés par les swaps du modèle de coûts.
6. **Or, argent, pétrole : futures.** yfinance ne fournit pas de spot pour l'or et l'argent. On utilise donc GC=F, SI=F, CL=F et BZ=F. Le rendement d'un future intègre déjà le coût de portage. Comme le swap CFD est quand même débité intégralement dans le modèle de coûts, le coût de portage est compté deux fois pour les positions acheteuses. C'est volontairement conservateur.
7. **Nettoyage OHLC.** Les highs/lows de Yahoo n'encadrent pas toujours l'open et la close (surtout en FX et sur les futures de métaux). Ces barres sont réparées en élargissant le range, et leur nombre figure dans le rapport qualité (`data/quality/`). Les prix nuls ou négatifs sont supprimés : c'est le cas du WTI le 20/04/2020, où la clôture était négative.
