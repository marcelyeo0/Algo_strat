# Proposition d'univers étendu (en attente de validation)

La TSMOM vit de la diversification. Sur 2010–2021, l'univers actuel (17 instruments) a une corrélation mensuelle absolue moyenne de 0,41, pour environ 3,9 paris indépendants (participation des valeurs propres de la matrice de corrélation). Avec les 26 candidats ci-dessous, hors crypto, on passe à une corrélation moyenne de 0,31 et à environ 6,0 paris indépendants.

La disponibilité de chaque symbole chez FTMO est à vérifier sur le terminal (TODO). Les coûts sont des estimations prudentes. Le coût par côté est calculé au prix médian 2015–2021, sous la forme demi-spread plus commission rapportée au notionnel.

| Instrument | Ticker yfinance | Classe | Historique | Opens figés | Spread estimé | Commission/lot/côté | Coût par côté (bp) | Swap |
|---|---|---|---|---|---|---|---|---|
| EURGBP | EURGBP=X | fx | 2000 → | 8 % | 0.00015 | 2.5 | 1.2 | complet −2/−2 % |
| EURJPY | EURJPY=X | fx | 2003 → | 4 % | 0.020 | 2.5 | 0.8 | complet −2/−2 % |
| GBPJPY | GBPJPY=X | fx | 2003 → | 3 % | 0.035 | 2.5 | 1.2 | complet −2/−2 % |
| AUDJPY | AUDJPY=X | fx | 2003 → | 3 % | 0.020 | 2.5 | 1.2 | complet −2/−2 % |
| EURCHF | EURCHF=X | fx | 2003 → | 6 % | 0.00020 | 2.5 | 1.1 | complet −2/−2 % |
| AUDNZD | AUDNZD=X | fx | 2003 → | 3 % | 0.00030 | 2.5 | 1.6 | complet −2/−2 % |
| CADJPY | CADJPY=X | fx | 2004 → | 3 % | 0.025 | 2.5 | 1.5 | complet −2/−2 % |
| AUS200 | ^AXJO | indice | 2000 → | 46 % | 2.0 | 0 | 1.7 | complet −6/−2 % |
| EU50 | ^STOXX50E | indice | 2007 → | 0 % | 2.0 | 0 | 2.9 | complet −5/−2 % |
| FRA40 | ^FCHI | indice | 2000 → | 0 % | 1.5 | 0 | 1.4 | complet −5/−2 % |
| HK50 | ^HSI | indice | 2000 → | 0 % | 10 | 0 | 1.9 | complet −6/−2 % |
| SPN35 | ^IBEX | indice | 2000 → | 5 % | 8 | 0 | 4.3 | complet −5/−2 % |
| N25 | ^AEX | indice | 2000 → | 0 % | 0.30 | 0 | 2.8 | complet −5/−2 % |
| US2000 | ^RUT | indice | 2000 → | 23 % | 0.6 | 0 | 2.0 | complet −6/−2 % |
| DXY | DX-Y.NYB | indice | 2000 → | 6 % | 0.05 | 0 | 2.6 | complet −2/−2 % |
| NATGAS | NG=F | énergie | 2000 → | 3 % | 0.010 | 0 | 18.1 | marge −2 % |
| XPTUSD | PL=F | métal | 2000 → | 2 % | 3.0 | 2.5 | 16.1 | marge −2 % |
| XPDUSD | PA=F | métal | 2000 → | 3 % | 8.0 | 2.5 | 39.5 | marge −2 % |
| COPPER | HG=F | métal | 2000 → | 3 % | 0.006 | 0 | 11.0 | marge −2 % |
| COCOA | CC=F | agricole | 2000 → | 7 % | 15 | 0 | 30.5 | marge −2 % |
| COFFEE | KC=F | agricole | 2000 → | 8 % | 0.60 | 0 | 24.4 | marge −2 % |
| CORN | ZC=F | agricole | 2000 → | 19 % | 1.0 | 0 | 13.5 | marge −2 % |
| COTTON | CT=F | agricole | 2000 → | 10 % | 0.30 | 0 | 21.4 | marge −2 % |
| SOYBEAN | ZS=F | agricole | 2000 → | 11 % | 1.5 | 0 | 7.8 | marge −2 % |
| WHEAT | ZW=F | agricole | 2000 → | 12 % | 1.5 | 0 | 14.9 | marge −2 % |
| SUGAR | SB=F | agricole | 2000 → | 15 % | 0.05 | 0 | 17.9 | marge −2 % |
| BTCUSD | BTC-USD | crypto | 2014 → | 1 % | 60 | 0 | 46.8 (*) | complet −25/−25 % |
| ETHUSD | ETH-USD | crypto | 2017 → | 1 % | 4 | 0 | 55.5 (*) | complet −25/−25 % |

(*) Pour la crypto, le coût en points de base est surestimé à cause du prix médian 2015–2021, très inférieur au prix actuel. Il faudrait plutôt un spread en pourcentage (de l'ordre de 0,05 %).

Remarques :

- **Rolls** : les futures agricoles, le gaz naturel, le platine, le palladium et le cuivre nécessitent des règles de roll supplémentaires dans `configs/rolls.yaml` (calendriers CME/ICE propres à chaque contrat).
- **Crypto** : elle se négocie 7 jours sur 7. Il faudrait gérer un calendrier propre (le week-end, la pire equity et les swaps changent), et son historique est court (2014 et 2017). Elle ne couvre que partiellement la période de développement.
- **Opens figés** : AUS200 (46 %), US2000 (23 %) et plusieurs contrats agricoles dépassent le seuil de 10 %. Ils seraient exécutés à la clôture t+1.
- **Obligations** : FTMO ne propose a priori pas de CFD sur les taux (à vérifier). C'est la classe la plus décorrélante pour une TSMOM, et elle manque.
