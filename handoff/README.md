# Projet NHL Shots Props — Modèle prédictif de tirs au but

## Objectif du projet
Construire un modèle prédictif pour les marchés de paris "Over/Under tirs au but"
par joueur (shots on goal props), identifier des value bets potentiels, et
éventuellement monétiser via un site web (trafic, affiliation, contenu).

## Données sources
- Fichiers bruts : `2008_to_2024.zip` (fourni par l'utilisateur) + `2025.zip`
  (saison 2025-2026, téléchargée depuis https://moneypuck.com/data.htm —
  `https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/skaters/<année>.zip`),
  format MoneyPuck "skaters" — une ligne = un joueur, un match, une situation
  de jeu. Les saisons sont nommées par leur année de début (`2025` = 2025-2026).
  Saison régulière seulement (pas de séries).
- Colonnes clés : `playerId`, `gameId`, `season`, `situation` (5on5/5on4/4on5/all/other),
  `icetime`, `I_F_shotsOnGoal`, `I_F_shotAttempts`, etc.
- 18 saisons (2008-09 à 2025-26), 3 059 joueurs uniques, 21 600 matchs.

## Pipeline — ordre d'exécution des scripts

Tous les scripts sont dans `scripts/`. Reproduire dans cet ordre depuis un
dossier contenant les CSV dézippés dans `extracted/` (`2008_to_2024.csv`,
`2025.csv`, ... — `clean_reduce.py` lit tous les `extracted/*.csv`) :

1. **clean_reduce.py** — nettoie le CSV brut (2,5 Go), garde seulement les
   colonnes utiles, écrit `reduced_shots.csv` (~460 Mo).
2. **explore.py** — exploration de base, validation de la structure.
3. **overdispersion.py** — teste Poisson vs binomiale négative (ratio
   variance/moyenne = 1,41 → surdispersion confirmée → NB préférable).
4. **build_features.py** — construit les features joueur : moyennes mobiles
   de tirs/TOI (fenêtres 3/5/10/20), EWMA, moyenne de carrière, taux par 60 min,
   repos entre matchs. **Toutes les moyennes sont décalées (`shift(1)`) avant
   le rolling pour éviter toute fuite de données.** → `features_base.csv`
5. **build_opponent_context.py** — agrège au niveau équipe-match (pace
   offensif, tirs concédés), avec le même principe de décalage anti-fuite.
   → `team_game_context.csv`
6. **merge_all_features.py** — fusionne le contexte d'équipe/adversaire avec
   les features joueur. → `features_full_v1.csv`
7. **build_pp_features.py** — feature d'unité spéciale : usage en avantage
   numérique (5on4), classement relatif du joueur dans son équipe
   (`pp_usage_rank_team`), flags `is_pp1_likely`/`is_top5_pp_usage`.
   → `features_full_v2.csv` (dataset final, 777 367 lignes × 44 colonnes)
8. **correlation_analysis.py** — analyse de colinéarité, élagage des features
   redondantes (ex: toutes les fenêtres TOI étaient corrélées >0,92 entre elles ;
   `pp_toi_avg_last10` et `pp_usage_rank_team` corrélées à -0,91).
9. **prepare_model_data.py** — split temporel train (2008-09 à 2023-24) / test
   (2024-25 et 2025-26), imputation des NaN de contexte d'équipe (débuts de franchise),
   filtre des matchs "zéro" (premier match de carrière, pas d'historique
   disponible — 3 059 lignes, flag `games_played_before==0`), encodage
   catégoriel, et construction des termes d'interaction position×feature.
   → `model_train_clean.csv` / `model_test_clean.csv` (15 features finales),
   + `model_all_clean.csv` (tous les matchs joués, saison en cours comprise,
   pour le modèle de production). Les transformations sont dans
   `make_model_frame()`, réutilisée telle quelle par `predict_today.py`.
10. **fit_poisson.py** — modèle Poisson baseline (scikit-learn PoissonRegressor).
11. **fit_negbinom.py** — modèle binomiale négative maison (MLE via
    scipy.optimize, utile si `statsmodels` n'est pas disponible).
12. **fit_negbinom_statsmodels.py** — même modèle NB via `statsmodels`
    (`NegativeBinomial` natif) : mêmes coefficients que la version maison,
    avec en plus p-values / intervalles de confiance pour chaque feature.
13. **fit_gbm.py** — modèles gradient boosting (XGBoost et LightGBM,
    objectif Poisson) pour comparaison avec les modèles paramétriques.
14. **fit_segmented.py** — expérience : modèles Poisson séparés attaquants
    vs défenseurs. Conservé pour référence mais **non retenu** (voir
    "Points d'attention" ci-dessous — remplacé par des termes d'interaction
    dans le modèle poolé).
15. **calibration_check.py** — vérifie la calibration des probabilités
    Over/Under et calcule le Brier score (ligne 2,5 tirs).
16. **calibration_multi_line.py** — même vérification sur plusieurs lignes
    (0,5 / 1,5 / 2,5 / 3,5 / 4,5 tirs).
17. **fit_recalibration.py** — recalibration de mu (sortie NB) : ré-entraîne
    la NB sur 2008-2022, prédit 2023-24 (saison de validation), apprend une
    correction mu → mu' (log-linéaire et isotonique) sur cette saison, puis
    l'évalue sur le test avec les prédictions du modèle principal.
    → `recalibration.pkl`. **Diagnostic seulement — non appliqué en v4**
    (voir "Recalibration" ci-dessous).

## Résultats actuels (v4 — saison 2025-26 ajoutée + log des features de tirs)

- **Features finales retenues** (15 features, voir `model_artifacts/final_features.json`) :
  `log_shots_avg_last5`, `log_shots_career_avg`, `toi_avg_last10`,
  `pp_usage_rank_team`, `log_shots_per60_avg_last10`, `games_played_before`,
  `days_rest`, `team_pace_shots_for_last10`, `team_defense_shots_against_last10`,
  `position_D`, `position_L`, `position_R`, `home_or_away_HOME`,
  `toi_avg_last10_x_D`, `pp_usage_rank_team_x_D`.
  Les trois features de volume de tirs sont en `log(x + 0,25)` depuis v4
  (voir point d'attention 6).

- **Comparaison des modèles (test 2024-25 + 2025-26, n=94 215 ; train 2008-09 à 2023-24, n=680 093)** :

  | Modèle | MAE test | Brier score (Over 2,5) |
  |---|---|---|
  | Baseline (moyenne de ligue) | 1,1948 | 0,1715 |
  | Poisson (sklearn) | 1,0194 | 0,1481 |
  | NB (scipy maison / statsmodels) | 1,0193 | 0,1482 |
  | XGBoost (objectif Poisson) | 1,0200 | 0,1480 |
  | LightGBM (objectif Poisson) | 1,0205 | 0,1480 |

  Amélioration relative ~14,7% (MAE) / ~13,6% (Brier) vs baseline naïve.
  MAE par saison de test (NB) : 1,0233 en 2024-25, 1,0152 en 2025-26 — pas de
  dégradation sur la saison la plus récente. Les modèles paramétriques sont
  maintenant au niveau des GBM, y compris sur le haut de la distribution.
  *(v2, split précédent train 2008-2022 / test 2023-2024 : MAE NB 1,0400,
  Brier 0,1545 vs baseline 1,2123 / 0,1803 — les chiffres absolus ne sont pas
  directement comparables car l'ensemble de test a changé.)*

- **Interactions** toujours très significatives (statsmodels : z=-14,9 pour
  `toi_avg_last10_x_D`, z=-9,2 pour `pp_usage_rank_team_x_D` — cette
  dernière plus faible qu'avant, une partie de l'effet étant maintenant
  captée par la forme log des features de tirs).

- **Alpha (dispersion NB) estimé** : 0,063 — la surdispersion "brute" venait
  surtout de l'hétérogénéité entre joueurs, pas de bruit résiduel une fois les
  features connues.

- **Calibration** : par décile de mu, réel/prédit va de 0,97 à 1,04 (v3,
  features brutes : 0,91 à 1,06). Erreur moyenne par décile de probabilité :
  0,99 / 1,10 / 0,81 / 0,59 / 0,41 point sur les lignes 0,5 / 1,5 / 2,5 /
  3,5 / 4,5 (v3 : 1,55 / 1,60 / 1,13 / 0,68 / 0,36). Il reste une
  **sous-estimation de 2-3 points dans les 3 déciles hauts** (ex. Over 2,5,
  décile 10 : 51,0% prédit vs 53,9% réel ; Over 3,5 : 31,0% vs 32,8%).
  Ce biais est conservateur : il peut faire rater des Overs, mais ne crée
  pas de faux edge.

- **Recalibration de mu — non retenue** (`fit_recalibration.py`). Apprise sur
  2023-24 et évaluée sur le test, la version log-linéaire (b=1,10) réduit
  l'erreur moyenne par décile (0,61 / 0,62 / 0,46 / 0,40 / 0,26 point), mais
  **surestime à nouveau le haut** : décile 10 de l'Over 3,5 à 34,4% prédit
  vs 32,8% réel, top 0,1% des prédictions à 4,44 tirs vs 3,96 réels
  (+12%). L'isotonique est pire sur ce point (+15%). Une correction apprise
  sur une seule saison ne se transfère pas assez bien ; une erreur
  conservatrice est préférable pour parier. À réévaluer avec plusieurs
  saisons de validation (validation glissante) ou une fois les cotes
  disponibles.

## Points d'attention identifiés

1. **[RÉSOLU] Colinéarité PP** : `pp_toi_avg_last10` et `pp_usage_rank_team`
   étaient corrélées à -0,91, rendant le coefficient de `pp_toi_avg_last10`
   instable (signe contre-intuitif). `pp_toi_avg_last10` a été retiré du
   feature set ; `pp_usage_rank_team` conservé seul (légèrement plus corrélé
   à la cible et plus important en GBM).

2. **[RÉSOLU] Segmentation par position testée** : un modèle Poisson séparé
   attaquants vs défenseurs (`fit_segmented.py`) n'a apporté aucun gain de
   MAE/Brier par rapport au modèle poolé (qui contenait déjà les dummies de
   position). En revanche, les coefficients segmentés ont révélé deux pentes
   clairement différentes : `toi_avg_last10` compte plus pour les attaquants,
   `pp_usage_rank_team` compte plus pour les défenseurs. Plutôt que dupliquer
   le modèle, ces effets ont été intégrés comme **termes d'interaction**
   (`toi_avg_last10_x_D`, `pp_usage_rank_team_x_D`) dans le modèle poolé.
   Les deux interactions sont statistiquement très significatives
   (statsmodels : p<0,001, z=-15,5 et z=-20,4) et améliorent légèrement le
   MAE des modèles Poisson/NB (1,0407→1,0405 et 1,0402→1,0400).

3. **Cold start des recrues** : les matchs de premier match en carrière
   (3 059 lignes) n'ont aucune feature historique — actuellement exclus du
   modèle. Pourrait nécessiter un modèle séparé basé sur des priors
   (repêchage, ligue junior, etc.) si on veut couvrir ce cas.

4. **Contexte d'équipe faible** : le pace/défense d'équipe a une corrélation
   étonnamment faible avec la cible (~0,04-0,06). Possiblement un signal réel
   mais petit, ou une mesure trop bruitée (à tester : écart à la moyenne de
   ligue plutôt que valeur brute).

5. **[RÉSOLU] Calibration testée sur d'autres lignes** (0,5 / 1,5 / 2,5 / 3,5 /
   4,5 tirs, `scripts/calibration_multi_line.py`). Le modèle reste calibré
   en moyenne sur toutes les lignes ; en v4 il reste une légère
   sous-estimation des cas à forte probabilité (voir "Calibration"
   ci-dessus). L'edge vs baseline naïve (Brier, v4) est le plus fort au
   milieu de la distribution (~13,6% sur 1,5 et 2,5 tirs) et plus faible aux
   extrêmes (~9% sur 0,5, ~11% sur 3,5, ~8% sur 4,5 tirs). NB bat
   systématiquement Poisson en calibration (pas en Brier) sur toutes les
   lignes, cohérent avec la surdispersion confirmée en début de projet.

6. **[RÉSOLU] Gros tireurs surestimés** : en v3, sur le 1% des prédictions
   les plus hautes (MacKinnon, Kaprizov, Matthews, Pastrnak, Ovechkin...), la
   NB prédisait 3,92 tirs pour 3,61 réels, et 5,01 pour 3,88 sur le top
   0,1%, alors que XGBoost/LightGBM étaient justes (3,60 / 4,06). Cause :
   avec un lien log, les features brutes en tirs font croître mu
   exponentiellement, et ces joueurs sont extrêmes sur les trois features de
   tirs à la fois. Correction : `log(x + 0,25)` sur `shots_avg_last5`,
   `shots_career_avg` et `shots_per60_avg_last10` (mu varie alors comme une
   puissance de la moyenne de tirs). Résultat sur les mêmes tranches :
   3,52 prédit vs 3,66 réel (p99-99,9) et 4,04 vs 3,96 (top 0,1%), au
   niveau des GBM. Testé aussi : log du TOI en plus (pas mieux), features
   brutes + log en parallèle (MAE à peine meilleur mais top 0,1% encore
   +15%).

7. **Modèle ~5 points sous le marché** (première capture de cotes, 15 oct.
   2026, 215 lignes bookmaker × joueur × ligne) : P(Over) du modèle en
   moyenne 5 points sous la probabilité "juste" du marché, sur toutes les
   lignes, les deux bookmakers et tous les niveaux de mu → 61% des Unders
   ont une EV > 0 contre 8% des Overs. Ce n'est **pas** un edge exploitable
   en l'état. Pas de dérive de ligue (2026-27 : 1,50 tir/match réel vs 1,48
   prédit). Explications probables : (a) la sous-estimation connue du haut
   de la distribution — sur le quintile supérieur de 2026-27, 2,42 prédit
   vs 2,54 réel, or ce sont les joueurs cotés ; (b) des cotes Over gonflées
   par les bookmakers (le public parie surtout les Overs), ce qui fausse le
   retrait du vig par simple normalisation. **Impossible à trancher sans résultats** : il faut d'abord
   noter les cotes capturées contre les tirs réels (Brier modèle vs marché)
   sur quelques semaines. **Ne pas parier sur ces edges avant.**

## Environnement / contraintes techniques

- Le pipeline tourne maintenant **en local** dans un venv Python 3.14
  (`pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `xgboost`,
  `lightgbm`). `xgboost`/`lightgbm` nécessitent `libomp` (installé via
  Homebrew — absent par défaut sur macOS).
- Le sandbox utilisé pour le travail initial n'avait pas d'accès réseau ni
  `statsmodels`/`xgboost`/`lightgbm`, d'où l'implémentation "maison" de la
  binomiale négative (`fit_negbinom.py`, MLE scipy). Cette version est
  conservée pour référence/portabilité ; `fit_negbinom_statsmodels.py` est
  la version à privilégier en local (résultats identiques + tests
  statistiques complets).

## Pipeline live (saison 2026-27)

`run_daily.py` enchaîne tout, chaque jour de match (log :
`pipeline_run/logs/daily_<date>.log`) :

```
.venv/bin/python handoff/scripts/run_daily.py              # matchs d'aujourd'hui
NO_ODDS=1 .venv/bin/python handoff/scripts/run_daily.py    # sans dépenser de crédits
```

0. **grade_results.py** — note les jours précédents (voir "Test du modèle").

1. **update_live_data.py** — télécharge le fichier MoneyPuck de la saison en
   cours (`extracted/<saison>.csv`, mis à jour vers midi avec les matchs de
   la veille), lit le calendrier et les rosters via l'API NHL
   (`api-web.nhle.com`), et écrit `upcoming_rows.csv` : une ligne par joueur
   probable des matchs du jour (roster actuel, a joué dans les 30 derniers
   jours), stats de tirs vides. Comme toutes les features sont décalées, les
   scripts de features existants calculent pour ces lignes les features
   d'entrée du match sans modification. Prévient si des matchs déjà joués
   manquent encore dans MoneyPuck. Écrit aussi `upcoming_games.json`.
2. **clean_reduce.py --upcoming** puis les scripts de features (étapes 4-7).
   Sans `--upcoming`, ces lignes ne sont jamais incluses (un vieux fichier ne
   peut pas polluer l'historique d'entraînement).
3. **predict_today.py** — applique `make_model_frame()` et le modèle de
   production ; → `predictions/predictions_<date>.csv` (mu et P(Over) pour
   0,5 à 4,5 tirs, et cotes justes décimales `fair_over_<ligne>` /
   `fair_under_<ligne>` = 1/P, pour comparer à la main avec bet365 : une cote
   offerte au-dessus de la cote juste a une EV positive si le modèle a
   raison. Tant que le biais de ~5 points sous le marché n'est pas résolu,
   les cotes justes Over sont trop hautes et les Under trop basses).
4. **fetch_odds.py** — cotes `player_shots_on_goal` (The Odds API, liste de
   10 bookmakers nommés dont Pinnacle — même coût qu'une région, 1 crédit par match, matchs pas encore commencés seulement ; `--dry-run`
   pour ne rien dépenser). Brut → `odds/raw/<date>/`, à plat →
   `odds/odds_<date>.csv`. S'arrête s'il reste moins de 20 crédits.
5. **compare_odds.py** — associe les noms (normalisés, homonymes résolus par
   le temps de glace), retire le vig (normalisation Over+Under), calcule
   edge et EV ; → `odds/edges_<date>.csv`.
6. **make_picks.py** — picks du jour, figés (voir "Test du modèle").

**Automatisation** (agents launchd dans `~/Library/LaunchAgents/`) :
- `com.nhlshots.daily` — `run_daily.py` tous les jours à 13 h (ET) : après la
  mise à jour MoneyPuck (~midi), avant les matchs du soir. Les matchs déjà
  commencés à ce moment ne sont pas capturés.
- `com.nhlshots.closing` — `fetch_odds.py --closing` toutes les 30 min de
  13 h à 22 h (la dernière couvre les matchs jusqu'à 22 h 30 ; avant 13 h
  inutile, les matchs déjà commencés à 13 h n'ont pas de picks) : cotes de fermeture des matchs qui commencent dans les 30
  minutes, une fois par match (log : `logs/closing.log`).
- `com.nhlshots.awake` — `caffeinate -i` de 11 h 59 à 22 h 05 : empêche
  la mise en veille du Mac par inactivité, **sur secteur et sur batterie**
  (l'écran s'éteint toujours ; fermer le couvercle met quand même le Mac en
  veille). Nécessaire car le réglage d'origine met le Mac en veille après
  1 min d'inactivité. Jusqu'au 27 oct. 2026 c'était `caffeinate -s`
  (secteur seulement) : sur batterie, les jobs échouaient faute de réseau. Le Mac est réveillé
  à 11 h 59 par un réveil programmé macOS, à configurer une fois avec
  `sudo pmset repeat wakeorpoweron MTWRFSU 11:59:00` (vérifier avec
  `pmset -g sched`, annuler avec `sudo pmset repeat cancel`). À laisser
  couvercle ouvert ; sur batterie, compter ~10 h d'éveil par jour.
- Si le Mac dort quand même à l'heure prévue, launchd lance le job une fois
  au réveil ; s'il est éteint, le job est perdu. Une journée manquée = une
  journée de cotes perdue (pas d'historique gratuit), mais aucune donnée de
  matchs perdue (MoneyPuck re-téléchargé en entier à chaque exécution).
- Les agents lancent directement le Python du projet (`.venv/bin/python`),
  pas `/bin/bash` : sur macOS, un job launchd lancé via bash n'a pas accès
  au dossier Documents ("Operation not permitted"), le Python oui.
- Désactiver : `launchctl bootout gui/$(id -u)/com.nhlshots.daily` (idem
  `.closing`, `.awake`) ; lancer à la main : `launchctl kickstart gui/$(id -u)/com.nhlshots.daily`.

## Test du modèle (saison 2026-27)

Trois mesures, de la plus rapide à la plus lente, calculées par
`grade_results.py` dès que MoneyPuck a les résultats (le lendemain) :

1. **Toutes les lignes capturées, modèle vs marché** (`graded/lines_<date>.csv`)
   — Brier et log loss du P(Over) du modèle et de la probabilité juste du
   marché (vig retiré), sur ~200-400 lignes par soir ; taux réel d'Over vs
   moyenne modèle vs moyenne marché, par ligne. Le test principal : il dit
   en quelques semaines qui est le plus juste, et d'où vient l'écart de ~5
   points (point d'attention 7). Joueur qui ne joue pas = ligne annulée.
2. **CLV des picks** — prix pris à 13 h évalué avec la probabilité juste de
   fermeture (Pinnacle si elle cote la ligne, sinon moyenne des
   bookmakers ; capture 0-30 min avant le match) :
   `clv = prix × P_fermeture − 1`, > 0 = on a battu la fermeture. Bien
   moins bruité que le gain/perte ; le standard pour juger un edge.
3. **Paper trading** (`picks/picks_<date>.csv`, notés dans
   `graded/picks_<date>.csv`) — règles fixées d'avance dans
   `make_picks.py` : seulement les lignes cotées par **Pinnacle, au prix
   Pinnacle** (un livre où l'utilisateur parie, le plus sharp, et pas de
   lignes alternatives à un seul livre), EV ≥ 5 %, un pari par joueur, au
   plus 10 par jour, mise fixe 1 u. Deux
   portefeuilles : modèle brut et modèle avec l'écart moyen au marché
   retiré. Fichier figé à 13 h, jamais réécrit. Le gain/perte est le moins
   précis des trois : à ~1,9 de cote, ± 12 points de ROI après un mois,
   ± 5 après une saison.

Historique des règles : le 15 oct. 2026, première version des picks au
prix médian entre bookmakers → les 20 picks venaient de lignes alternatives
cotées par un seul livre (theScore Bet, Hard Rock) à des prix longs (2,45 à
4,60), où le biais bas du modèle donnait des EV de +30 à +75 %. Règle
changée pour "Pinnacle seulement" et picks du jour régénérés **avant le
début des matchs** (aucun résultat connu).

Décision après 4-6 semaines : sans précision au moins égale au marché (1)
et CLV positive (2), le résultat du paper trading n'est que de la chance.

Crédits : capture de 13 h + fermeture ≈ 2 crédits par match, ~420-470 par
mois selon le calendrier, sous les 500 du tier gratuit mais sans grande
marge (`fetch_odds.py` s'arrête sous 20 crédits restants).

**Modèle de production** (`fit_production.py`, après `prepare_model_data.py`) :
même NB, entraînée sur tous les matchs joués (2008 → saison en cours) →
`prod_scaler.pkl` / `prod_nb_params.pkl`. Pas besoin de la ré-entraîner
chaque jour (les features, elles, sont à jour quotidiennement) ; une fois par
mois suffit.

Notes :
- Noms des joueurs du jour tirés du roster NHL : MoneyPuck perd les accents
  (« Tim Sttzle », « Alexis Lafrenire »), ce qui cassait l'association avec
  les noms des cotes.
- Joueurs blessés mais encore au roster : prédits quand même ; ils
  disparaissent à l'association avec les cotes (pas de prop). Ils peuvent
  décaler un peu `pp_usage_rank_team` de leurs coéquipiers.
- Crédits : ~1 312 matchs/saison ≈ 220 crédits/mois pour une capture par
  match, sous les 500 du tier gratuit. Première capture : 11 crédits
  (489 restants au 15 oct. 2026).
- Bookmakers : l'utilisateur parie sur **bet365 et Pinnacle**. bet365 n'est
  pas couvert par The Odds API (seulement `bet365_au`, payant, AFL/NRL).
  Pinnacle (région `eu`) est demandé à chaque capture : aucune prop de tirs
  NHL à ~1 h du matin le 15 oct., mais 159 joueurs sur les 11 matchs à 13 h
  (il publie dans la journée). Capture de 13 h du 15 oct. : 8 bookmakers,
  ~3 000 cotes. La doc indique
  que les props sont "mainly limited to US sports and US bookmakers".
- `certifi` ajouté au venv (le Python de python.org sur macOS n'a pas de
  certificats SSL par défaut).

## Cotes de marché — historique de la décision

- Le marché `player_shots_on_goal` (NHL, bookmakers US) existe bien sur
  The Odds API, avec un sport key `icehockey_nhl`.
- **L'endpoint historique nécessite un plan payant** (confirmé
  empiriquement : `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` sur le tier
  gratuit). Le tier payant le moins cher est 30$/mois — pas retenu.
- **Pas d'alternative gratuite trouvée** (Kaggle, BigDataBall, Princeton
  DSS, MoneyPuck, PropCruncher, Shot Propz, Odds Shark).
- **Décision** : construire notre propre historique en capturant les cotes
  live chaque jour depuis octobre 2026 (voir "Pipeline live").

## Prochaines étapes suggérées

1. **Noter les résultats** : script qui, pour chaque date capturée, relie
   les cotes aux tirs réels (MoneyPuck du lendemain) et compare modèle vs
   marché (Brier, calibration, P&L simulé des edges). Nécessaire pour
   trancher le point d'attention 7 avant tout pari.
2. Capture plus tôt pour les matchs de l'après-midi (week-end), manqués
   par l'exécution de 13 h.
3. Modèle séparé pour le cold start des recrues (priors repêchage/junior).
4. Réfléchir à l'architecture du site web (stack, hosting, comment publier
   les prédictions quotidiennes, stratégie de contenu/SEO pour le trafic).

## Fichiers inclus dans ce transfert
- `scripts/` — tous les scripts Python du pipeline, dans l'ordre d'exécution
  (voir section Pipeline ci-dessus).
- `model_artifacts/` — `final_features.json` (15 features finales),
  et les
  modèles déjà entraînés sur ce feature set : `poisson_model.pkl`,
  `nb_params.pkl` (scipy), `nb_params_sm.pkl` (statsmodels), `scaler.pkl`,
  `xgb_model.pkl`, `lgb_model.pkl` (entraînés sur 2008-2023, pour
  l'évaluation), et le modèle de production `prod_scaler.pkl` /
  `prod_nb_params.pkl` (entraîné sur tous les matchs jusqu'au 13 oct. 2026).
- Les CSV intermédiaires (jusqu'à 2,6 Go pour le CSV brut extrait) ne sont
  **pas inclus** — ils sont entièrement reproductibles en ré-exécutant les
  scripts sur le fichier `2008_to_2024.zip` original (également exclu du
  dépôt, voir `.gitignore`).
