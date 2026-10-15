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
   → `model_train_clean.csv` / `model_test_clean.csv` (15 features finales)
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

7. **Pas de comparaison au marché** : le modèle donne une probabilité, mais
   il n'y a pour l'instant aucune cote réelle à comparer pour calculer un
   edge. Voir "Cotes de marché" ci-dessous.

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

## Cotes de marché (en cours)

Objectif : calculer un edge réel en comparant les probabilités du modèle à
des cotes de marché, via [The Odds API](https://the-odds-api.com).

- Le marché `player_shots_on_goal` (NHL, bookmakers US) existe bien sur
  cette API, avec un sport key `icehockey_nhl`.
- **L'endpoint historique nécessite un plan payant** (confirmé
  empiriquement : `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` sur le tier
  gratuit, malgré une doc publique qui suggérait le contraire). Le tier
  payant le moins cher est 30$/mois (20K crédits) — pas retenu pour
  l'instant, décision de l'utilisateur de ne pas payer.
- **Pas d'alternative gratuite trouvée** : recherche faite sur Kaggle,
  BigDataBall, Princeton DSS, MoneyPuck, PropCruncher, Shot Propz, Odds
  Shark — soit ce sont des données de résultats réels (shots on goal en
  tant que stat de jeu, qu'on a déjà via MoneyPuck), soit des cotes de
  match (moneyline/total, pas de player props), soit des outils d'affichage
  live sans export ni archive. Les cotes historiques de player props ne
  semblent archivées nulle part gratuitement (marché encore récent, ~2021+).
- **Décision** : reporté au début de la saison 2026-2027 (octobre). Le tier
  gratuit de The Odds API donne accès aux cotes **live** (pas historiques),
  donc le plan est de scraper `player_shots_on_goal` au jour le jour dès le
  début de saison pour construire notre propre historique, gratuitement,
  et comparer aux prédictions du modèle au fil de l'eau. Scraper encore à
  écrire le moment venu (clé API déjà obtenue et testée, stockée hors dépôt
  git dans `pipeline_run/.env`, sport key et market key confirmés).

## Prochaines étapes suggérées

1. **Octobre 2026 (début de saison)** : écrire le scraper live
   `player_shots_on_goal` (The Odds API, tier gratuit) et commencer à
   construire un historique de cotes réelles pour comparer aux
   prédictions du modèle et calculer un edge au fil de la saison.
2. Modèle séparé pour le cold start des recrues (priors repêchage/junior).
3. Réfléchir à l'architecture du site web (stack, hosting, comment publier
   les prédictions quotidiennes, stratégie de contenu/SEO pour le trafic).

## Fichiers inclus dans ce transfert
- `scripts/` — tous les scripts Python du pipeline, dans l'ordre d'exécution
  (voir section Pipeline ci-dessus).
- `model_artifacts/` — `final_features.json` (15 features finales),
  et les
  modèles déjà entraînés sur ce feature set : `poisson_model.pkl`,
  `nb_params.pkl` (scipy), `nb_params_sm.pkl` (statsmodels), `scaler.pkl`,
  `xgb_model.pkl`, `lgb_model.pkl`.
- Les CSV intermédiaires (jusqu'à 2,6 Go pour le CSV brut extrait) ne sont
  **pas inclus** — ils sont entièrement reproductibles en ré-exécutant les
  scripts sur le fichier `2008_to_2024.zip` original (également exclu du
  dépôt, voir `.gitignore`).
