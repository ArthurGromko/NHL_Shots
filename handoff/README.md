# Projet NHL Shots Props — Modèle prédictif de tirs au but

## Objectif du projet
Construire un modèle prédictif pour les marchés de paris "Over/Under tirs au but"
par joueur (shots on goal props), identifier des value bets potentiels, et
éventuellement monétiser via un site web (trafic, affiliation, contenu).

## Données sources
- Fichier brut : `2008_to_2024.zip` (fourni par l'utilisateur), format MoneyPuck
  "skaters" — une ligne = un joueur, un match, une situation de jeu.
- Colonnes clés : `playerId`, `gameId`, `season`, `situation` (5on5/5on4/4on5/all/other),
  `icetime`, `I_F_shotsOnGoal`, `I_F_shotAttempts`, etc.
- 17 saisons (2008-2024), ~2 945 joueurs uniques, ~20 288 matchs.

## Pipeline — ordre d'exécution des scripts

Tous les scripts sont dans `scripts/`. Reproduire dans cet ordre depuis un
dossier contenant `extracted/2008_to_2024.csv` (dézippé du fichier original) :

1. **clean_reduce.py** — nettoie le CSV brut (2,5 Go), garde seulement les
   colonnes utiles, écrit `reduced_shots_2008_2024.csv` (~440 Mo).
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
   → `features_full_v2.csv` (dataset final, 730 155 lignes × 44 colonnes)
8. **correlation_analysis.py** — analyse de colinéarité, élagage des features
   redondantes (ex: toutes les fenêtres TOI étaient corrélées >0,92 entre elles ;
   `pp_toi_avg_last10` et `pp_usage_rank_team` corrélées à -0,91).
9. **prepare_model_data.py** — split temporel train (2008-2022) / test
   (2023-2024), imputation des NaN de contexte d'équipe (débuts de franchise),
   filtre des matchs "zéro" (premier match de carrière, pas d'historique
   disponible — 2 945 lignes, flag `games_played_before==0`), encodage
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
    Over/Under et calcule le Brier score.

## Résultats actuels (v2 — post correction colinéarité PP + interactions)

- **Features finales retenues** (15 features, voir `model_artifacts/final_features.json`) :
  `shots_avg_last5`, `shots_career_avg`, `toi_avg_last10`, `pp_usage_rank_team`,
  `shots_per60_avg_last10`, `games_played_before`, `days_rest`,
  `team_pace_shots_for_last10`, `team_defense_shots_against_last10`,
  `position_D`, `position_L`, `position_R`, `home_or_away_HOME`,
  `toi_avg_last10_x_D`, `pp_usage_rank_team_x_D`.

- **Comparaison des modèles (test 2023-2024, n=94 227)** :

  | Modèle | MAE test | Brier score (Over 2,5) |
  |---|---|---|
  | Baseline (moyenne de ligue) | 1,2123 | 0,1803 |
  | Poisson (sklearn) | 1,0405 | 0,1545 |
  | NB (scipy maison / statsmodels) | 1,0400 | 0,1545 |
  | XGBoost (objectif Poisson) | 1,0402 | 0,1540 |
  | LightGBM (objectif Poisson) | 1,0405 | 0,1540 |

  Amélioration relative ~14,2% vs baseline naïve. Les GBM font légèrement
  mieux sur le Brier score, mais l'écart avec les modèles paramétriques reste
  faible — le signal est déjà bien capturé linéairement une fois les bonnes
  features (et interactions) en place.

- **Alpha (dispersion NB) estimé** : 0,067 — la surdispersion "brute" venait
  surtout de l'hétérogénéité entre joueurs, pas de bruit résiduel une fois les
  features connues.

- **Calibration** : bonne à travers les déciles, léger biais conservateur dans
  le décile de probabilité le plus élevé (le modèle sous-estime un peu les cas
  à forte probabilité).

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
   (2 945 lignes) n'ont aucune feature historique — actuellement exclus du
   modèle. Pourrait nécessiter un modèle séparé basé sur des priors
   (repêchage, ligue junior, etc.) si on veut couvrir ce cas.

4. **Contexte d'équipe faible** : le pace/défense d'équipe a une corrélation
   étonnamment faible avec la cible (~0,04-0,06). Possiblement un signal réel
   mais petit, ou une mesure trop bruitée (à tester : écart à la moyenne de
   ligue plutôt que valeur brute).

5. **Pas encore testé** : calibration sur d'autres lignes (1,5 / 3,5 / 4,5
   tirs).

6. **Pas de comparaison au marché** : le modèle donne une probabilité, mais
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
des cotes de marché historiques, via [The Odds API](https://the-odds-api.com).

- Le marché `player_shots_on_goal` (NHL, bookmakers US) existe bien sur
  cette API.
- L'historique des player props n'est disponible qu'à partir du
  **2023-05-03** — ce qui coïncide bien avec notre période de test
  (saisons 2023-2024).
- Le tier gratuit donne 500 crédits/mois ; le coût exact en crédits d'une
  requête historique sur des player props n'est pas documenté publiquement
  et reste à vérifier empiriquement (probablement insuffisant pour couvrir
  une saison complète avec le tier gratuit).
- **Bloqué en attente d'une clé API** : la création de compte doit être
  faite par l'utilisateur (the-odds-api.com), le scraper n'est pas encore
  écrit.

## Prochaines étapes suggérées

1. Obtenir une clé The Odds API et écrire le scraper de cotes historiques
   `player_shots_on_goal` sur l'échantillon 2023-2024 couvert par l'API.
2. Une fois les cotes obtenues, calculer l'edge (probabilité modèle vs
   probabilité implicite du marché) et backtester une stratégie simple.
3. Tester la calibration sur d'autres lignes (1,5 / 3,5 / 4,5 tirs).
4. Modèle séparé pour le cold start des recrues (priors repêchage/junior).
5. Réfléchir à l'architecture du site web (stack, hosting, comment publier
   les prédictions quotidiennes, stratégie de contenu/SEO pour le trafic).

## Fichiers inclus dans ce transfert
- `scripts/` — tous les scripts Python du pipeline, dans l'ordre d'exécution
  (voir section Pipeline ci-dessus).
- `model_artifacts/` — `final_features.json` (15 features finales), et les
  modèles déjà entraînés sur ce feature set : `poisson_model.pkl`,
  `nb_params.pkl` (scipy), `nb_params_sm.pkl` (statsmodels), `scaler.pkl`,
  `xgb_model.pkl`, `lgb_model.pkl`.
- Les CSV intermédiaires (jusqu'à 2,6 Go pour le CSV brut extrait) ne sont
  **pas inclus** — ils sont entièrement reproductibles en ré-exécutant les
  scripts sur le fichier `2008_to_2024.zip` original (également exclu du
  dépôt, voir `.gitignore`).
