import pandas as pd
import numpy as np
import json

# --- Feature set final (après élagage de la corrélation) ---
# pp_toi_avg_last10 retiré: corrélé à -0.91 avec pp_usage_rank_team, ce qui rendait
# son coefficient instable (signe contre-intuitif). pp_usage_rank_team conservé car
# légèrement plus corrélé à la cible et plus important en GBM.
RAW_NUMERIC_FEATURES = [
    "shots_avg_last5", "shots_career_avg",
    "toi_avg_last10",
    "pp_usage_rank_team",
    "shots_per60_avg_last10",
    "games_played_before", "days_rest",
    "team_pace_shots_for_last10", "team_defense_shots_against_last10",
]

# Seasons used for the research train/test split. Later seasons (the current one)
# only go into model_all_clean.csv, used by fit_production.py.
TRAIN_LAST_SEASON = 2023
TEST_SEASONS = [2024, 2025]


def make_model_frame(df, verbose=True):
    """Applique les transformations du modèle à features_full_v2.

    Utilisé à la fois pour l'entraînement (ici) et pour les prédictions du jour
    (predict_today.py), pour que les deux passent exactement par le même code.
    Retourne (df_model, final_features).
    """
    df = df.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"])

    # --- Imputation des NaN de contexte d'équipe (débuts de franchise) par la moyenne de saison ---
    for col in ["team_pace_shots_for_last10", "team_defense_shots_against_last10"]:
        season_mean = df.groupby("season")[col].transform("mean")
        df[col] = df[col].fillna(season_mean)

    # days_rest NaN pour le tout premier match d'un joueur -> imputer par la médiane
    df["days_rest"] = df["days_rest"].fillna(df["days_rest"].median())

    if verbose:
        print("NaN restants dans les features numériques:")
        print(df[RAW_NUMERIC_FEATURES].isna().sum())

    # --- Filtre des matchs "zéro": premier match de carrière, aucun historique disponible ---
    n_before = len(df)
    df = df[df["games_played_before"] > 0].copy()
    if verbose:
        print(f"\nMatchs zéro (games_played_before==0) filtrés: {n_before - len(df)} lignes")

    # --- Encodage catégoriel (référence: centre C, match à l'extérieur) ---
    # Colonnes construites explicitement pour qu'un petit échantillon (ex. les matchs
    # du jour) ait toujours les mêmes colonnes que l'entraînement.
    for pos in ["D", "L", "R"]:
        df[f"position_{pos}"] = (df["position"] == pos).astype(int)
    df["home_or_away_HOME"] = (df["home_or_away"] == "HOME").astype(int)
    dummy_cols = ["position_D", "position_L", "position_R", "home_or_away_HOME"]

    # --- Termes d'interaction avec position_D ---
    # La segmentation attaquants/défenseurs a montré des pentes différentes pour ces deux
    # features (TOI plus déterminant chez les attaquants, usage PP plus déterminant chez
    # les défenseurs). Plutôt que dupliquer tout le modèle, on laisse le modèle poolé
    # apprendre une pente additionnelle pour les défenseurs sur ces deux variables
    # (nulle par construction pour les attaquants, où position_D == 0).
    df["toi_avg_last10_x_D"] = df["toi_avg_last10"] * df["position_D"]
    df["pp_usage_rank_team_x_D"] = df["pp_usage_rank_team"] * df["position_D"]
    interaction_cols = ["toi_avg_last10_x_D", "pp_usage_rank_team_x_D"]

    # --- Log des features de volume de tirs ---
    # Avec un lien log, une feature brute en tirs fait croître mu exponentiellement :
    # les gros tireurs (extrêmes sur les trois features à la fois) étaient surestimés
    # d'environ 10% sur le top 1% des prédictions, alors que les GBM ne l'étaient pas.
    # En log, mu varie comme une puissance de la moyenne de tirs. +0.25 car un joueur
    # peut avoir 0 tir de moyenne sur ses 5 derniers matchs.
    log_shot_features = {}
    for col in ["shots_avg_last5", "shots_career_avg", "shots_per60_avg_last10"]:
        df[f"log_{col}"] = np.log(df[col] + 0.25)
        log_shot_features[col] = f"log_{col}"
    numeric_features = [log_shot_features.get(c, c) for c in RAW_NUMERIC_FEATURES]

    return df, numeric_features + dummy_cols + interaction_cols


if __name__ == "__main__":
    df = pd.read_csv("features_full_v2.csv", low_memory=False)

    # Matchs pas encore joués (lignes "à venir" ajoutées par update_live_data.py) : pas de cible
    n_unplayed = df["target_shots"].isna().sum()
    df = df[df["target_shots"].notna()]
    print(f"Matchs pas encore joués exclus: {n_unplayed} lignes\n")

    df_model, final_features = make_model_frame(df)

    # --- Split temporel: 2008-2023 = train, 2024-2025 = test ---
    train = df_model[df_model["season"] <= TRAIN_LAST_SEASON].copy()
    test = df_model[df_model["season"].isin(TEST_SEASONS)].copy()

    print(f"\nTrain: {len(train)} lignes ({train['season'].min()}-{train['season'].max()})")
    print(f"Test:  {len(test)} lignes ({test['season'].min()}-{test['season'].max()})")
    print(f"Tout (fit production): {len(df_model)} lignes ({df_model['season'].min()}-{df_model['season'].max()})")

    train.to_csv("model_train_clean.csv", index=False)
    test.to_csv("model_test_clean.csv", index=False)
    df_model.to_csv("model_all_clean.csv", index=False)

    with open("final_features.json", "w") as f:
        json.dump(final_features, f)
    print("\nFeatures finales retenues:", final_features)
