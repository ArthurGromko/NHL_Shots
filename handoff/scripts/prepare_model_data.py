import pandas as pd
import numpy as np

df = pd.read_csv("features_full_v2.csv", low_memory=False)
df["gameDate"] = pd.to_datetime(df["gameDate"])

# --- Feature set final (après élagage de la corrélation) ---
# pp_toi_avg_last10 retiré: corrélé à -0.91 avec pp_usage_rank_team, ce qui rendait
# son coefficient instable (signe contre-intuitif). pp_usage_rank_team conservé car
# légèrement plus corrélé à la cible et plus important en GBM.
numeric_features = [
    "shots_avg_last5", "shots_career_avg",
    "toi_avg_last10",
    "pp_usage_rank_team",
    "shots_per60_avg_last10",
    "games_played_before", "days_rest",
    "team_pace_shots_for_last10", "team_defense_shots_against_last10",
]
categorical_features = ["position", "home_or_away"]

# --- Imputation des 630 NaN de contexte d'équipe (débuts de franchise) par la moyenne de saison ---
for col in ["team_pace_shots_for_last10", "team_defense_shots_against_last10"]:
    season_mean = df.groupby("season")[col].transform("mean")
    df[col] = df[col].fillna(season_mean)

# days_rest NaN pour le tout premier match d'un joueur -> imputer par la médiane
df["days_rest"] = df["days_rest"].fillna(df["days_rest"].median())

print("NaN restants dans les features numériques:")
print(df[numeric_features].isna().sum())

# --- Filtre des matchs "zéro": premier match de carrière, aucun historique disponible ---
n_before = len(df)
df = df[df["games_played_before"] > 0].copy()
print(f"\nMatchs zéro (games_played_before==0) filtrés: {n_before - len(df)} lignes")

# --- Encodage catégoriel ---
df_model = pd.get_dummies(df, columns=categorical_features, drop_first=True)
dummy_cols = [c for c in df_model.columns if c.startswith("position_") or c.startswith("home_or_away_")]
print("\nColonnes dummy créées:", dummy_cols)

# --- Termes d'interaction avec position_D ---
# La segmentation attaquants/défenseurs a montré des pentes différentes pour ces deux
# features (TOI plus déterminant chez les attaquants, usage PP plus déterminant chez
# les défenseurs). Plutôt que dupliquer tout le modèle, on laisse le modèle poolé
# apprendre une pente additionnelle pour les défenseurs sur ces deux variables
# (nulle par construction pour les attaquants, où position_D == 0).
df_model["toi_avg_last10_x_D"] = df_model["toi_avg_last10"] * df_model["position_D"]
df_model["pp_usage_rank_team_x_D"] = df_model["pp_usage_rank_team"] * df_model["position_D"]
interaction_cols = ["toi_avg_last10_x_D", "pp_usage_rank_team_x_D"]

final_features = numeric_features + dummy_cols + interaction_cols

# --- Split temporel: 2008-2022 = train, 2023-2024 = test ---
train = df_model[df_model["season"] <= 2022].copy()
test = df_model[df_model["season"] >= 2023].copy()

print(f"\nTrain: {len(train)} lignes ({train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test)} lignes ({test['season'].min()}-{test['season'].max()})")

train.to_csv("model_train_clean.csv", index=False)
test.to_csv("model_test_clean.csv", index=False)

import json
with open("final_features.json", "w") as f:
    json.dump(final_features, f)
print("\nFeatures finales retenues:", final_features)
