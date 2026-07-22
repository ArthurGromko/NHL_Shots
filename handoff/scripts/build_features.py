import pandas as pd
import numpy as np

df = pd.read_csv("reduced_shots_2008_2024.csv", low_memory=False)

# On travaille sur la situation "all" = agrégat complet du match par joueur
df = df[df["situation"] == "all"].copy()
df = df[df["icetime"] > 0]  # exclure les scratchs/DNP

df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")
df["toi_min"] = df["icetime"] / 60.0
df["shots_per60"] = df["I_F_shotsOnGoal"] / (df["icetime"] / 3600.0)

# Trier par joueur puis par date -- essentiel avant tout rolling
df = df.sort_values(["playerId", "gameDate"]).reset_index(drop=True)

g = df.groupby("playerId")

# IMPORTANT: on shift(1) AVANT le rolling pour ne jamais inclure le match courant
# dans ses propres features (sinon fuite de données -> le modèle "trichera")
shots_shifted = g["I_F_shotsOnGoal"].shift(1)
toi_shifted = g["toi_min"].shift(1)

def rolling_by_player(shifted_series, window, player_ids):
    return shifted_series.groupby(player_ids).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

for window in [3, 5, 10, 20]:
    df[f"shots_avg_last{window}"] = rolling_by_player(shots_shifted, window, df["playerId"])
    df[f"toi_avg_last{window}"] = rolling_by_player(toi_shifted, window, df["playerId"])

# Moyenne mobile exponentielle (plus de poids aux matchs récents)
df["shots_ewma"] = shots_shifted.groupby(df["playerId"]).transform(lambda x: x.ewm(span=10, min_periods=1).mean())
df["toi_ewma"] = toi_shifted.groupby(df["playerId"]).transform(lambda x: x.ewm(span=10, min_periods=1).mean())

# Moyenne de carrière jusqu'à ce match (expanding, exclut le match courant)
df["shots_career_avg"] = shots_shifted.groupby(df["playerId"]).transform(lambda x: x.expanding(min_periods=1).mean())
df["games_played_before"] = g.cumcount()  # nb de matchs joués AVANT celui-ci (0 pour le 1er match)

# Taux de tirs par 60 min (moyenne mobile, plus stable que le compte brut car normalise le TOI)
shots_per60_shifted = g["shots_per60"].shift(1)
df["shots_per60_avg_last10"] = rolling_by_player(shots_per60_shifted, 10, df["playerId"])

# Repos entre les matchs
df["days_rest"] = g["gameDate"].diff().dt.days

# Variable cible = ce qu'on veut prédire
df["target_shots"] = df["I_F_shotsOnGoal"]

feature_cols = [
    "playerId","name","gameId","season","playerTeam","opposingTeam","home_or_away",
    "gameDate","position","games_played_before","days_rest",
    "shots_avg_last3","shots_avg_last5","shots_avg_last10","shots_avg_last20",
    "toi_avg_last3","toi_avg_last5","toi_avg_last10","toi_avg_last20",
    "shots_ewma","toi_ewma","shots_career_avg","shots_per60_avg_last10",
    "target_shots","toi_min"
]

df_features = df[feature_cols].copy()
df_features.to_csv("features_base.csv", index=False)
print("Shape finale:", df_features.shape)
print("\nAperçu Connor McDavid (validation manuelle anti-fuite):")
mcd = df_features[df_features["name"].str.contains("McDavid", na=False)].tail(8)
print(mcd[["gameDate","games_played_before","shots_avg_last5","shots_ewma","toi_avg_last5","target_shots"]].to_string())
