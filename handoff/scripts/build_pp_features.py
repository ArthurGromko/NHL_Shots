import pandas as pd
import numpy as np

raw = pd.read_csv("reduced_shots.csv", low_memory=False)
raw["gameDate"] = pd.to_datetime(raw["gameDate"], format="%Y%m%d")

pp = raw[raw["situation"] == "5on4"][["playerId","gameId","playerTeam","gameDate","icetime","I_F_shotsOnGoal"]].copy()
pp = pp.rename(columns={"icetime":"pp_icetime", "I_F_shotsOnGoal":"pp_shots"})
pp["pp_toi_min"] = pp["pp_icetime"] / 60.0

pp = pp.sort_values(["playerId","gameDate"]).reset_index(drop=True)
g = pp.groupby("playerId")

pp_toi_shifted = g["pp_toi_min"].shift(1)
pp_shots_shifted = g["pp_shots"].shift(1)

def rolling_by_player(shifted_series, window, player_ids):
    return shifted_series.groupby(player_ids).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

for window in [5, 10]:
    pp[f"pp_toi_avg_last{window}"] = rolling_by_player(pp_toi_shifted, window, pp["playerId"])
    pp[f"pp_shots_avg_last{window}"] = rolling_by_player(pp_shots_shifted, window, pp["playerId"])

# --- Classement relatif dans l'équipe pour ce match (basé sur l'usage PP RÉCENT, donc pas de fuite) ---
# On classe les joueurs de la même équipe/même match selon leur pp_toi_avg_last10 entrant dans le match
pp["pp_usage_rank_team"] = pp.groupby(["gameId","playerTeam"])["pp_toi_avg_last10"] \
                              .rank(method="first", ascending=False)

# Flag simple: fait-il partie du "top" de l'unité PP de son équipe pour ce match (rang <= 5, soit l'unité PP1+PP2 combinée forwards+D)
pp["is_top5_pp_usage"] = (pp["pp_usage_rank_team"] <= 5).astype(int)
pp["is_pp1_likely"] = (pp["pp_usage_rank_team"] <= 2).astype(int)  # coeur de PP1 (souvent 2 attaquants clés)

pp_final = pp[["playerId","gameId","pp_toi_avg_last5","pp_toi_avg_last10",
               "pp_shots_avg_last5","pp_shots_avg_last10",
               "pp_usage_rank_team","is_top5_pp_usage","is_pp1_likely"]]

pp_final.to_csv("pp_features.csv", index=False)
print("Shape:", pp_final.shape)

# --- Fusion avec le dataset complet ---
features = pd.read_csv("features_full_v1.csv", low_memory=False)
features = features.merge(pp_final, on=["playerId","gameId"], how="left")
features.to_csv("features_full_v2.csv", index=False)
print("Shape finale avec PP:", features.shape)

print("\n--- Validation: McDavid (PP1 quasi-garanti) ---")
mcd = features[features["name"].str.contains("McDavid", na=False)].tail(6)
print(mcd[["gameDate","pp_toi_avg_last10","pp_usage_rank_team","is_pp1_likely","target_shots"]].to_string())

print("\n--- Comparaison: un joueur de 4e trio typique ---")
sample_low_toi = features[(features["toi_avg_last10"] < 10) & (features["games_played_before"] > 50)]
print(sample_low_toi[["name","pp_toi_avg_last10","pp_usage_rank_team","is_pp1_likely"]].head(5))
