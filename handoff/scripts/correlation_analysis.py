import pandas as pd
import numpy as np

df = pd.read_csv("features_full_v2.csv", low_memory=False)

feature_cols = [
    "games_played_before","days_rest",
    "shots_avg_last3","shots_avg_last5","shots_avg_last10","shots_avg_last20",
    "toi_avg_last3","toi_avg_last5","toi_avg_last10","toi_avg_last20",
    "shots_ewma","toi_ewma","shots_career_avg","shots_per60_avg_last10",
    "team_pace_shots_for_last5","team_pace_shots_for_last10","team_pace_shots_for_last20",
    "team_pace_attempts_for_last5","team_pace_attempts_for_last10","team_pace_attempts_for_last20",
    "team_defense_shots_against_last5","team_defense_shots_against_last10","team_defense_shots_against_last20",
    "team_defense_attempts_against_last5","team_defense_attempts_against_last10","team_defense_attempts_against_last20",
    "pp_toi_avg_last5","pp_toi_avg_last10","pp_shots_avg_last5","pp_shots_avg_last10",
    "pp_usage_rank_team","is_top5_pp_usage","is_pp1_likely",
    "target_shots"
]

corr = df[feature_cols].corr()

# Corrélation de chaque feature avec la cible
print("=== Corrélation avec target_shots (triée) ===")
print(corr["target_shots"].drop("target_shots").sort_values(ascending=False))

print("\n=== Paires de features avec |corr| > 0.90 (candidates à élaguer) ===")
seen = set()
pairs = []
for col in corr.columns:
    for row in corr.index:
        if col != row and abs(corr.loc[row, col]) > 0.90:
            pair = tuple(sorted([row, col]))
            if pair not in seen:
                seen.add(pair)
                pairs.append((pair[0], pair[1], corr.loc[row, col]))

pairs.sort(key=lambda x: -abs(x[2]))
for a, b, v in pairs:
    print(f"{a}  <->  {b} : {v:.3f}")
