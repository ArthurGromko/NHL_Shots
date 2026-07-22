import pandas as pd

features = pd.read_csv("features_base.csv", low_memory=False)
team_ctx = pd.read_csv("team_game_context.csv", low_memory=False)

pace_cols = ["team_pace_shots_for_last5","team_pace_shots_for_last10","team_pace_shots_for_last20",
             "team_pace_attempts_for_last5","team_pace_attempts_for_last10","team_pace_attempts_for_last20"]
defense_cols = ["team_defense_shots_against_last5","team_defense_shots_against_last10","team_defense_shots_against_last20",
                "team_defense_attempts_against_last5","team_defense_attempts_against_last10","team_defense_attempts_against_last20"]

# 1) Rattacher le PACE de la propre équipe du joueur
own_team = team_ctx[["gameId","playerTeam"] + pace_cols].copy()
features = features.merge(own_team, on=["gameId","playerTeam"], how="left")

# 2) Rattacher la DÉFENSE de l'adversaire
#    -> on cherche, pour ce gameId, la ligne team_game où playerTeam == opposingTeam du joueur
opp_defense = team_ctx[["gameId","playerTeam"] + defense_cols].copy()
opp_defense = opp_defense.rename(columns={"playerTeam": "opposingTeam"})
features = features.merge(opp_defense, on=["gameId","opposingTeam"], how="left")

print("Shape finale avec contexte adverse:", features.shape)
print("Valeurs manquantes (contexte adverse):")
print(features[pace_cols + defense_cols].isna().sum())

features.to_csv("features_full_v1.csv", index=False)

print("\n--- Validation: McDavid vs contexte adverse ---")
mcd = features[features["name"].str.contains("McDavid", na=False)].tail(6)
print(mcd[["gameDate","opposingTeam","shots_avg_last10",
           "team_pace_shots_for_last10","team_defense_shots_against_last10","target_shots"]].to_string())
