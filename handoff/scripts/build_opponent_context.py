import pandas as pd
import numpy as np

df = pd.read_csv("reduced_shots_2008_2024.csv", low_memory=False)
df = df[df["situation"] == "all"].copy()
df = df[df["icetime"] > 0]
df["gameDate"] = pd.to_datetime(df["gameDate"], format="%Y%m%d")

# --- Étape 1: agréger au niveau équipe-match ---
team_game = (
    df.groupby(["gameId", "playerTeam", "opposingTeam", "season", "gameDate", "home_or_away"])
    .agg(team_shots_for=("I_F_shotsOnGoal", "sum"),
         team_attempts_for=("I_F_shotAttempts", "sum"))
    .reset_index()
)
print("Team-game shape:", team_game.shape)  # attendu ~= 2 x nb de matchs

# --- Étape 2: obtenir les tirs CONCÉDÉS par équipe-match ---
# = les tirs produits par l'adversaire dans ce même match
self_merge = team_game.merge(
    team_game[["gameId", "playerTeam", "team_shots_for", "team_attempts_for"]],
    left_on=["gameId", "opposingTeam"],
    right_on=["gameId", "playerTeam"],
    suffixes=("", "_opp")
)
self_merge["team_shots_against"] = self_merge["team_shots_for_opp"]
self_merge["team_attempts_against"] = self_merge["team_attempts_for_opp"]
team_game = self_merge[["gameId","playerTeam","opposingTeam","season","gameDate","home_or_away",
                         "team_shots_for","team_attempts_for","team_shots_against","team_attempts_against"]].copy()

print("Après self-merge:", team_game.shape)
print(team_game.head(4))

# --- Étape 3: trier par équipe puis par date, calculer les moyennes mobiles DÉCALÉES ---
team_game = team_game.sort_values(["playerTeam", "gameDate"]).reset_index(drop=True)
g = team_game.groupby("playerTeam")

shots_for_shifted = g["team_shots_for"].shift(1)
shots_against_shifted = g["team_shots_against"].shift(1)
attempts_for_shifted = g["team_attempts_for"].shift(1)
attempts_against_shifted = g["team_attempts_against"].shift(1)

def rolling_by_team(shifted_series, window, team_ids):
    return shifted_series.groupby(team_ids).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)

for window in [5, 10, 20]:
    team_game[f"team_pace_shots_for_last{window}"] = rolling_by_team(shots_for_shifted, window, team_game["playerTeam"])
    team_game[f"team_defense_shots_against_last{window}"] = rolling_by_team(shots_against_shifted, window, team_game["playerTeam"])
    team_game[f"team_pace_attempts_for_last{window}"] = rolling_by_team(attempts_for_shifted, window, team_game["playerTeam"])
    team_game[f"team_defense_attempts_against_last{window}"] = rolling_by_team(attempts_against_shifted, window, team_game["playerTeam"])

team_game.to_csv("team_game_context.csv", index=False)
print("\nAperçu final team_game_context:")
print(team_game[["gameDate","playerTeam","opposingTeam","team_shots_for","team_shots_against",
                  "team_pace_shots_for_last10","team_defense_shots_against_last10"]].tail(6))
