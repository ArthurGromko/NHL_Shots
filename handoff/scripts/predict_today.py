import json, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from prepare_model_data import make_model_frame

# Prédictions pour les matchs du jour (lignes "à venir" de update_live_data.py), avec
# le modèle de production (fit_production.py). À lancer après le pipeline de features
# exécuté avec `clean_reduce.py --upcoming`.

LINES = [0.5, 1.5, 2.5, 3.5, 4.5]

with open("upcoming_games.json") as f:
    upcoming = json.load(f)
game_ids = {g["gameId"] for g in upcoming["games"]}

df = pd.read_csv("features_full_v2.csv", low_memory=False)
is_upcoming = df["gameId"].isin(game_ids) & df["target_shots"].isna()
print(f"{is_upcoming.sum()} lignes à venir pour le {upcoming['date']} ({len(game_ids)} matchs)")

# Même transformation que l'entraînement, sur tout l'historique (imputations par saison),
# puis on ne garde que les matchs du jour
df_model, _ = make_model_frame(df, verbose=False)
today = df_model[df_model.index.isin(df.index[is_upcoming])].copy()
n_debut = is_upcoming.sum() - len(today)
if n_debut:
    print(f"{n_debut} joueur(s) sans historique NHL (premier match) non prédits")

with open("prod_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("prod_nb_params.pkl", "rb") as f:
    nb = pickle.load(f)

X = np.column_stack([np.ones(len(today)), scaler.transform(today[nb["features"]].values)])
today["mu"] = np.exp(np.clip(X @ nb["beta"], -20, 20))

r = 1.0 / nb["alpha"]
for line in LINES:
    today[f"p_over_{line}"] = 1 - nbinom.cdf(int(np.floor(line)), r, r / (r + today["mu"]))

cols = ["gameId", "gameDate", "name", "playerId", "playerTeam", "opposingTeam", "home_or_away", "position",
        "mu"] + [f"p_over_{l}" for l in LINES] + \
       ["shots_avg_last5", "shots_career_avg", "toi_avg_last10", "pp_usage_rank_team", "games_played_before"]
out = today[cols].sort_values("mu", ascending=False)

os.makedirs("predictions", exist_ok=True)
path = f"predictions/predictions_{upcoming['date']}.csv"
out.to_csv(path, index=False)
print(f"{len(out)} joueurs prédits -> {path}")
print(f"(modèle entraîné jusqu'au {nb['last_game_date']}, MoneyPuck à jour jusqu'au {upcoming['moneypuck_last_game']})\n")
print(out.head(15)[["name", "playerTeam", "opposingTeam", "mu", "p_over_2.5", "p_over_3.5"]].round(3).to_string(index=False))
