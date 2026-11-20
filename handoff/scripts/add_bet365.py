import argparse, glob, json, os, pickle, sys
from datetime import date, datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from compare_odds import match_players, recal_factor

# Cotes bet365 relevées à la main (bet365 n'est pas dans The Odds API). Marché "X+ tirs au but":
# Over seulement, pas d'Under, donc pas de vig retirable par ligne: on garde la cote brute.
# Entrée: CSV "name,1,2,3,4,5" (cote du marché 1+, 2+, ... ; case vide = pas offert), par fichier
# ou sur stdin. "3+" = Over 2,5.
#   .venv/bin/python handoff/scripts/add_bet365.py --date 2026-11-18 --at 15:36 prix.csv
# -> bet365/bet365_<date>.csv (une ligne par joueur x ligne; une nouvelle saisie du même joueur x
#    ligne remplace l'ancienne), avec la probabilité du modèle recalibré et celle de Pinnacle
#    (vig retiré) au moment de la capture de 13 h, pour comparaison.
# Ajout du 2026-11-18. Noté par grade_results.py (graded/bet365_<date>.csv + résumé).
#
# Paper trading bet365 (ajout du 2026-11-18, picks à partir du 2026-11-18), mêmes règles que
# make_picks.py: probabilité "recalibre", EV >= 5 % au prix bet365, un pari par joueur, au plus
# 10 par jour et par portefeuille, mise 1 u, Over seulement (bet365 n'a pas d'Under).
# - "bet365": tous les paliers 1+ à 5+
# - "bet365_max3": paliers 1+ à 3+ seulement. Le modèle suppose plus de dispersion que la
#   réalité (r ajusté 22,7 vs 15,7 en prod), donc il surestime probablement les 4+/5+ des
#   joueurs moyens, que les livres ne cotent pas en Over 3,5+ et qui n'ont jamais été notés.
# Les picks sont figés à chaque saisie, parmi les cotes de cette saisie, jusqu'au plafond du jour,
# et jamais réécrits (picks/bet365_picks_<date>.csv). Une cote relevée après le début du match
# n'est pas éligible. Relever TOUS les joueurs d'un match, pas seulement ceux qui intéressent.

WORKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipeline_run")
THRESHOLDS = [1, 2, 3, 4, 5]
MIN_EV = 0.05
MAX_PICKS = 10
PORTFOLIOS = {"bet365": 4.5, "bet365_max3": 2.5}  # ligne maximale (Over)


def game_starts(day):
    """Heure de début (UTC) par gameId, depuis upcoming_games.json s'il est pour ce jour."""
    try:
        with open("upcoming_games.json") as f:
            u = json.load(f)
    except FileNotFoundError:
        return {}
    if u.get("date") != day:
        return {}
    return {g["gameId"]: pd.Timestamp(g["startTimeUTC"]) for g in u["games"]}


def freeze_picks(day, new):
    """Ajoute aux picks du jour les meilleurs paris des cotes `new`, sans toucher aux picks déjà figés."""
    path = f"picks/bet365_picks_{day}.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["portfolio", "playerId"])
    starts = game_starts(day)
    cap = pd.to_datetime(new["captured_at"]).dt.tz_localize(datetime.now().astimezone().tzinfo).dt.tz_convert("UTC")
    start = new["gameId"].map(starts)
    if start.isna().any():
        print("Heure de début inconnue pour certains matchs: pas de vérification 'avant le match' pour eux")
    eligible = new[~(cap >= start).fillna(False)]
    if len(eligible) < len(new):
        print(f"{len(new) - len(eligible)} cote(s) relevée(s) après le début du match: pas éligibles aux picks")
    added = []
    for portfolio, max_line in PORTFOLIOS.items():
        mine = old[old["portfolio"] == portfolio]
        room = MAX_PICKS - len(mine)
        c = eligible[(eligible["line"] <= max_line) & (eligible["ev_model"] >= MIN_EV)
                     & ~eligible["playerId"].isin(mine["playerId"])]
        best = c.sort_values("ev_model", ascending=False).drop_duplicates("playerId").head(max(room, 0))
        added.append(best.assign(portfolio=portfolio, side="Over", stake=1.0,
                                 frozen_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        print(f"{portfolio}: {len(best)} pick(s) ajouté(s), {len(mine) + len(best)}/{MAX_PICKS} pour le {day}")
    out = pd.concat([old] + added) if len(old) else pd.concat(added)
    os.makedirs("picks", exist_ok=True)
    out.to_csv(path, index=False)
    new_picks = pd.concat(added)
    if len(new_picks):
        print(new_picks[["portfolio", "name", "line", "price", "p_model_over", "p_pin_over", "ev_model"]]
              .round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="CSV name,1,2,3,4,5 (défaut: stdin)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--at", help="heure du relevé, HH:MM (le jour de --date) ou YYYY-MM-DDTHH:MM (défaut: maintenant)")
    args = parser.parse_args()
    raw = pd.read_csv(args.input or sys.stdin, dtype={"name": str})
    os.chdir(WORKDIR)

    preds = pd.read_csv(f"predictions/predictions_{args.date}.csv")
    mapping = match_players(raw["name"].unique(), preds)
    missing = sorted(set(raw["name"]) - set(mapping))
    if missing:
        print(f"Pas de prédiction (ignorés): {', '.join(missing)}")

    with open("prod_nb_params.pkl", "rb") as f:
        r = 1.0 / pickle.load(f)["alpha"]
    edges_path = f"odds/edges_{args.date}.csv"
    edges = pd.read_csv(edges_path) if os.path.exists(edges_path) else None
    k = edges["recal_factor"].iloc[0] if edges is not None else recal_factor(args.date)[0]

    at = args.at or datetime.now().strftime("%Y-%m-%dT%H:%M")
    captured_at = at if "T" in at else f"{args.date}T{at}"
    rows = []
    for _, x in raw[raw["name"].isin(mapping)].iterrows():
        pid = mapping[x["name"]]
        pr = preds[preds["playerId"] == pid].iloc[0]
        for t in THRESHOLDS:
            price = pd.to_numeric(x.get(str(t)), errors="coerce")
            if pd.isna(price):
                continue
            rows.append({"date": args.date, "captured_at": captured_at, "gameId": pr["gameId"],
                         "playerId": pid, "name": pr["name"], "playerTeam": pr["playerTeam"],
                         "line": t - 0.5, "price": float(price), "mu": pr["mu"], "recal_factor": k})
    new = pd.DataFrame(rows)
    if new.empty:
        print("Rien à ajouter")
        return
    new["p_model_over"] = 1 - nbinom.cdf(np.floor(new["line"]), r, r / (r + k * new["mu"]))
    new["p_implied"] = 1 / new["price"]
    if edges is not None:
        pin = edges[edges["bookmaker"] == "pinnacle"][["playerId", "line", "p_market_over", "price_over"]]
        new = new.merge(pin.rename(columns={"p_market_over": "p_pin_over", "price_over": "pin_price_over"}),
                        on=["playerId", "line"], how="left")
    else:
        new["p_pin_over"], new["pin_price_over"] = np.nan, np.nan
    new["ev_model"] = new["p_model_over"] * new["price"] - 1
    new["ev_pin"] = new["p_pin_over"] * new["price"] - 1

    os.makedirs("bet365", exist_ok=True)
    path = f"bet365/bet365_{args.date}.csv"
    batch = new
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates(["playerId", "line"], keep="last")
    new.to_csv(path, index=False)
    print(f"{len(rows)} cote(s) ajoutée(s), {len(new)} au total -> {path}  (recalibration mu x {k:.3f})\n")
    freeze_picks(args.date, batch)
    print()
    show = batch[(batch["ev_model"] >= 0.05) | (batch["ev_pin"] > 0)].sort_values("ev_model", ascending=False)
    if len(show):
        print("EV >= 5 % selon le modèle, ou cote au-dessus du juste Pinnacle:")
        print(show[["name", "line", "price", "p_model_over", "p_pin_over", "pin_price_over", "ev_model", "ev_pin"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
