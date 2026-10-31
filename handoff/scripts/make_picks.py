import argparse, json, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Picks du jour pour le suivi "paper trading", figés au moment de l'exécution de 13 h.
# Règles fixées d'avance (ne pas les changer en cours de saison sans le noter):
# - seulement les lignes cotées par Pinnacle, au prix Pinnacle: un livre où l'utilisateur
#   parie vraiment, le plus "sharp", et pas de lignes alternatives cotées par un seul livre
#   (où le biais bas du modèle produisait des EV énormes et fausses). Pas de Pinnacle ce
#   jour-là = pas de picks.
# - EV >= 5% à ce prix, un seul pari par joueur (le meilleur), au plus 10 par portefeuille
# - mise fixe de 1 unité
# Deux portefeuilles: "brut" (probabilités du modèle telles quelles) et "ecart_retire" (modèle
# décalé de son écart moyen au marché sur la ligne, calculé sur le slate du jour).
# -> picks/picks_<date>.csv, jamais réécrit une fois créé.
# Changement du 2026-10-19 (picks à partir du 2026-10-20): plus d'Unders sur les lignes >= 3,5.
# Le modèle sous-estime les gros tireurs (README, point 7a; en live sur 15-18 oct.: 23,7% d'Over
# prédit vs 30,3% réel sur 3,5), ce qui produit des Unders à EV gonflée sur ces lignes. Mesure
# temporaire, à retirer quand une recalibration apprise sur les lignes notées sera en place.
# Ajout du 2026-10-30 (picks à partir du 2026-10-31): 3e portefeuille "recalibre", modèle
# recalibré (p_recal_over, voir compare_odds.py), sans la règle des Unders >= 3,5 puisque la
# recalibration corrige la sous-estimation. "brut" et "ecart_retire" restent inchangés pour
# pouvoir comparer les trois.

BOOK = "pinnacle"
MIN_EV = 0.05
MAX_PICKS = 10
NO_UNDER_FROM_LINE = 3.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="date (YYYY-MM-DD), défaut: celle de upcoming_games.json")
    args = parser.parse_args()
    if args.date is None:
        with open("upcoming_games.json") as f:
            args.date = json.load(f)["date"]

    path = f"picks/picks_{args.date}.csv"
    if os.path.exists(path):
        print(f"Picks déjà figés pour le {args.date} ({path}), rien à faire")
        return

    e = pd.read_csv(f"odds/edges_{args.date}.csv")
    odds_fetched_at = pd.read_csv(f"odds/odds_{args.date}.csv")["fetched_at"].max()

    # Écart moyen modèle - marché par ligne, sur tout le slate (pour le portefeuille "ecart_retire")
    gap_by_line = (e["p_model_over"] - e["p_market_over"]).groupby(e["line"]).mean()
    n_books = e.groupby(["gameId", "playerId", "line"])["bookmaker"].nunique().rename("n_books")

    g = e[e["bookmaker"] == BOOK].merge(n_books.reset_index(), on=["gameId", "playerId", "line"])
    if g.empty:
        print(f"Pas de lignes {BOOK} le {args.date}: pas de picks")
        return
    g["p_adj_over"] = (g["p_model_over"] - g["line"].map(gap_by_line)).clip(0.01, 0.99)

    picks = []
    for portfolio, p_col in [("brut", "p_model_over"), ("ecart_retire", "p_adj_over"), ("recalibre", "p_recal_over")]:
        sides = pd.concat([
            g.assign(side="Over", price=g["price_over"], p_model=g[p_col], p_market=g["p_market_over"]),
            g.assign(side="Under", price=g["price_under"], p_model=1 - g[p_col], p_market=1 - g["p_market_over"]),
        ])
        sides["ev"] = sides["p_model"] * sides["price"] - 1
        if portfolio != "recalibre":
            sides = sides[~((sides["side"] == "Under") & (sides["line"] >= NO_UNDER_FROM_LINE))]
        best = (sides[sides["ev"] >= MIN_EV].sort_values("ev", ascending=False)
                .drop_duplicates("playerId").head(MAX_PICKS))
        picks.append(best.assign(portfolio=portfolio))
        print(f"{portfolio}: {len(best)} pick(s)")

    out = pd.concat(picks)
    out["date"] = args.date
    out["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["odds_fetched_at"] = odds_fetched_at
    out["stake"] = 1.0
    out["bookmaker"] = BOOK
    cols = ["date", "portfolio", "gameId", "playerId", "name", "player", "playerTeam", "opposingTeam",
            "line", "side", "bookmaker", "price", "n_books", "mu", "p_model", "p_market", "ev", "stake",
            "odds_fetched_at", "frozen_at"]
    os.makedirs("picks", exist_ok=True)
    out[cols].to_csv(path, index=False)
    print(f"-> {path}\n")
    print(out[["portfolio", "name", "line", "side", "price", "p_model", "p_market", "ev"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
