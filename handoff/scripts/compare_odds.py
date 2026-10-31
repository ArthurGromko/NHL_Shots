import argparse, glob, json, pickle, re, unicodedata
from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from update_live_data import season_for

# Compare les cotes capturées (fetch_odds.py) aux prédictions du jour (predict_today.py).
# Pour chaque (bookmaker, joueur, ligne): probabilité "juste" du marché (vig retiré en
# normalisant Over+Under), probabilité du modèle, edge et espérance de gain par mise de 1.
# -> odds/edges_<date>.csv
#
# p_recal_over (ajout du 2026-10-30): modèle recalibré, mu multiplié par le ratio tirs réels /
# mu prédit sur les RECAL_DAYS jours précédents (tous les joueurs prédits, pas seulement les
# lignes cotées). En live, le modèle sous-estimait tout le monde d'environ 4% (6,6% sur
# 15-22 oct., 1,7% sur 23-29 oct.), sans dépendance au niveau de mu (log mu' = a + b log mu
# donne b = 1,01): un simple facteur suffit, et une fenêtre glissante suit sa décroissance.

RECAL_DAYS = 14
RECAL_MIN_ROWS = 500  # en dessous (début de saison), pas de correction


def recal_factor(day, mp=None):
    """(facteur, n joueur-matchs) appris sur les RECAL_DAYS jours avant `day`, sans regarder `day`."""
    start = (date.fromisoformat(day) - timedelta(days=RECAL_DAYS)).isoformat()
    files = [f for f in glob.glob("predictions/predictions_*.csv") if start <= f[-14:-4] < day]
    if not files:
        return 1.0, 0
    if mp is None:
        mp = pd.read_csv(f"extracted/{season_for(date.fromisoformat(day))}.csv",
                         usecols=["playerId", "gameId", "situation", "icetime", "I_F_shotsOnGoal"])
    mp = mp[(mp["situation"] == "all") & (mp["icetime"] > 0)]
    d = pd.concat(map(pd.read_csv, files)).merge(mp, on=["gameId", "playerId"])
    if len(d) < RECAL_MIN_ROWS:
        return 1.0, len(d)
    return d["I_F_shotsOnGoal"].sum() / d["mu"].sum(), len(d)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def match_players(odds_players, preds_game):
    """Associe les noms de l'Odds API aux joueurs prédits d'un même match.
    Nom complet normalisé d'abord, puis nom de famille + initiale (Mitch/Mitchell...).
    Homonymes dans un même match (ex. deux Elias Pettersson à VAN): on garde le joueur
    avec le plus de temps de glace, les bookmakers ne cotant que les réguliers."""
    preds_game = preds_game.sort_values("toi_avg_last10")  # le plus gros TOI écrase les autres
    full = {norm(n): pid for n, pid in zip(preds_game["name"], preds_game["playerId"])}
    by_last = {}
    for n, pid in zip(preds_game[::-1]["name"], preds_game[::-1]["playerId"]):
        parts = norm(n).split()
        by_last.setdefault((parts[0][:1], parts[-1]), []).append(pid)
    out = {}
    for p in odds_players:
        n = norm(p)
        if n in full:
            out[p] = full[n]
            continue
        parts = n.split()
        cands = by_last.get((parts[0][:1], parts[-1]), []) if parts else []
        if cands:
            out[p] = cands[0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="date (YYYY-MM-DD), défaut: celle de upcoming_games.json")
    args = parser.parse_args()
    if args.date is None:
        with open("upcoming_games.json") as f:
            args.date = json.load(f)["date"]

    preds = pd.read_csv(f"predictions/predictions_{args.date}.csv")
    odds = pd.read_csv(f"odds/odds_{args.date}.csv")
    # dernière capture par match si fetch_odds.py a tourné plusieurs fois
    odds = odds[odds["fetched_at"] == odds.groupby("gameId")["fetched_at"].transform("max")]
    with open("prod_nb_params.pkl", "rb") as f:
        r = 1.0 / pickle.load(f)["alpha"]

    mapping, unmatched = {}, set()
    for gid, g in odds.groupby("gameId"):
        players = g["player"].dropna().unique()
        m = match_players(players, preds[preds["gameId"] == gid])
        mapping.update({(gid, p): pid for p, pid in m.items()})
        unmatched.update(p for p in players if p not in m)
    odds["playerId"] = [mapping.get((gid, p)) for gid, p in zip(odds["gameId"], odds["player"])]
    if unmatched:
        print(f"{len(unmatched)} joueur(s) des cotes sans prédiction (premier match, nom différent, "
              f"ou absent du roster actif): {', '.join(sorted(unmatched))}\n")

    # Over et Under côte à côte, par bookmaker / joueur / ligne
    wide = odds.dropna(subset=["playerId"]).pivot_table(
        index=["gameId", "bookmaker", "player", "playerId", "line"], columns="side", values="price", aggfunc="first"
    ).reset_index().rename(columns={"Over": "price_over", "Under": "price_under"})
    wide = wide.dropna(subset=["price_over", "price_under"])
    wide["playerId"] = wide["playerId"].astype(int)
    wide = wide.merge(preds[["gameId", "playerId", "name", "playerTeam", "opposingTeam", "mu"]],
                      on=["gameId", "playerId"])

    inv_o, inv_u = 1 / wide["price_over"], 1 / wide["price_under"]
    wide["vig"] = inv_o + inv_u - 1
    wide["p_market_over"] = inv_o / (inv_o + inv_u)
    wide["p_model_over"] = 1 - nbinom.cdf(np.floor(wide["line"]), r, r / (r + wide["mu"]))
    wide["edge_over"] = wide["p_model_over"] - wide["p_market_over"]
    wide["ev_over"] = wide["p_model_over"] * wide["price_over"] - 1
    wide["ev_under"] = (1 - wide["p_model_over"]) * wide["price_under"] - 1
    k, n_recal = recal_factor(args.date)
    wide["recal_factor"] = k
    wide["p_recal_over"] = 1 - nbinom.cdf(np.floor(wide["line"]), r, r / (r + k * wide["mu"]))

    path = f"odds/edges_{args.date}.csv"
    wide.sort_values("edge_over", ascending=False).to_csv(path, index=False)
    print(f"{len(wide)} lignes (bookmaker x joueur x ligne), {wide['playerId'].nunique()} joueurs -> {path}")
    print(f"Vig médian: {wide['vig'].median():.1%}")
    print(f"Recalibration: mu x {k:.3f} ({n_recal} joueur-matchs sur les {RECAL_DAYS} jours précédents)\n")

    # Meilleure espérance par joueur/ligne/côté, tous bookmakers confondus
    best = pd.concat([
        wide.assign(side="Over", price=wide["price_over"], ev=wide["ev_over"]),
        wide.assign(side="Under", price=wide["price_under"], ev=wide["ev_under"]),
    ]).sort_values("ev", ascending=False).drop_duplicates(["playerId", "line", "side"])
    cols = ["name", "playerTeam", "opposingTeam", "line", "side", "bookmaker", "price",
            "mu", "p_model_over", "p_market_over", "ev"]
    print("Meilleures espérances (EV par mise de 1):")
    print(best[cols].head(20).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
