import argparse, json, pickle, re, unicodedata
import numpy as np
import pandas as pd
from scipy.stats import nbinom

# Compare les cotes capturées (fetch_odds.py) aux prédictions du jour (predict_today.py).
# Pour chaque (bookmaker, joueur, ligne): probabilité "juste" du marché (vig retiré en
# normalisant Over+Under), probabilité du modèle, edge et espérance de gain par mise de 1.
# -> odds/edges_<date>.csv


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

    path = f"odds/edges_{args.date}.csv"
    wide.sort_values("edge_over", ascending=False).to_csv(path, index=False)
    print(f"{len(wide)} lignes (bookmaker x joueur x ligne), {wide['playerId'].nunique()} joueurs -> {path}")
    print(f"Vig médian: {wide['vig'].median():.1%}\n")

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
