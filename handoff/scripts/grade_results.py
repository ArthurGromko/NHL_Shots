import argparse, glob, os
from datetime import date
import numpy as np
import pandas as pd
from update_live_data import season_for

# Note les jours capturés une fois leurs matchs présents dans MoneyPuck (le lendemain midi).
# 1) Toutes les lignes capturées (odds/edges_<date>.csv): modèle vs marché, Brier et log loss.
#    C'est le test principal: chaque ligne compte, pas seulement les picks.
# 2) Picks (picks/picks_<date>.csv): CLV vs cotes de fermeture (odds/closing_<date>.csv),
#    puis gain/perte à mise fixe. Un joueur qui ne joue pas = pari annulé (règle des books).
# -> graded/lines_<date>.csv, graded/picks_<date>.csv, puis un résumé cumulé de tous les jours notés.


def actual_shots(day):
    """Tirs réels par (gameId, playerId) pour les matchs d'une date, ou None si MoneyPuck n'a pas tout."""
    mp = pd.read_csv(f"extracted/{season_for(date.fromisoformat(day))}.csv",
                     usecols=["playerId", "gameId", "gameDate", "situation", "icetime", "I_F_shotsOnGoal"])
    mp = mp[(mp["situation"] == "all") & (mp["gameDate"] == int(day.replace("-", "")))]
    return mp[mp["icetime"] > 0][["gameId", "playerId", "I_F_shotsOnGoal"]].rename(columns={"I_F_shotsOnGoal": "shots"})


def fair_over(price_over, price_under):
    inv_o, inv_u = 1 / price_over, 1 / price_under
    return inv_o / (inv_o + inv_u)


def grade_day(day):
    edges = pd.read_csv(f"odds/edges_{day}.csv")
    shots = actual_shots(day)
    missing = set(edges["gameId"]) - set(shots["gameId"])
    if missing:
        print(f"{day}: {len(missing)} match(s) pas encore dans MoneyPuck, notation reportée")
        return False

    # --- 1) toutes les lignes ---
    lines = edges.merge(shots, on=["gameId", "playerId"], how="left")
    lines["void"] = lines["shots"].isna()
    lines["over"] = (lines["shots"] > lines["line"]).astype(float).where(~lines["void"])
    lines.to_csv(f"graded/lines_{day}.csv", index=False)

    # --- 2) picks ---
    pick_path = f"picks/picks_{day}.csv"
    if os.path.exists(pick_path):
        picks = pd.read_csv(pick_path).merge(shots, on=["gameId", "playerId"], how="left")
        picks["void"] = picks["shots"].isna()
        won = np.where(picks["side"] == "Over", picks["shots"] > picks["line"], picks["shots"] < picks["line"])
        picks["result"] = np.where(picks["void"], "void", np.where(won, "win", "loss"))
        picks["pnl"] = np.select([picks["result"] == "win", picks["result"] == "loss"],
                                 [picks["stake"] * (picks["price"] - 1), -picks["stake"]], 0.0)

        # CLV: probabilité juste de fermeture (vig retiré) sur la même ligne: celle de Pinnacle
        # si elle cote la ligne à la fermeture (référence sharp), sinon la moyenne des bookmakers
        closing_path = f"odds/closing_{day}.csv"
        if os.path.exists(closing_path):
            c = pd.read_csv(closing_path)
            c = c.pivot_table(index=["gameId", "bookmaker", "player", "line"], columns="side",
                              values="price", aggfunc="last").reset_index().dropna(subset=["Over", "Under"])
            c["p_close_over"] = fair_over(c["Over"], c["Under"])
            keys = ["gameId", "player", "line"]
            pin = c[c["bookmaker"] == "pinnacle"].groupby(keys)["p_close_over"].mean().rename("p_close_pin")
            avg = c.groupby(keys)["p_close_over"].mean().rename("p_close_avg")
            c = pd.concat([pin, avg], axis=1).reset_index()
            c["p_close_over"] = c["p_close_pin"].fillna(c["p_close_avg"])
            c["clv_source"] = np.where(c["p_close_pin"].notna(), "pinnacle", "moyenne")
            picks = picks.merge(c[keys + ["p_close_over", "clv_source"]], on=keys, how="left")
            p_close_side = np.where(picks["side"] == "Over", picks["p_close_over"], 1 - picks["p_close_over"])
            # EV du prix pris, évaluée avec la probabilité de fermeture: > 0 = on a battu la fermeture
            picks["clv"] = picks["price"] * p_close_side - 1
        else:
            picks["p_close_over"] = np.nan
            picks["clv"] = np.nan
        picks.to_csv(f"graded/picks_{day}.csv", index=False)

    n_void = int(lines["void"].sum())
    print(f"{day}: {len(lines) - n_void} lignes notées ({n_void} annulées)")
    return True


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summary():
    files = sorted(glob.glob("graded/lines_*.csv"))
    if not files:
        return
    lines = pd.concat(map(pd.read_csv, files))
    lines = lines[~lines["void"].astype(bool)]
    y = lines["over"].values
    days = f"{files[0][-14:-4]} → {files[-1][-14:-4]}"
    print(f"\n=== Toutes les lignes ({days}, {len(lines)} lignes bookmaker x joueur x ligne) ===")
    print(f"Taux réel d'Over: {y.mean():.3f} | P(Over) moyenne modèle: {lines['p_model_over'].mean():.3f} "
          f"| marché: {lines['p_market_over'].mean():.3f}")
    print(f"Brier    modèle {brier(lines['p_model_over'], y):.4f} | marché {brier(lines['p_market_over'], y):.4f}  (plus bas = mieux)")
    print(f"Log loss modèle {log_loss(lines['p_model_over'], y):.4f} | marché {log_loss(lines['p_market_over'], y):.4f}")
    pin = lines[lines["bookmaker"] == "pinnacle"]
    if len(pin):
        print(f"Lignes Pinnacle seulement ({len(pin)}): Brier modèle {brier(pin['p_model_over'], pin['over']):.4f} "
              f"| Pinnacle {brier(pin['p_market_over'], pin['over']):.4f}")
    # par ligne: une seule rangée par joueur x match x ligne (marché = moyenne des bookmakers), sinon
    # un joueur coté par 6 livres pèse 6 fois et quelques gros tireurs faussent les lignes hautes
    per_line = (lines.groupby(["gameId", "playerId", "line"])
                .agg(over=("over", "first"), p_model_over=("p_model_over", "first"),
                     p_market_over=("p_market_over", "mean")).reset_index())
    print(f"Par ligne ({len(per_line)} joueur x match x ligne, marché = moyenne des bookmakers):")
    by_line = per_line.groupby("line").apply(lambda d: pd.Series({
        "n": len(d), "reel": d["over"].mean(), "modele": d["p_model_over"].mean(),
        "marche": d["p_market_over"].mean(),
        "brier_modele": brier(d["p_model_over"], d["over"]), "brier_marche": brier(d["p_market_over"], d["over"]),
    }), include_groups=False)
    print(by_line.round(4).to_string())

    pick_files = sorted(glob.glob("graded/picks_*.csv"))
    if not pick_files:
        return
    picks = pd.concat(map(pd.read_csv, pick_files))
    print(f"\n=== Picks (paper trading, mise fixe 1 u) ===")
    for portfolio, p in picks.groupby("portfolio"):
        settled = p[p["result"] != "void"]
        clv = p["clv"].dropna()
        print(f"{portfolio:13s} {len(settled):4d} paris | {int((settled['result'] == 'win').sum())} gagnés "
              f"| {settled['pnl'].sum():+.2f} u | ROI {settled['pnl'].sum() / max(len(settled), 1):+.1%} "
              f"| CLV moyenne {clv.mean():+.1%} sur {len(clv)} ({(clv > 0).mean():.0%} positives)"
              if len(clv) else
              f"{portfolio:13s} {len(settled):4d} paris | {settled['pnl'].sum():+.2f} u | pas de cotes de fermeture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="noter une date précise (sinon: toutes les dates capturées pas encore notées)")
    parser.add_argument("--force", action="store_true", help="re-noter même si déjà fait")
    args = parser.parse_args()
    os.makedirs("graded", exist_ok=True)

    if args.date:
        days = [args.date]
    else:
        days = sorted(f[-14:-4] for f in glob.glob("odds/edges_*.csv"))
    today = date.today().isoformat()
    for day in days:
        if day >= today:
            continue  # matchs pas encore joués
        if os.path.exists(f"graded/lines_{day}.csv") and not args.force:
            continue
        grade_day(day)
    summary()


if __name__ == "__main__":
    main()
