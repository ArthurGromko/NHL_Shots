import argparse, glob, os, pickle
from datetime import date
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom
from compare_odds import recal_factor
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


def closing_fair(day):
    """Probabilité juste (vig retiré) d'Over à la fermeture, par match x bookmaker x joueur x ligne, ou None."""
    path = f"odds/closing_{day}.csv"
    if not os.path.exists(path):
        return None
    c = pd.read_csv(path)
    c = c.pivot_table(index=["gameId", "bookmaker", "player", "line"], columns="side",
                      values="price", aggfunc="last").reset_index().dropna(subset=["Over", "Under"])
    c["p_close_over"] = fair_over(c["Over"], c["Under"])
    return c


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
        c = closing_fair(day)
        if c is not None:
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


def movement(files):
    # Mouvement Pinnacle entre la capture de 13 h et la fermeture, sur toutes les lignes notées.
    # La CLV des picks est surtout la marge Pinnacle (les lignes bougent peu); ce qui compte est
    # le sens: si le marché va vers le modèle plus d'une fois sur deux, le modèle voit quelque
    # chose que le marché intègre ensuite. Point de décision vers le 1er nov.: ~50% -> arrêter
    # la capture de fermeture (elle double les crédits Odds API), > ~58% -> la garder.
    rows = []
    for f in files:
        c = closing_fair(f[-14:-4])
        if c is None:
            continue
        l = pd.read_csv(f)
        l = l[(l["bookmaker"] == "pinnacle") & ~l["void"].astype(bool)]
        rows.append(l.merge(c[c["bookmaker"] == "pinnacle"][["gameId", "player", "line", "p_close_over"]],
                            on=["gameId", "player", "line"]))
    if not rows:
        return
    d = pd.concat(rows)
    move = d["p_close_over"] - d["p_market_over"]
    moved = move.abs() > 0.005
    toward_model = (np.sign(move) == np.sign(d["p_model_over"] - d["p_market_over"]))[moved]
    print(f"\n=== Mouvement Pinnacle 13 h -> fermeture ({len(d)} lignes) ===")
    print(f"Mouvement moyen {100 * move.abs().mean():.2f} pt | {moved.mean():.0%} des lignes bougent de plus de 0,5 pt "
          f"| vers le modèle: {toward_model.mean():.0%} des {int(moved.sum())} lignes qui bougent")
    print(f"Brier Pinnacle 13 h {brier(d['p_market_over'], d['over']):.4f} | fermeture "
          f"{brier(d['p_close_over'], d['over']):.4f} | modèle {brier(d['p_model_over'], d['over']):.4f}")


def versions(lines):
    # Modèle brut vs recalibré (facteur glissant de compare_odds.py, recalculé ici pour les jours
    # d'avant son ajout) vs mélange modèle recalibré / marché vs marché. Le mélange est une
    # régression logistique sur les logits; chaque jour est prédit avec les poids appris sur les
    # autres jours. Il sert à suivre la précision, pas à parier: il colle trop au marché pour
    # dépasser la marge de Pinnacle.
    with open("prod_nb_params.pkl", "rb") as f:
        r = 1.0 / pickle.load(f)["alpha"]
    mp = pd.read_csv(f"extracted/{season_for(date.today())}.csv",
                     usecols=["playerId", "gameId", "situation", "icetime", "I_F_shotsOnGoal"])
    lines = lines.reset_index(drop=True)
    k = {d: recal_factor(d, mp)[0] for d in lines["day"].unique()}
    lines = lines.assign(p_recal=1 - nbinom.cdf(np.floor(lines["line"]), r, r / (r + lines["day"].map(k) * lines["mu"])))

    def logits(d):
        p = np.clip(d[["p_recal", "p_market_over"]].values, 1e-4, 1 - 1e-4)
        return sm.add_constant(np.log(p / (1 - p)), has_constant="add")
    lines["p_blend"] = np.nan
    for d in lines["day"].unique():
        train, test = lines["day"] != d, lines["day"] == d
        fit = sm.GLM(lines.loc[train, "over"], logits(lines[train]), family=sm.families.Binomial()).fit()
        lines.loc[test, "p_blend"] = fit.predict(logits(lines[test]))

    cols = {"modèle": "p_model_over", "recalibré": "p_recal", "mélange": "p_blend", "marché": "p_market_over"}
    print(f"\n=== Versions du modèle (facteur de recalibration au dernier jour noté: {k[max(k)]:.3f}) ===")
    for label, d in [("Toutes les lignes", lines), ("Pinnacle", lines[lines["bookmaker"] == "pinnacle"])]:
        print(f"{label:17s} ({len(d)}) Brier " + " | ".join(f"{n} {brier(d[c], d['over']):.4f}" for n, c in cols.items())
              + f"  | P(Over) réel {d['over'].mean():.3f}, recalibré {d['p_recal'].mean():.3f}")


def summary():
    files = sorted(glob.glob("graded/lines_*.csv"))
    if not files:
        return
    lines = pd.concat(pd.read_csv(f).assign(day=f[-14:-4]) for f in files)
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

    if len(lines["day"].unique()) > 1:
        versions(lines)
    movement(files)

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


def grade_bet365(force=False):
    # Cotes bet365 relevées à la main (add_bet365.py), notées une fois les matchs dans MoneyPuck.
    today = date.today().isoformat()
    for f in sorted(glob.glob("bet365/bet365_*.csv")):
        day = f[-14:-4]
        out = f"graded/bet365_{day}.csv"
        if day >= today or (os.path.exists(out) and not force):
            continue
        b = pd.read_csv(f)
        shots = actual_shots(day)
        if set(b["gameId"]) - set(shots["gameId"]):
            continue  # pas encore dans MoneyPuck
        b = b.merge(shots, on=["gameId", "playerId"], how="left")
        b["void"] = b["shots"].isna()
        b["over"] = (b["shots"] > b["line"]).astype(float).where(~b["void"])
        b.to_csv(out, index=False)
        print(f"{day}: {int((~b['void']).sum())} cotes bet365 notées")
        pick_path = f"picks/bet365_picks_{day}.csv"
        if os.path.exists(pick_path):
            p = pd.read_csv(pick_path).merge(shots, on=["gameId", "playerId"], how="left")
            p["result"] = np.where(p["shots"].isna(), "void", np.where(p["shots"] > p["line"], "win", "loss"))
            p["pnl"] = np.select([p["result"] == "win", p["result"] == "loss"],
                                 [p["stake"] * (p["price"] - 1), -p["stake"]], 0.0)
            p.to_csv(f"graded/bet365_picks_{day}.csv", index=False)


def bet365_summary():
    # Over seulement: la proba implicite bet365 (1/cote) contient la marge, son Brier est donc
    # pénalisé; la question utile est la colonne "vs Pinnacle" et le paper trading.
    files = sorted(glob.glob("graded/bet365_2*.csv"))
    if not files:
        return
    b = pd.concat(map(pd.read_csv, files))
    b = b[~b["void"].astype(bool)]
    print(f"\n=== bet365, Over X+ relevés à la main ({len(files)} jour(s), {len(b)} cotes joueur x ligne) ===")
    print(f"Brier modèle {brier(b['p_model_over'], b['over']):.4f} | bet365 implicite (marge incluse) "
          f"{brier(b['p_implied'], b['over']):.4f} | Over réel {b['over'].mean():.3f}, modèle {b['p_model_over'].mean():.3f}")
    both = b.dropna(subset=["p_pin_over"])
    if len(both):
        print(f"Lignes aussi cotées par Pinnacle ({len(both)}): Brier modèle {brier(both['p_model_over'], both['over']):.4f} "
              f"| Pinnacle {brier(both['p_pin_over'], both['over']):.4f} | bet365 implicite - Pinnacle juste "
              f"{100 * (both['p_implied'] - both['p_pin_over']).mean():+.1f} pt")

    def paper(label, bets):
        if not len(bets):
            print(f"  {label}: aucun pari")
            return
        pnl = np.where(bets["over"] == 1, bets["price"] - 1, -1.0)
        print(f"  {label}: {len(bets)} paris | {int(bets['over'].sum())} gagnés | {pnl.sum():+.2f} u | ROI {pnl.mean():+.1%}")
    # par palier: chaque joueur relevé a toute l'échelle 1+ à 5+, y compris des paliers que les
    # livres ne cotent pas en Over/Under (ex. 4+ d'un joueur à mu 2): teste la queue du modèle
    print("Par palier (X+ = Over X-0,5):")
    by = b.groupby("line").apply(lambda d: pd.Series({
        "n": len(d), "reel": d["over"].mean(), "modele": d["p_model_over"].mean(),
        "bet365": d["p_implied"].mean(), "brier_modele": brier(d["p_model_over"], d["over"]),
        "brier_bet365": brier(d["p_implied"], d["over"]),
    }), include_groups=False)
    by.index = [f"{int(l + 0.5)}+" for l in by.index]
    print(by.round(4).to_string())

    print("Paper trading au prix bet365 (1 u):")
    pf = sorted(glob.glob("graded/bet365_picks_*.csv"))
    if pf:
        bp = pd.concat(map(pd.read_csv, pf))
        for portfolio, g in bp.groupby("portfolio"):
            g = g[g["result"] != "void"]
            print(f"  {portfolio:13s} {len(g):4d} paris | {int((g['result'] == 'win').sum())} gagnés | "
                  f"{g['pnl'].sum():+.2f} u | ROI {g['pnl'].sum() / max(len(g), 1):+.1%}")
    paper("toute cote au-dessus du juste Pinnacle (non figé)", b[b["ev_pin"] > 0])

    # Picks Over des portefeuilles: prix Pinnacle figé vs meilleur des deux livres
    pick_files = [f"graded/picks_{f[-14:-4]}.csv" for f in files if os.path.exists(f"graded/picks_{f[-14:-4]}.csv")]
    if pick_files:
        p = pd.concat(map(pd.read_csv, pick_files))
        p = p[(p["side"] == "Over") & (p["result"] != "void")].merge(
            b[["gameId", "playerId", "line", "price"]].rename(columns={"price": "b365_price"}),
            on=["gameId", "playerId", "line"])
        if len(p):
            p["best_price"] = p[["price", "b365_price"]].max(axis=1)
            p["pnl_best"] = np.where(p["result"] == "win", p["best_price"] - 1, -1.0)
            for portfolio, g in p.groupby("portfolio"):
                print(f"  picks Over {portfolio} avec cote bet365 ({len(g)}): Pinnacle {g['pnl'].sum():+.2f} u "
                      f"| meilleur des deux {g['pnl_best'].sum():+.2f} u ({(g['b365_price'] > g['price']).mean():.0%} meilleurs chez bet365)")


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
    grade_bet365(args.force)
    summary()
    bet365_summary()


if __name__ == "__main__":
    main()
