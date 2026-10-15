import argparse, io, json, ssl, urllib.request, zipfile
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import certifi

# Mise à jour quotidienne des données:
# 1) télécharge le fichier MoneyPuck de la saison en cours -> extracted/<saison>.csv
# 2) écrit des lignes "à venir" pour les matchs du jour -> upcoming_rows.csv
#    (hors de extracted/: seul `clean_reduce.py --upcoming` les inclut, pour qu'un
#    vieux fichier ne se retrouve jamais dans l'historique d'entraînement)
#    (une ligne 'all' + une ligne '5on4' par joueur, stats de tirs vides). Comme toutes
#    les features sont décalées (shift(1)), les scripts de features existants calculent
#    pour ces lignes exactement les features d'entrée du match, sans modification.
# 3) écrit upcoming_games.json (matchs du jour, utilisé par fetch_odds.py)

MONEYPUCK_URL = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/skaters/{season}.zip"
NHL_API = "https://api-web.nhle.com/v1"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
ACTIVE_DAYS = 30  # un joueur du roster est "actif" s'il a joué dans les 30 derniers jours
                  # (assez large pour un retour de blessure de quelques semaines)

# mêmes colonnes que clean_reduce.py
COLS = [
    "playerId", "name", "gameId", "season", "playerTeam", "opposingTeam",
    "home_or_away", "gameDate", "position", "situation", "icetime", "shifts",
    "I_F_shotsOnGoal", "I_F_missedShots", "I_F_blockedShotAttempts",
    "I_F_shotAttempts", "I_F_goals", "I_F_xGoals", "I_F_rebounds",
    "I_F_highDangerShots", "I_F_mediumDangerShots", "I_F_lowDangerShots",
    "OnIce_F_shotAttempts", "OnIce_A_shotAttempts",
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nhl-shots-model"})
    # certifi: le Python de python.org sur macOS n'a pas de certificats SSL par défaut
    with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
        return r.read()


def season_for(d):
    # Les saisons sont nommées par leur année de début; la saison régulière commence en octobre
    return d.year if d.month >= 8 else d.year - 1


def nhl_games_on(d):
    sched = json.loads(get(f"{NHL_API}/schedule/{d.isoformat()}"))
    day = next((w for w in sched["gameWeek"] if w["date"] == d.isoformat()), None)
    # 2 = saison régulière, 3 = séries (MoneyPuck n'a que la saison régulière dans ses fichiers)
    return [g for g in (day["games"] if day else []) if g["gameType"] in (2, 3)]


def team_name(t):
    return f"{t['placeName']['default']} {t['commonName']['default']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat(), help="date des matchs à prédire (YYYY-MM-DD)")
    target = date.fromisoformat(parser.parse_args().date)
    season = season_for(target)

    # --- 1) MoneyPuck, saison en cours ---
    print(f"Téléchargement MoneyPuck saison {season}...")
    with zipfile.ZipFile(io.BytesIO(get(MONEYPUCK_URL.format(season=season)))) as z:
        name = z.namelist()[0]
        with z.open(name) as f, open(f"extracted/{season}.csv", "wb") as out:
            out.write(f.read())
    mp = pd.read_csv(f"extracted/{season}.csv", usecols=["playerId", "name", "gameId", "gameDate", "situation", "icetime"])
    played = mp[(mp["situation"] == "all") & (mp["icetime"] > 0)].copy()
    played["gameDate"] = pd.to_datetime(played["gameDate"].astype(str), format="%Y%m%d").dt.date
    last_mp = played["gameDate"].max() if len(played) else None
    print(f"  {played['gameId'].nunique()} matchs, dernier match: {last_mp}")

    # Matchs déjà terminés mais absents de MoneyPuck -> features un peu en retard
    if last_mp is not None:
        d, missing = last_mp + timedelta(days=1), 0
        while d < target:
            missing += len(nhl_games_on(d))
            d += timedelta(days=1)
        if missing:
            print(f"  ATTENTION: {missing} match(s) joué(s) entre {last_mp} et {target} pas encore dans MoneyPuck "
                  f"(mise à jour ~ midi le lendemain). Les features n'en tiennent pas compte.")

    # --- 2) Matchs du jour + joueurs probables ---
    games = [g for g in nhl_games_on(target) if g["id"] not in set(played["gameId"])]
    print(f"\n{len(games)} match(s) à prédire le {target}")

    recent = played[played["gameDate"] >= target - timedelta(days=ACTIVE_DAYS)]
    active_ids = set(recent["playerId"])

    rosters, rows, games_out = {}, [], []
    for g in games:
        home, away = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        games_out.append({
            "gameId": g["id"], "startTimeUTC": g["startTimeUTC"],
            "home": home, "away": away,
            "home_name": team_name(g["homeTeam"]), "away_name": team_name(g["awayTeam"]),
        })
        for team, opp, side in [(home, away, "HOME"), (away, home, "AWAY")]:
            if team not in rosters:
                rosters[team] = json.loads(get(f"{NHL_API}/roster/{team}/current"))
            skaters = rosters[team]["forwards"] + rosters[team]["defensemen"]
            # en tout début de saison personne n'a encore joué: on garde tout le roster
            if active_ids:
                skaters = [p for p in skaters if p["id"] in active_ids]
            for p in skaters:
                # noms du roster NHL: ceux de MoneyPuck perdent les accents (« Tim Sttzle »),
                # ce qui empêche de les associer aux noms des cotes
                pname = f"{p['firstName']['default']} {p['lastName']['default']}"
                base = {
                    "playerId": p["id"], "name": pname, "gameId": g["id"], "season": season,
                    "playerTeam": team, "opposingTeam": opp, "home_or_away": side,
                    "gameDate": int(target.strftime("%Y%m%d")), "position": p["positionCode"],
                    "icetime": 1.0,  # > 0 pour passer le filtre des scratchs; jamais utilisé (shift)
                }
                for situation in ["all", "5on4"]:
                    rows.append({**base, "situation": situation})
        print(f"  {away} @ {home}")

    upcoming = pd.DataFrame(rows, columns=COLS)
    upcoming.to_csv("upcoming_rows.csv", index=False)
    with open("upcoming_games.json", "w") as f:
        json.dump({"date": target.isoformat(), "moneypuck_last_game": str(last_mp), "games": games_out}, f, indent=1)
    print(f"\n{upcoming['playerId'].nunique()} joueurs dans {len(games)} matchs -> upcoming_rows.csv")


if __name__ == "__main__":
    main()
