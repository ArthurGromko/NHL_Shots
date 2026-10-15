import argparse, json, os, re, ssl, unicodedata, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone
import certifi
import pandas as pd
from update_live_data import nhl_games_on, team_name

# Cotes live du marché player_shots_on_goal (The Odds API, tier gratuit = 500 crédits/mois).
# - /events est gratuit; /events/{id}/odds coûte 1 crédit par match (1 marché x 1 région).
# - Une saison ~1 312 matchs => ~220 crédits/mois pour une capture par match.
# Deux modes:
# - défaut (run_daily.sh, 13 h): tous les matchs du jour pas encore commencés, lus dans
#   upcoming_games.json -> odds/odds_<date>.csv. Ce sont les cotes utilisées pour les picks.
# - --closing (agent launchd toutes les 30 min): les matchs qui commencent dans les 30
#   prochaines minutes, une seule fois par match -> odds/closing_<date>.csv. Ces cotes de
#   fermeture servent à mesurer la CLV des picks (grade_results.py). Le calendrier vient
#   directement de l'API NHL, pour ne pas dépendre de l'exécution de 13 h.
# Cotes brutes -> odds/raw/<date>/

API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
MARKET = "player_shots_on_goal"
# Bookmakers nommés plutôt qu'une région: jusqu'à 10 coûtent comme 1 région (1 crédit/match).
# Pinnacle (région eu) sert de référence "sharp" pour noter le modèle. bet365 n'est pas couvert
# par The Odds API (seulement bet365_au, payant, AFL/NRL).
BOOKMAKERS = ["pinnacle", "fanduel", "draftkings", "betmgm", "espnbet",
              "betrivers", "bovada", "betonlineag", "lowvig", "hardrockbet"]
assert len(BOOKMAKERS) <= 10, "au-delà de 10 bookmakers, chaque match coûte 2 crédits"
MIN_CREDITS_LEFT = 20  # marge de sécurité
# Fenêtre de fermeture = intervalle entre deux exécutions: chaque heure de début tombe dans
# exactement une fenêtre (now, now + 30 min]
CLOSING_WINDOW = timedelta(minutes=30)
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def api_key():
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    with open(".env") as f:
        for line in f:
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("ODDS_API_KEY introuvable (.env ou variable d'environnement)")


def get(path, **params):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as r:
        return json.loads(r.read()), r.headers


def todays_games_from_schedule(d):
    return [{"gameId": g["id"], "startTimeUTC": g["startTimeUTC"],
             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
             "home_name": team_name(g["homeTeam"]), "away_name": team_name(g["awayTeam"])}
            for g in nhl_games_on(d)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="liste les matchs sans dépenser de crédits")
    parser.add_argument("--closing", action="store_true", help="cotes de fermeture (matchs qui commencent dans 30 min)")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)

    if args.closing:
        day = date.today().isoformat()
        games = todays_games_from_schedule(date.today())
        done_path = f"odds/closing_done_{day}.json"
        done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
        start_of = lambda g: datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        games = [g for g in games if g["gameId"] not in done and now < start_of(g) <= now + CLOSING_WINDOW]
        if not games:
            return  # rien à capturer: pas d'appel à l'API
        out_path = f"odds/closing_{day}.csv"
    else:
        with open("upcoming_games.json") as f:
            upcoming = json.load(f)
        day, games = upcoming["date"], upcoming["games"]
        out_path = f"odds/odds_{day}.csv"

    key = api_key()
    by_teams = {(norm(g["home_name"]), norm(g["away_name"])): g for g in games}
    events, headers = get("/events", apiKey=key)
    credits = int(headers.get("x-requests-remaining", 0))

    todo, seen = [], set()
    for e in events:
        g = by_teams.get((norm(e["home_team"]), norm(e["away_team"])))
        if g is None:
            continue
        seen.add(g["gameId"])
        start = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if start <= now:
            print(f"  déjà commencé, ignoré: {g['away']} @ {g['home']}")
            continue
        todo.append((e, g))
    for g in games:
        if g["gameId"] not in seen:
            print(f"  pas d'événement Odds API pour {g['away']} @ {g['home']} (nom d'équipe différent ?)")

    print(f"{now:%H:%M}Z {'fermeture' if args.closing else 'capture'}: {len(todo)} match(s), crédits restants: {credits}")
    if args.dry_run:
        return
    if credits - len(todo) < MIN_CREDITS_LEFT:
        raise SystemExit(f"Crédits insuffisants ({credits} restants, {len(todo)} nécessaires + marge {MIN_CREDITS_LEFT})")

    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_dir = f"odds/raw/{day}"
    os.makedirs(raw_dir, exist_ok=True)
    prefix = "closing_" if args.closing else ""
    rows = []
    for e, g in todo:
        data, headers = get(f"/events/{e['id']}/odds", apiKey=key, bookmakers=",".join(BOOKMAKERS), markets=MARKET, oddsFormat="decimal")
        with open(f"{raw_dir}/{prefix}{g['gameId']}_{fetched_at.replace(':', '')}.json", "w") as f:
            json.dump(data, f)
        n = 0
        for book in data.get("bookmakers", []):
            for market in book["markets"]:
                if market["key"] != MARKET:
                    continue
                for o in market["outcomes"]:
                    rows.append({
                        "fetched_at": fetched_at, "gameId": g["gameId"], "event_id": e["id"],
                        "commence_time": e["commence_time"], "bookmaker": book["key"],
                        "last_update": market.get("last_update"), "player": o.get("description"),
                        "side": o["name"], "line": o.get("point"), "price": o["price"],
                    })
                    n += 1
        print(f"  {g['away']} @ {g['home']}: {n} cotes")
    credits = int(headers.get("x-requests-remaining", credits))

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
    print(f"{len(rows)} cotes -> {out_path} ; crédits restants: {credits}")
    if args.closing:
        # une seule tentative par match, même sans cotes, pour ne pas payer deux fois
        with open(done_path, "w") as f:
            json.dump(sorted(done | {g["gameId"] for _, g in todo}), f)


if __name__ == "__main__":
    main()
