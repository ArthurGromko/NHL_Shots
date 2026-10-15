import argparse, json, os, re, ssl, unicodedata, urllib.parse, urllib.request
from datetime import datetime, timezone
import certifi
import pandas as pd

# Cotes live du marché player_shots_on_goal (The Odds API, tier gratuit = 500 crédits/mois).
# - /events est gratuit; /events/{id}/odds coûte 1 crédit par match (1 marché x 1 région).
# - Une saison ~1 312 matchs => ~220 crédits/mois pour une capture par match.
# Matchs du jour lus dans upcoming_games.json (update_live_data.py). Seuls les matchs
# pas encore commencés sont capturés. Cotes brutes -> odds/raw/<date>/, à plat -> odds/odds_<date>.csv

API = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
MARKET = "player_shots_on_goal"
REGION = "us"
MIN_CREDITS_LEFT = 20  # marge de sécurité
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="liste les matchs sans dépenser de crédits")
    dry_run = parser.parse_args().dry_run
    key = api_key()

    with open("upcoming_games.json") as f:
        upcoming = json.load(f)
    by_teams = {(norm(g["home_name"]), norm(g["away_name"])): g for g in upcoming["games"]}

    events, headers = get("/events", apiKey=key)
    credits = int(headers.get("x-requests-remaining", 0))
    now = datetime.now(timezone.utc)

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
    for g in upcoming["games"]:
        if g["gameId"] not in seen:
            print(f"  pas d'événement Odds API pour {g['away']} @ {g['home']} (nom d'équipe différent ?)")

    print(f"{len(todo)} match(s) à capturer, crédits restants: {credits}")
    if dry_run:
        return
    if credits - len(todo) < MIN_CREDITS_LEFT:
        raise SystemExit(f"Crédits insuffisants ({credits} restants, {len(todo)} nécessaires + marge {MIN_CREDITS_LEFT})")

    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_dir = f"odds/raw/{upcoming['date']}"
    os.makedirs(raw_dir, exist_ok=True)
    rows = []
    for e, g in todo:
        data, headers = get(f"/events/{e['id']}/odds", apiKey=key, regions=REGION, markets=MARKET, oddsFormat="decimal")
        with open(f"{raw_dir}/{g['gameId']}_{fetched_at.replace(':', '')}.json", "w") as f:
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

    path = f"odds/odds_{upcoming['date']}.csv"
    out = pd.DataFrame(rows)
    out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"{len(out)} cotes -> {path} ; crédits restants: {credits}")


if __name__ == "__main__":
    main()
