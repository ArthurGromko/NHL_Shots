import os, subprocess, sys
from datetime import date, datetime

# Pipeline quotidien: notation des jours précédents -> données du jour -> features ->
# prédictions -> cotes -> edges -> picks figés. Lancé par launchd à 13 h, ou à la main:
#   ../.venv/bin/python ../handoff/scripts/run_daily.py [YYYY-MM-DD]
#   NO_ODDS=1 ...  (sans dépenser de crédits Odds API, donc sans edges ni picks)
# En Python plutôt qu'en bash: sur macOS, un job launchd lancé via /bin/bash n'a pas accès
# au dossier Documents, alors que le Python du projet l'a (accès accordé par programme).
# Le modèle de production (prod_*.pkl) doit exister: prepare_model_data.py puis fit_production.py.

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.join(SCRIPTS, "..", "..", "pipeline_run")


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    os.chdir(WORKDIR)
    os.makedirs("logs", exist_ok=True)
    log = open(f"logs/daily_{day}.log", "a")

    def run(script, *args, quiet=False, required=True):
        log.write(f"\n===== {datetime.now():%H:%M:%S} {script} {' '.join(args)}\n")
        log.flush()
        out = subprocess.DEVNULL if quiet else log
        subprocess.run([sys.executable, os.path.join(SCRIPTS, script), *args],
                       stdout=out, stderr=log, check=required)

    run("update_live_data.py", "--date", day)
    # les matchs de la veille sont maintenant dans MoneyPuck; un échec de notation
    # ne doit pas empêcher les prédictions et la capture du jour
    run("grade_results.py", required=False)
    run("clean_reduce.py", "--upcoming", quiet=True)
    for script in ["build_features.py", "build_opponent_context.py", "merge_all_features.py", "build_pp_features.py"]:
        run(script, quiet=True)
    run("predict_today.py")
    if not os.environ.get("NO_ODDS"):
        run("fetch_odds.py")
        run("compare_odds.py", "--date", day)
        run("make_picks.py", "--date", day)
    log.write(f"\n===== {datetime.now():%H:%M:%S} terminé\n")


if __name__ == "__main__":
    main()
