#!/bin/bash
# Pipeline quotidien: données du jour -> features -> prédictions -> cotes -> edges.
# À lancer depuis le dossier de travail (pipeline_run/), ex:
#   ../handoff/scripts/run_daily.sh            (matchs d'aujourd'hui)
#   ../handoff/scripts/run_daily.sh 2026-10-16 (autre date)
#   NO_ODDS=1 ../handoff/scripts/run_daily.sh  (sans dépenser de crédits Odds API)
# Le modèle de production (prod_*.pkl) doit exister: prepare_model_data.py puis fit_production.py.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-../.venv/bin/python}"
DATE="${1:-$(date +%F)}"

$PY "$S/update_live_data.py" --date "$DATE"
$PY "$S/clean_reduce.py" --upcoming > /dev/null
for s in build_features build_opponent_context merge_all_features build_pp_features; do
    $PY "$S/$s.py" > /dev/null
done
$PY "$S/predict_today.py"
if [ -z "${NO_ODDS:-}" ]; then
    $PY "$S/fetch_odds.py"
    $PY "$S/compare_odds.py" --date "$DATE"
fi
