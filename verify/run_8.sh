#!/usr/bin/env bash
#
# run_8.sh -- the cheap check.
#
# Runs only the 8 Large-Scale-OR instances that were correct in the March run
# and wrong in all seven runs made after the packaging change:
#
#     41, 50, 52  (RA)   58, 61, 78  (UFLP)   70 (Mixture)   99 (Others)
#
# They are the deterministic part of the 85.1% -> 75.2% drop; the rest of the
# benchmark flips between runs and cannot settle the question. If the three
# reverted settings were the cause, these come back.
#
#   ./verify/run_8.sh
#
# ~8 instances of generation + solve. Minutes, not hours.

set -euo pipefail
cd "$(dirname "$0")/.."

ROWS="41,50,52,58,61,70,78,99"
NOTEBOOK="LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
DATA="Test_Dataset/Large-scale-or/Large-scale-or-101.csv"

python3 verify/_preflight.py

echo "running instances ${ROWS}"
echo

python3 run_notebook.py "$NOTEBOOK" \
    --data "$DATA" \
    --rows "$ROWS"

# The run directory run_notebook.py just wrote into.
RUN_ID="$(python3 verify/_latest_run.py)"
echo
echo "scoring runs/${RUN_ID}"
echo

# -o points into verify/ on purpose: score_runs.py also writes scored_all.csv
# next to --out, and that file at the repo root is the merged history of the
# real runs. A verification pass must not land in it.
mkdir -p verify/out
python3 score_runs.py runs/ --run "$RUN_ID" -o verify/out/gold_labels.csv

echo
python3 verify/verify_report.py "runs/${RUN_ID}"
