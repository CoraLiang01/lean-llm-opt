#!/usr/bin/env bash
#
# run_full.sh -- the full 101-instance Large-Scale-OR run, once.
#
#   ./verify/run_full.sh              # repeat_index 0, fresh run
#   ./verify/run_full.sh 1            # repeat_index 1, a second measurement
#   ./verify/run_full.sh 0 resume     # continue an interrupted run -- see below
#
# Reports the headline accuracy, the per-type table against the March
# baseline, and the 8 regression instances separately.
#
# One caveat worth keeping in mind while reading the result: across the seven
# post-packaging runs, 40 of these 101 instances flipped between correct and
# incorrect despite temperature=0 and a fixed seed, and the run totals spread
# over 73.3%-78.2%. A single run therefore carries roughly +/-2-3 points of
# noise on its own. Treat one number above 80% as encouraging, not as proof --
# the per-instance regression list at the bottom of the report is the part
# that actually distinguishes a fix from a lucky draw. For a number to quote,
# run this at several repeat indices:
#
#   for r in 0 1 2 3 4; do ./verify/run_full.sh $r; done
#
# Costs a full 101-instance pass of the pipeline. Not a smoke test.

set -euo pipefail
cd "$(dirname "$0")/.."

REPEAT="${1:-0}"
RESUME="${2:-}"
NOTEBOOK="LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
DATA="Test_Dataset/Large-scale-or/Large-scale-or-101.csv"

python3 verify/_preflight.py

echo "running all 101 instances, repeat_index=${REPEAT}"
echo

# Resume is opt-in, and deliberately so. run_notebook.py picks the directory to
# continue by matching method/profile/dataset/repeat off the run_id -- not the
# config hash, which is the last field of that name. So if the config changed
# between the interrupted run and the restart, --resume appends the new
# pipeline's instances into the old pipeline's directory, under the old
# config.json. That happened on 2026-08-22: 34 instances of one pipeline landed
# on top of 8 of another, and the directory is not a usable measurement of
# either. Only pass `resume` when the config has not moved since.
# Two spelled-out calls rather than an args array: macOS ships bash 3.2, where
# expanding an empty array as "${arr[@]}" counts as an unbound variable under
# `set -u` and aborts the script before it runs anything.
if [ "$RESUME" = "resume" ]; then
    echo "!! --resume: only valid if exp_config.yaml and the notebook are"
    echo "!! unchanged since the interrupted run. The config hash is the last"
    echo "!! field of the run_id; if it differs, start a fresh run instead."
    echo
    python3 run_notebook.py "$NOTEBOOK" \
        --data "$DATA" \
        --repeat "$REPEAT" \
        --resume
else
    python3 run_notebook.py "$NOTEBOOK" \
        --data "$DATA" \
        --repeat "$REPEAT"
fi

RUN_ID="$(python3 verify/_latest_run.py)"
echo
echo "scoring runs/${RUN_ID}"
echo

mkdir -p verify/out
python3 score_runs.py runs/ --run "$RUN_ID" -o verify/out/gold_labels.csv

echo
python3 verify/verify_report.py "runs/${RUN_ID}"
