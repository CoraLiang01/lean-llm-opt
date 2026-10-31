#!/bin/bash
# Run the supplied V2 notebook on the automatic 101-case benchmark, then
# score the generated Gurobi programs and build accuracy/cost tables.
#
# Full run:
#   ./run_v2_101.sh
# Resume the newest interrupted V2 run:
#   ./run_v2_101.sh --resume
# Free plan check (no API calls):
#   ./run_v2_101.sh --list
# Three-case smoke run:
#   ./run_v2_101.sh --rows 15,45,81

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${LEANOPT_PYTHON:-/opt/anaconda3/envs/leanopt/bin/python}"
MARKER="${SCRIPT_DIR}/outputs/large_scale_or_test_final/latest_run_dir.txt"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    echo "Set LEANOPT_PYTHON to the Python executable for your leanopt environment." >&2
    exit 1
fi

cd "$SCRIPT_DIR"

for argument in "$@"; do
    if [ "$argument" = "--list" ]; then
        exec "$PYTHON_BIN" run_v2_101.py "$@"
    fi
done

if [ "${LEANOPT_SKIP_PREFLIGHT:-0}" != "1" ]; then
    # The repository-wide preflight also checks optional label-building and
    # unrelated notebooks. This run needs only the V2 automatic 101 pipeline.
    "$PYTHON_BIN" run_v2_101.py --preflight
fi

"$PYTHON_BIN" run_v2_101.py --run-dir-marker "$MARKER" "$@"

if [ ! -s "$MARKER" ]; then
    echo "Run completed but no run-directory marker was written: $MARKER" >&2
    exit 1
fi

RUN_DIR="$(tail -n 1 "$MARKER")"
RUN_ID="$(basename "$RUN_DIR")"

echo
echo "Scoring generated Gurobi programs for $RUN_ID"
"$PYTHON_BIN" score_runs.py runs/ \
    --run "$RUN_ID" \
    --time-limit 120 \
    -o "$RUN_DIR/gold_labels.csv"

echo
echo "Building the accuracy, token, cost, tool-call, and runtime tables"
"$PYTHON_BIN" aggregate_runs.py runs/ \
    --run "$RUN_ID" \
    --gold "$RUN_DIR/gold_labels.csv" \
    -o "$RUN_DIR/tables"

echo
echo "Complete: $RUN_DIR"
echo "Headline table: $RUN_DIR/tables/cost_table.csv"
