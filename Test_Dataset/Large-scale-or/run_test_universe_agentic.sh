#!/bin/bash
# Air-NRM large-scale one-request inline-data benchmark launcher (macOS Bash).
#
# Examples:
#   ./run_test_universe_agentic.sh gemini
#   ./run_test_universe_agentic.sh openai --rows 0-2
#   ./run_test_universe_agentic.sh all --limit 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_SCRIPT="${REPO_ROOT}/Test_Dataset/Air_NRM/large_scale/run_agentic_benchmark.py"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/leanopt/bin/python}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: benchmark Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

usage() {
    echo "Usage: $0 <gemini|openai|all> [runner options]"
    echo ""
    echo "Runner options include:"
    echo "  --rows 0,2-4       Run selected zero-based query rows"
    echo "  --limit N          Run only the first N selected rows"
    echo "  --repeat-index N   Repeat identifier stored in the run log"
    echo "  --timeout N        Solver/report timeout per instance"
    echo "  --if-proxy         Explicitly enable the configured proxy"
    echo "  --no-report        Skip automatic scoring/report generation"
    echo "  --dry-run          Validate configuration without API calls"
    exit 1
}

run_model() {
    local profile="$1"
    shift
    local log_file="${LOG_DIR}/airnrm_agentic_${profile}_${TIMESTAMP}.log"

    echo "Starting ${profile}"
    echo "  Python: $PYTHON_BIN"
    echo "  Log:    $log_file"
    nohup "$PYTHON_BIN" "$PYTHON_SCRIPT" "$profile" "$@" \
        > "$log_file" 2>&1 &
    echo "  PID:    $!"
}

if [ "$#" -lt 1 ]; then
    usage
fi

model_type="$1"
shift
case "$model_type" in
    gemini)
        run_model gemini "$@"
        ;;
    openai)
        run_model openai "$@"
        ;;
    all)
        run_model gemini "$@"
        run_model openai "$@"
        echo "Both isolated runs were launched."
        ;;
    *)
        echo "Error: unknown model type '$model_type'" >&2
        usage
        ;;
esac

echo "Each model writes to its own runs/<run_id>/ directory and log file."
