#!/usr/bin/env bash
set -euo pipefail

# Terminal launcher for the Large-Scale-OR 101 single-call benchmark.
# The Python runner asks for the API key with hidden input and prints only
# the final four characters as confirmation. The full key is never placed in
# the shell command or command history.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_singlecall_cost_benchmark.py"
DEFAULT_PYTHON="/opt/anaconda3/envs/leanopt/bin/python"
PYTHON_BIN="${LEANOPT_PYTHON:-${DEFAULT_PYTHON}}"

usage() {
    echo "Usage:"
    echo "  bash $0 gpt-5 [options]"
    echo "  bash $0 gpt-4.1 [options]"
    echo "  bash $0 gpt-5.2 [options]"
    echo "  bash $0 gemini-3.5-flash [options]"
    echo "  bash $0 gemini-3.1-pro [options]"
    echo
    echo "Examples:"
    echo "  bash $0 gpt-5.2"
    echo "  bash $0 gemini-3.5-flash"
    echo "  bash $0 gemini-3.1-pro"
    echo "  bash $0 gpt-5.2 --rows 0-2 --max-run-cost-usd 5"
    echo "  bash $0 gemini-3.5-flash --dry-run"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

MODEL="$1"
shift

case "${MODEL}" in
    gpt-5|gpt-4.1|gpt-5.2|gemini-3.5-flash|gemini-3.1-pro|gemini-3-pro)
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unsupported model '${MODEL}'; see usage below." >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python environment not found at ${PYTHON_BIN}" >&2
    echo "Set LEANOPT_PYTHON to the correct Python executable and retry." >&2
    exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
    echo "Error: benchmark runner not found at ${RUNNER}" >&2
    exit 1
fi

echo "Large-Scale-OR 101 single-call benchmark"
echo "Model: ${MODEL}"
echo "API key input will be hidden; only its final four characters will be shown."

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${RUNNER}" "${MODEL}" --prompt-key "$@"
