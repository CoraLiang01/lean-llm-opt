#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LEANOPT_PYTHON:-/opt/anaconda3/envs/leanopt/bin/python}"
ONE_SHOT_RUNNER="${SCRIPT_DIR}/run_airnrm_fewshot_benchmark.py"
AGENTIC_RUNNER="${SCRIPT_DIR}/Test_Dataset/Air_NRM/large_scale/run_agentic_benchmark.py"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 {gpt-5.4|gemini-3.1-pro} [common benchmark options]" >&2
    exit 2
fi

MODEL="$1"
shift

case "${MODEL}" in
    gpt-5.4)
        PROVIDER_NAME="OpenAI"
        KEY_VARIABLE="OPENAI_API_KEY"
        ;;
    gemini-3.1-pro)
        PROVIDER_NAME="Gemini"
        KEY_VARIABLE="GOOGLE_API_KEY"
        ;;
    *)
        echo "Error: model must be gpt-5.4 or gemini-3.1-pro." >&2
        exit 2
        ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: leanopt Python not found at ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${ONE_SHOT_RUNNER}" || ! -f "${AGENTIC_RUNNER}" ]]; then
    echo "Error: an Air-NRM benchmark runner is missing." >&2
    exit 1
fi

DRY_RUN=0
HAS_COST_LIMIT=0
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
    [[ "${arg}" == "--max-run-cost-usd" || "${arg}" == --max-run-cost-usd=* ]] && HAS_COST_LIMIT=1
    if [[ "${arg}" == "--prompt-key" ]]; then
        echo "Error: this suite already prompts once; do not pass --prompt-key." >&2
        exit 2
    fi
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
    SECRET=""
    printf "Enter %s API key (input hidden): " "${PROVIDER_NAME}" >&2
    IFS= read -r -s SECRET
    printf "\n" >&2
    if [[ ${#SECRET} -lt 20 ]]; then
        echo "Error: ${PROVIDER_NAME} API key looks empty or incomplete." >&2
        exit 1
    fi
    printf -v "${KEY_VARIABLE}" '%s' "${SECRET}"
    export "${KEY_VARIABLE}"
    echo "${PROVIDER_NAME} API key accepted: ****${SECRET: -4}"
    SECRET=""
fi

trap 'unset OPENAI_API_KEY GOOGLE_API_KEY LEANOPT_RUN_SUFFIX' EXIT

COMMON_ARGS=("$@")
if [[ "${HAS_COST_LIMIT}" -eq 0 ]]; then
    COMMON_ARGS+=(--max-run-cost-usd 45)
fi

LOG_DIR="${SCRIPT_DIR}/outputs/airnrm_provider_suite"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SUITE_UID="$("${PYTHON_BIN}" -c 'import uuid; print(uuid.uuid4().hex[:12])')"
export LEANOPT_RUN_SUFFIX="${SUITE_UID}"
LOG_FILE="${LOG_DIR}/${MODEL}__${STAMP}__${SUITE_UID}.log"
RUN_LIST_FILE="${LOG_DIR}/${MODEL}__${STAMP}__${SUITE_UID}.runs.txt"
FAILURES=0

echo "Suite ID: ${SUITE_UID}" | tee -a "${LOG_FILE}"
echo "Cost policy: --max-run-cost-usd is an independent limit for EACH run." | tee -a "${LOG_FILE}"
echo "There is no combined cost limit across the 6-run suite." | tee -a "${LOG_FILE}"
echo "Output policy: every run_id includes this Suite ID; existing runs are never reused." | tee -a "${LOG_FILE}"

run_case() {
    local label="$1"
    shift
    echo | tee -a "${LOG_FILE}"
    echo "===== ${label} =====" | tee -a "${LOG_FILE}"
    echo "Starting a new process with a fresh per-run cost counter." | tee -a "${LOG_FILE}"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
    local status=${PIPESTATUS[0]}
    if [[ ${status} -ne 0 ]]; then
        echo "FAILED (${status}): ${label}" | tee -a "${LOG_FILE}"
        FAILURES=$((FAILURES + 1))
    else
        echo "COMPLETED: ${label}" | tee -a "${LOG_FILE}"
    fi
}

cd "${SCRIPT_DIR}"

# Strict one-example single-call arm: one matching LEANOPT exemplar, raw CSVs,
# and the current query; empty system prompt and no auxiliary task hints.
for scale in small large; do
    for family in ca np; do
        run_case \
            "SingleCall-OneExample ${scale}/${family} ${MODEL}" \
            "${PYTHON_BIN}" "${ONE_SHOT_RUNNER}" \
            "${scale}" "${family}" "${MODEL}" "${COMMON_ARGS[@]}"
    done
done

# Existing Air-NRM large-scale agentic arm: formulation -> code -> at most one
# repair.  It is logged separately from the strict single-call arm.
for family in ca np; do
    run_case \
        "Agentic-2Stage-1Repair large/${family} ${MODEL}" \
        "${PYTHON_BIN}" "${AGENTIC_RUNNER}" "${MODEL}" \
        --dataset "${family}" --workflow two-stage-repair "${COMMON_ARGS[@]}"
done

echo | tee -a "${LOG_FILE}"
echo "Suite finished: 6 runs, ${FAILURES} failed." | tee -a "${LOG_FILE}"
echo "Suite log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
find "${SCRIPT_DIR}/runs" -maxdepth 1 -type d \
    -name "*__u${SUITE_UID}__*" -print | sort > "${RUN_LIST_FILE}"
echo "Saved run directories: ${RUN_LIST_FILE}" | tee -a "${LOG_FILE}"

if [[ ${FAILURES} -ne 0 ]]; then
    exit 1
fi
