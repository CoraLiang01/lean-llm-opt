#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_airnrm_fewshot_benchmark.py"
DEFAULT_PYTHON="/opt/anaconda3/envs/leanopt/bin/python"
PYTHON_BIN="${LEANOPT_PYTHON:-${DEFAULT_PYTHON}}"

usage() {
    echo "Usage:"
    echo "  bash $0 {small|large|all} {ca|np|all} {gpt-5.4|gemini-3.1-pro|all} [options]"
    echo
    echo "Examples:"
    echo "  bash $0 small ca gpt-5.4 --rows 0-2"
    echo "  bash $0 large np gemini-3.1-pro --limit 3"
    echo "  bash $0 all all all --dry-run"
    echo "  bash $0 all all all --max-run-cost-usd 45"
}

if [[ $# -lt 3 ]]; then
    usage
    exit 2
fi

SCALE="$1"
FAMILY="$2"
MODEL="$3"
shift 3

case "${SCALE}" in small|large|all) ;; *) usage >&2; exit 2 ;; esac
case "${FAMILY}" in ca|np|all) ;; *) usage >&2; exit 2 ;; esac
case "${MODEL}" in gpt-5.4|gemini-3.1-pro|all) ;; *) usage >&2; exit 2 ;; esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python environment not found at ${PYTHON_BIN}" >&2
    echo "Set LEANOPT_PYTHON to the leanopt Python executable." >&2
    exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
    echo "Error: runner not found at ${RUNNER}" >&2
    exit 1
fi

DRY_RUN=0
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        DRY_RUN=1
    fi
done

read_key() {
    local provider="$1"
    local variable_name="$2"
    local secret=""
    printf "Enter %s API key (input hidden): " "${provider}" >&2
    IFS= read -r -s secret
    printf "\n" >&2
    if [[ ${#secret} -lt 20 ]]; then
        echo "Error: ${provider} API key looks empty or incomplete." >&2
        exit 1
    fi
    printf -v "${variable_name}" '%s' "${secret}"
    export "${variable_name}"
    echo "${provider} API key accepted: ****${secret: -4}"
    secret=""
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ "${MODEL}" == "gpt-5.4" || "${MODEL}" == "all" ]]; then
        read_key "OpenAI" "OPENAI_API_KEY"
    fi
    if [[ "${MODEL}" == "gemini-3.1-pro" || "${MODEL}" == "all" ]]; then
        read_key "Gemini" "GOOGLE_API_KEY"
    fi
fi

if [[ "${SCALE}" == "all" ]]; then
    SCALES=(small large)
else
    SCALES=("${SCALE}")
fi
if [[ "${FAMILY}" == "all" ]]; then
    FAMILIES=(ca np)
else
    FAMILIES=("${FAMILY}")
fi
if [[ "${MODEL}" == "all" ]]; then
    MODELS=(gpt-5.4 gemini-3.1-pro)
else
    MODELS=("${MODEL}")
fi

cd "${SCRIPT_DIR}"
for selected_scale in "${SCALES[@]}"; do
    for selected_family in "${FAMILIES[@]}"; do
        for selected_model in "${MODELS[@]}"; do
            echo
            echo "=== Air-NRM ${selected_scale}/${selected_family} | ${selected_model} ==="
            "${PYTHON_BIN}" "${RUNNER}" \
                "${selected_scale}" "${selected_family}" "${selected_model}" "$@"
        done
    done
done

unset OPENAI_API_KEY GOOGLE_API_KEY
