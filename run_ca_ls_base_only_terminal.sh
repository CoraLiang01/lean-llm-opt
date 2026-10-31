#!/usr/bin/env bash
set -euo pipefail
cd "/Users/yikeshuilingnencong/Desktop/lean-llm-opt-main"
PY="$PWD/.venv-ablation/bin/python"
read -r -s -p "OpenAI API key (hidden): " OPENAI_API_KEY </dev/tty
printf '\n'
export OPENAI_API_KEY
if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "API key is empty."
  exit 1
fi
trap 'unset OPENAI_API_KEY' EXIT
mkdir -p outputs
TASK_DIR=$(mktemp -d "$PWD/outputs/ca_ls_base_only_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export LEANOPT_OUT_DIR="$TASK_DIR/runs"
mkdir -p "$LEANOPT_OUT_DIR"
printf '%s\n' "$TASK_DIR" > outputs/latest_ca_ls_base_only.txt
echo "Results: $TASK_DIR"
"$PY" -u run_baseline.py --dataset Air-NRM-CA-LS --profile gpt-4.1 \
  --method Abl-BaseModelOnly --api-files --repeat-index 0 \
  2>&1 | tee "$TASK_DIR/base_model_only.log"
# The scoring subprocess does not need an API key.
unset OPENAI_API_KEY
"$PY" -u score_runs.py "$LEANOPT_OUT_DIR" --dataset Air-NRM-CA-LS \
  --tol 1e-4 -o "$TASK_DIR/gold_labels.csv" \
  2>&1 | tee "$TASK_DIR/scoring.log"
"$PY" -u aggregate_runs.py "$LEANOPT_OUT_DIR" \
  --gold "$TASK_DIR/gold_labels.csv" -o "$TASK_DIR/tables" \
  2>&1 | tee "$TASK_DIR/aggregation.log"
echo "Completed. Results: $TASK_DIR/tables/cost_table_per_run.csv"
