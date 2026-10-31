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
TASK_DIR=$(mktemp -d "$PWD/outputs/ca_ls_ablations_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export LEANOPT_OUT_DIR="$TASK_DIR/runs"
mkdir -p "$LEANOPT_OUT_DIR"
printf '%s\n' "$TASK_DIR" > outputs/latest_ca_ls_ablations.txt
echo "Results: $TASK_DIR"

run_arm() {
  local arm="$1"
  local notebook="$2"
  # Prevent notebook load_dotenv(override=True) from replacing the entered key.
  "$PY" -u -c 'import dotenv, runpy, sys; dotenv.load_dotenv = lambda *a, **k: False; sys.argv = ["run_notebook.py"] + sys.argv[1:]; runpy.run_path("run_notebook.py", run_name="__main__")' \
    "$notebook" --profile gpt-4.1 --dataset Air-NRM-CA-LS \
    --data Test_Dataset/Air_NRM/large_scale/query_CA.csv --repeat 0 \
    --output-csv "$TASK_DIR/${arm}_outputs.csv" \
    2>&1 | tee "$TASK_DIR/${arm}.log"
  "$PY" -u score_runs.py "$LEANOPT_OUT_DIR" \
    --dataset Air-NRM-CA-LS --tol 1e-4 \
    -o "$TASK_DIR/gold_labels.csv" 2>&1 | tee "$TASK_DIR/scoring.log"
  "$PY" -u aggregate_runs.py "$LEANOPT_OUT_DIR" \
    --gold "$TASK_DIR/gold_labels.csv" -o "$TASK_DIR/tables" \
    2>&1 | tee "$TASK_DIR/aggregation.log"
}

run_arm rag_only Test_Dataset/Air_NRM/large_scale/Ablation_Study_Air_NRM_RAG_Only_LargeScale.ipynb
run_arm fewshot_only Test_Dataset/Air_NRM/large_scale/Ablation_Study_Air_NRM_Few-shot_Only_LargeScale.ipynb
echo "Completed. Results: $TASK_DIR/tables/cost_table_per_run.csv"
