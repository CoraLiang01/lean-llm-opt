#!/usr/bin/env bash
# Rerun only the two repaired arms; never rerun or modify tools-only.
set -euo pipefail
cd "/Users/yikeshuilingnencong/Desktop/lean-llm-opt-main"
PY="$PWD/.venv-ablation/bin/python"
selection="${1:-both}"
case "$selection" in
  ca) datasets=(Air-NRM-CA-LS) ;;
  np) datasets=(Air-NRM-NP-LS) ;;
  both) datasets=(Air-NRM-CA-LS Air-NRM-NP-LS) ;;
  *) echo "Usage: bash run_airnrm_file_ablations_terminal.sh [ca|np|both]"; exit 2 ;;
esac
read -r -s -p "OpenAI API key (hidden): " OPENAI_API_KEY </dev/tty
printf '\n'
export OPENAI_API_KEY
[[ -n "$OPENAI_API_KEY" ]] || { echo "API key is empty."; exit 1; }
trap 'unset OPENAI_API_KEY' EXIT
mkdir -p outputs
TASK_DIR=$(mktemp -d "$PWD/outputs/airnrm_api_file_ablations_$(date +%Y%m%d_%H%M%S)_XXXXXX")
export LEANOPT_OUT_DIR="$TASK_DIR/runs"
mkdir -p "$LEANOPT_OUT_DIR"
printf '%s\n' "$TASK_DIR" > outputs/latest_airnrm_api_file_ablations.txt
shasum -a 256 airnrm_api_files.py build_ls_ablations.py run_baseline.py \
  Test_Dataset/Air_NRM/large_scale/np_lean_workflow.py \
  Test_Dataset/Air_NRM/large_scale/workflow_only_np_example.txt \
  Test_Dataset/Air_NRM/large_scale/Ablation_Study_Air_NRM_Few-shot_Only_LargeScale.ipynb \
  > "$TASK_DIR/source_sha256.txt"
echo "Results: $TASK_DIR"
for dataset in "${datasets[@]}"; do
  query=Test_Dataset/Air_NRM/large_scale/query_CA.csv
  [[ "$dataset" != Air-NRM-NP-LS ]] || query=Test_Dataset/Air_NRM/large_scale/query_NP_Flow.csv
  "$PY" -u -c 'import dotenv,runpy,sys; dotenv.load_dotenv=lambda *a,**k:False; sys.argv=["run_notebook.py"]+sys.argv[1:]; runpy.run_path("run_notebook.py",run_name="__main__")' \
    Test_Dataset/Air_NRM/large_scale/Ablation_Study_Air_NRM_Few-shot_Only_LargeScale.ipynb \
    --profile gpt-4.1 --dataset "$dataset" --data "$query" --repeat 0 \
    --output-csv "$TASK_DIR/${dataset}_workflow_only.csv" \
    2>&1 | tee "$TASK_DIR/${dataset}_workflow_only.log"
  "$PY" -u run_baseline.py --dataset "$dataset" --profile gpt-4.1 \
    --method Abl-BaseModelOnly --api-files --repeat-index 0 \
    2>&1 | tee "$TASK_DIR/${dataset}_base_model_only.log"
done
unset OPENAI_API_KEY
"$PY" -u score_runs.py "$LEANOPT_OUT_DIR" --tol 1e-4 \
  -o "$TASK_DIR/gold_labels.csv" 2>&1 | tee "$TASK_DIR/scoring.log"
"$PY" -u aggregate_runs.py "$LEANOPT_OUT_DIR" --gold "$TASK_DIR/gold_labels.csv" \
  -o "$TASK_DIR/tables" 2>&1 | tee "$TASK_DIR/aggregation.log"
echo "Completed. Results: $TASK_DIR/tables/cost_table_per_run.csv"
