#!/usr/bin/env bash
# First-time setup on this machine, then the smoke test for one task.
#
#     bash setup_and_smoke.sh large_scale_or
#     bash setup_and_smoke.sh loto
#     bash setup_and_smoke.sh air_nrm_ls
#
# Safe to re-run: every step checks before it acts, so later invocations skip
# the pip install and the model pulls.
#
# This never starts a full benchmark. It stops after the smoke test and prints
# a verdict; the real runs are a separate, deliberate command (see README).
set -uo pipefail

TASK="${1:-}"
case "$TASK" in
    large_scale_or|loto|air_nrm_ls) ;;
    *) echo "usage: bash setup_and_smoke.sh {large_scale_or|loto|air_nrm_ls}" >&2
       exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die()  { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 1; }

step "0. environment"
"$PY" -c 'import sys; print("python", sys.version.split()[0], "at", sys.executable); \
sys.exit(0 if sys.version_info >= (3,10) else 1)' \
    || die "need Python 3.10+ (conda create -n leanopt python=3.11 && conda activate leanopt)"

step "1. python packages"
if "$PY" -c 'import langchain_classic, langchain_ollama, langchain_openai, faiss, pandas, yaml, dotenv, tiktoken, gurobipy' 2>/dev/null; then
    echo "all packages already present"
else
    echo "installing from requirements-server.txt ..."
    "$PY" -m pip install -q -r "$ROOT/requirements-server.txt" || die "pip install failed"
fi

step "2. ollama"
command -v ollama >/dev/null 2>&1 || die "ollama not found -- https://ollama.com/download"
curl -sf "$OLLAMA_URL/api/tags" >/dev/null || die "no Ollama server at $OLLAMA_URL
     start it with:  ollama serve
     or point elsewhere:  export OLLAMA_BASE_URL=http://host:port"
echo "server reachable at $OLLAMA_URL"
for model in gpt-oss:20b nomic-embed-text; do
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$model"; then
        echo "  $model already pulled"
    else
        echo "  pulling $model ..."
        ollama pull "$model" || die "could not pull $model"
    fi
done

step "3. package self-check (no model calls)"
cd "$ROOT/$TASK" || die "missing directory: $TASK"
"$PY" "$ROOT/verify_package.py" || die "self-check failed -- fix the FAILs above"

step "4. dry run: execute the setup cells with no model"
# Ten seconds, no GPU. This is what stands between a typo in a notebook cell
# and finding out about it three hours into a run on someone else's machine.
cd "$ROOT" || exit 1
"$PY" dry_run_setup.py "$TASK" || die "the notebook does not run from a clean kernel"

step "5. smoke test (a couple of instances, graded automatically)"
"$PY" run_oss20b.py --task "$TASK" --smoke
exit $?
