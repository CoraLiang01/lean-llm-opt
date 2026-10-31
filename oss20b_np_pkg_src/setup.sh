#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Create a private environment; never install into the server's shared Python.
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Please use Python 3.11: PYTHON_BIN=python3.11 bash setup.sh"'
case "${1:---smoke}" in
  --smoke) "$PYTHON_BIN" check_server.py ;;
  --offline) ;;
  *) echo 'Usage: bash setup.sh [--smoke|--offline]' >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-server.txt
.venv/bin/python -m pip check
.venv/bin/python verify_package.py

case "${1:---smoke}" in
  --offline)
    echo 'Offline checks passed. No Ollama connection or inference was attempted.'
    ;;
  --smoke)
    # Models are managed by the server owner; an existing remote Ollama works too.
    .venv/bin/python -u run_airnrm_np.py --smoke
    ;;
  *)
    echo 'Usage: bash setup.sh [--smoke|--offline]' >&2
    exit 2
    ;;
esac
