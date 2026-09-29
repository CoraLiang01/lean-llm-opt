#!/usr/bin/env python3
"""Run one LEAN Large-scale-or notebook from a repository checkout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    "oss": ROOT / "LEAN_LLM_OPT_gpt_oss_20b_Large-scale.ipynb",
    "gpt41": ROOT / "LEAN_LLM_OPT_4.1_Large-scale.ipynb",
}
INPUTS = (
    ROOT / "Test_Dataset/Large-scale-or/Large-scale-or-101.csv",
    ROOT / "Large_Scale_Or_Files/RAG_Examples_All.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=NOTEBOOKS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="check files without an API or model call")
    args = parser.parse_args()

    missing = [path for path in (*INPUTS, NOTEBOOKS[args.model]) if not path.is_file()]
    if missing:
        parser.error("Missing required files: " + ", ".join(str(path) for path in missing))
    if args.check:
        print(f"Input files present; notebook={NOTEBOOKS[args.model].name}")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for a run")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"Output directory already exists; choose a fresh one: {output_dir}")
    if args.model == "gpt41" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required for GPT-4.1")

    os.environ["LEAN_PROJECT_ROOT"] = str(ROOT)
    os.environ["LEAN_LLM_OPT_ROOT"] = str(ROOT)
    os.environ["LEAN_RESULTS_DIR"] = str(output_dir)
    os.environ["LEAN_RUN_AUTOMATIC"] = "1"
    os.environ["LEAN_RUN_FORCED_ROUTES"] = "0"
    os.environ["LEAN_ROW_START"] = "0"
    os.environ["LEAN_ROW_END"] = "101"
    os.chdir(ROOT)

    notebook_path = NOTEBOOKS[args.model]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{notebook_path}#cell-{index}", "exec"), namespace)
        except Exception:
            print(f"NOTEBOOK_CELL_FAILED={index}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
