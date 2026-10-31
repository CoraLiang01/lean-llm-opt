#!/usr/bin/env python3
"""Validate or run the tuned small-scale Air-NRM NP workflow."""

from __future__ import annotations

import argparse
import contextlib
import getpass
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
    format_parameters,
)
from Test_Dataset.Air_NRM.small_scale.np_lean_workflow import (
    AirNrmSmallScaleData,
    SMALL_EXAMPLE_MODEL_CODE,
    parse_np_query,
)


QUERY = HERE / "query_NP_Flow.csv"
NOTEBOOK = HERE / "LEAN_LLM_OPT_4.1_Air_NRM_NP_SmallScale.ipynb"
BUILDER = HERE / "build_np_lean_notebook.py"


def offline_check() -> None:
    rows = pd.read_csv(QUERY)
    if list(rows.columns) != ["Decision Variables", "Query", "Label-objective"]:
        raise ValueError("unexpected small-scale NP query columns")
    if len(rows) != 21:
        raise ValueError(f"expected 21 NP queries, got {len(rows)}")

    data = AirNrmSmallScaleData.from_path(HERE)
    worst = 0.0
    for idx, row in rows.iterrows():
        parsed = parse_np_query(row["Query"])
        parameters = data.assemble(row["Query"])
        expected_vars = (len(parsed.flights) * (len(parameters["products"]) + 1)
                         + len(parameters["avg_pax"]))
        if int(row["Decision Variables"]) != expected_vars:
            raise ValueError(f"row {idx}: variable count mismatch")
        code = format_parameters(parameters) + "\n" + SMALL_EXAMPLE_MODEL_CODE
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, f"<small-np-{idx}>", "exec"), namespace)  # noqa: S102
        objective = float(namespace["model"].ObjVal)
        gold = float(row["Label-objective"])
        relative = abs(objective - gold) / max(1.0, abs(gold))
        worst = max(worst, relative)
        if relative > 1e-6:
            raise ValueError(f"row {idx}: objective differs by {relative:.3e}")
        namespace["model"].dispose()

    subprocess.run([sys.executable, str(BUILDER), "--check"], cwd=ROOT, check=True)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    required = (
        "Air-NRM-NP", "install_np_workflow", "NP_SET_CURRENT_QUERY",
        "validated_direct_LLM_code", "query_NP_Flow.csv",
    )
    missing = [marker for marker in required if marker not in notebook_source]
    if missing:
        raise ValueError(f"NP notebook missing markers: {missing}")
    if "query_CA.csv" in notebook_source:
        raise ValueError("NP-only notebook still has a CA execution path")

    print("OK: reference CSV values passed for 21/21 small-scale NP queries")
    print(f"OK: original big-M model reproduced 21/21 labels; worst relative error {worst:.3e}")
    print("OK: dedicated NP notebook is reproducible and contains no CA fallback")
    print("No model API was called.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="offline validation; no API")
    parser.add_argument("--list", action="store_true", help="show plan; no API")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row", type=int, help="one 1-based query number (1-21)")
    selection.add_argument("--n", type=int, help="first N queries")
    parser.add_argument("--profile", default="gpt-4.1")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--prompt-key", action="store_true",
                        help="prompt securely; use key for this process only")
    args = parser.parse_args()

    if args.check:
        offline_check()
        return 0
    if args.row is not None and not 1 <= args.row <= 21:
        raise SystemExit("--row must be between 1 and 21")
    if args.n is not None and not 1 <= args.n <= 21:
        raise SystemExit("--n must be between 1 and 21")

    env = os.environ.copy()
    if args.prompt_key and not args.list:
        key = getpass.getpass("OPENAI_API_KEY (hidden; not saved): ").strip()
        if not key:
            raise SystemExit("empty API key")
        env["OPENAI_API_KEY"] = key
        print("OPENAI_API_KEY accepted: ***" + key[-4:])

    command = [
        sys.executable,
        str(ROOT / "run_notebook.py"),
        str(NOTEBOOK.relative_to(ROOT)),
        "--data", str(QUERY.relative_to(ROOT)),
        "--dataset", "Air-NRM-NP",
        "--profile", args.profile,
        "--repeat", str(args.repeat),
    ]
    if args.row is not None:
        command += ["--rows", str(args.row - 1)]
    elif args.n is not None:
        command += ["--n", str(args.n)]
    if args.resume is not None:
        command += ["--resume"] if args.resume == "auto" else ["--resume", args.resume]
    if args.list:
        command.append("--list")
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
