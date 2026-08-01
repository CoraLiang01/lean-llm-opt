#!/usr/bin/env python3
"""
run_notebook.py
===============
Run a pipeline notebook from the terminal, on a few instances.

    # see what would happen -- no API calls
    python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb --list

    # 2 instances of the CA query set
    python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb \
        --data Test_Dataset/Air_NRM/small_scale/query_CA.csv --n 2

    # specific rows, as the notebook would index them
    python run_notebook.py LEAN_LLM_OPT_4.1_Large-scale-or.ipynb \
        --data Test_Dataset/Large-scale-or/Large-scale-or-101.csv --rows 15,45,81

Why this exists
---------------
Smoke-testing a notebook otherwise means opening it, restarting the kernel,
running fifteen setup cells by hand and remembering not to touch the cell that
runs the whole test set. That is slow and easy to get wrong -- one stray
Shift+Enter on the wrong cell spends real money on 101 instances.

This executes the notebook's setup cells (everything up to, but not including,
the first cell that *calls* the batch function) in one namespace, then calls
that batch function itself on the slice you asked for. Same code, same config
cell, same logging -- the only thing skipped is the cell that would have run
everything.

`jupyter nbconvert --execute` cannot do this: it runs every cell, including the
full test set.

Notes
-----
* Run from the repository root. The notebooks use repo-relative data paths.
* The config cell prompts for an API key if one is not in the environment,
  exactly as it does in Jupyter.
* `--list` stops before any cell executes, so it costs nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The two batch entry points across the notebook family. Air-NRM and the
# ablations expose Batch_Process_Queries(df); the large-scale notebooks expose
# run_test(df, agent). Both are detected at runtime rather than configured.
BATCH_FNS = ("Batch_Process_Queries", "run_test")
CALL_RE = re.compile(r"^(?!\s*def\s)\s*(?:\w+\s*(?:,\s*\w+\s*)*=\s*)?"
                     r"(" + "|".join(BATCH_FNS) + r")\s*\(", re.M)


def code_cells(nb: dict):
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        yield i, src


def split_at_first_call(nb: dict):
    """Setup cells, and the index of the first cell that runs the test set."""
    setup, stop = [], None
    for i, src in code_cells(nb):
        if CALL_RE.search(src):
            stop = i
            break
        setup.append((i, src))
    return setup, stop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--data", help="CSV to run; default = whatever the "
                                   "notebook's own run cell reads")
    ap.add_argument("--n", type=int, help="first N rows")
    ap.add_argument("--rows", help="comma-separated row labels, e.g. 15,45,81")
    ap.add_argument("--query-column", default="Query")
    ap.add_argument("--list", action="store_true",
                    help="show the plan and exit; executes nothing")
    args = ap.parse_args()

    nb_path = (HERE / args.notebook).resolve()
    if not nb_path.exists():
        raise SystemExit(f"notebook not found: {nb_path}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    setup, stop = split_at_first_call(nb)
    if stop is None:
        raise SystemExit(
            f"no cell in {nb_path.name} calls any of {BATCH_FNS}; "
            f"this script does not know how to drive it")

    print(f"notebook : {nb_path.name}")
    print(f"setup    : {len(setup)} code cell(s), up to cell {setup[-1][0]}")
    print(f"skipping : cell {stop} onwards (that is the full-test-set cell)")

    if args.list:
        print("\n--list: nothing executed, nothing spent.")
        return 0

    # The notebooks address their data relative to the repository root.
    import os
    os.chdir(HERE)

    ns: dict = {"__name__": "__main__"}
    for idx, src in setup:
        stripped = src.strip()
        if stripped.startswith(("!", "%")):
            continue                       # shell / magic lines: notebook-only
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)   # noqa: S102
        except Exception as e:                                # noqa: BLE001
            print(f"\n!! setup cell {idx} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise

    fn_name = next((f for f in BATCH_FNS if callable(ns.get(f))), None)
    if fn_name is None:
        raise SystemExit(f"setup finished but none of {BATCH_FNS} is defined")
    fn = ns[fn_name]

    import pandas as pd
    if args.data:
        df = pd.read_csv(HERE / args.data)
    else:
        src = "".join(nb["cells"][stop]["source"])
        m = re.search(r"read_csv\(\s*['\"]([^'\"]+)['\"]", src)
        if not m:
            raise SystemExit("could not infer the data file; pass --data")
        df = pd.read_csv(HERE / m.group(1))
        print(f"data     : {m.group(1)} (inferred from cell {stop})")

    if args.rows:
        labels = [int(r) for r in args.rows.split(",") if r.strip()]
        missing = [r for r in labels if r not in df.index]
        if missing:
            raise SystemExit(f"row labels not in the dataset: {missing}")
        df = df.loc[labels]          # .loc, not reset_index: instance ids must
    elif args.n:                     # stay unique across batches
        df = df.head(args.n)

    print(f"running  : {len(df)} instance(s) through {fn_name}()\n")

    if fn_name == "Batch_Process_Queries":
        result = fn(df, query_column=args.query_column)
    else:
        agent = next((ns[k] for k in ("classification_agent", "agent",
                                      "classify_problem")
                      if callable(ns.get(k))), None)
        if agent is None:
            raise SystemExit("run_test needs a classification agent and none "
                             "was found in the notebook namespace")
        result = fn(df, agent)

    log = ns.get("LOG")
    if log is not None:
        agg = log.close()
        print(f"\n{agg['n_instances']} instance(s) | "
              f"${agg['total_cost_usd']:.4f} | {agg['total_wall_s']:.0f}s")
        print(f"log: {log.dir}")

    if isinstance(result, tuple):
        result = result[0]
    try:
        print(f"\nresult frame: {len(result)} row(s)")
    except Exception:                                        # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
