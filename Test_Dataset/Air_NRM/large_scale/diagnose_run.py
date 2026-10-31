#!/usr/bin/env python3
"""
diagnose_run.py
===============
Say why each Air-NRM-CA-LS instance got the objective it did.

    python diagnose_run.py                     # newest Air-NRM-CA-LS run
    python diagnose_run.py --run <run_id>
    python diagnose_run.py --instance 12       # one instance, in detail

`score_runs.py` reports correct / not correct, and that verdict is the one that
counts. Everything below is about what to fix next, not about the score: an
instance whose balance constraint carries v1 where v2 belongs is a wrong model
and a wrong answer, however it came to be written that way. Do not read these
categories as partial credit.

What they are for is that the fixes differ. A coefficient copied from the
neighbouring dictionary points at how the parameters are handed over; a
reversed inequality points at the exemplar. Prompt work will not fix the first
and shortening the parameter dump will not fix the second. Sorting them by hand
takes twenty minutes an instance; this does it from the log.

For every instance it re-extracts the generated code, executes the parameter
assignments only, and compares all eight dictionaries against
SBLP_CA_Label/CA_parameters.json -- the dump the labels were solved from. Then:

  * every dictionary matches            -> the formulation is what differs
  * a dictionary has wrong entries      -> report how many, from which position,
                                           and whether the wrong values happen to
                                           equal another dictionary's, which is
                                           the signature of the model losing its
                                           place while transcribing two
                                           same-keyed dicts side by side
  * a dictionary is missing entirely    -> the answer was truncated

Nothing here calls an LLM.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = "LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb"
DATASET = "Air-NRM-CA-LS"
GROUPS = ["avg_price", "value_list", "ratio_list", "value_0_list",
          "ratio_0_list", "avg_pax", "capacity_consum", "flight_capacity"]


def repo_root() -> Path:
    root = HERE
    while not (root / "exp_config.yaml").exists() and root != root.parent:
        root = root.parent
    if not (root / "exp_config.yaml").exists():
        sys.exit("ERROR: not inside the lean-llm-opt-main tree")
    return root


def load_extractor(nb_path: Path):
    """Reuse the notebook's own extractor, so this sees what scoring saw."""
    nb = json.loads(nb_path.read_text())
    cell = next(("".join(c["source"]) for c in nb["cells"]
                 if "_extract_gurobipy_code" in "".join(c["source"])), None)
    if cell is None:
        sys.exit(f"ERROR: no _extract_gurobipy_code in {nb_path.name}")
    ns = {"_re": re}
    exec(cell[cell.find("_FENCE_RE"): cell.find("\ndef Batch_Process_Queries")], ns)
    return ns["_extract_gurobipy_code"]


def cell_key(k):
    od, time, short = ast.literal_eval(k)
    origin, destination = ast.literal_eval(od)
    return f"({origin}{destination},{time},{short})"


def od_key(k):
    inner = ast.literal_eval(k)
    if isinstance(inner, tuple):
        inner = inner[0]
    return "".join(ast.literal_eval(inner)) if isinstance(inner, str) else "".join(inner)


def flight_key(k):
    od, time = ast.literal_eval(k)
    origin, destination = ast.literal_eval(od)
    return f"({origin}{destination},{time})"


KEYMAP = {"avg_price": cell_key, "value_list": cell_key, "ratio_list": cell_key,
          "value_0_list": od_key, "ratio_0_list": od_key,
          "avg_pax": lambda k: "".join(ast.literal_eval(k)),
          "flight_capacity": flight_key, "capacity_consum": lambda k: k}


def compare(got: dict, truth: dict, name: str, all_truth: dict):
    """Per-dictionary verdict: (n_wrong, n_missing, first_bad_index, note)."""
    keys = list(truth)
    wrong, missing, first = [], [], None
    for i, k in enumerate(keys):
        mapped = KEYMAP[name](k)
        if mapped not in got:
            missing.append(i)
            continue
        if abs(float(got[mapped]) - float(truth[k])) > 1e-9:
            wrong.append((i, mapped, float(got[mapped]), float(truth[k])))
            if first is None:
                first = i

    note = ""
    if wrong:
        # Did the wrong values come from a sibling dictionary? avg_price,
        # value_list and ratio_list share their keys, so a model that loses its
        # place while transcribing one can land in another.
        for other in ("value_list", "ratio_list", "avg_price"):
            if other == name or other not in all_truth:
                continue
            hits = sum(1 for _, _, g, _ in wrong
                       for k in [keys[_i] for _i, _m, _g, _w in wrong if _g == g][:1]
                       if abs(g - float(all_truth[other].get(k, 1e30))) < 1e-9)
            hits = sum(1 for i, _m, g, _w in wrong
                       if abs(g - float(all_truth[other].get(keys[i], 1e30))) < 1e-9)
            if hits == len(wrong):
                note = f"every wrong value equals this cell's {other}"
                break
            if hits:
                note = f"{hits}/{len(wrong)} wrong values equal this cell's {other}"
                break
        tail = wrong[-1][0]
        if tail - first + 1 == len(wrong):
            note += ("; " if note else "") + \
                f"contiguous, entries {first}-{tail} of {len(keys)}"
    return wrong, missing, first, note


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run_id substring; default newest")
    ap.add_argument("--instance", type=int, default=None)
    ap.add_argument("--notebook", default=NOTEBOOK)
    args = ap.parse_args()

    root = repo_root()
    os.chdir(root)
    runs = sorted(p.parent for p in (root / "runs").glob("*/config.json")
                  if json.loads(p.read_text()).get("dataset") == DATASET
                  and (not args.run or args.run in p.parent.name))
    if not runs:
        sys.exit(f"ERROR: no {DATASET} run found" +
                 (f" matching {args.run!r}" if args.run else ""))
    run = runs[-1]

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        sys.exit("ERROR: gurobipy is needed to re-solve the generated models")
    gp.setParam("OutputFlag", 0)

    extract = load_extractor(HERE / args.notebook)
    truths = json.loads(
        (HERE / "SBLP_CA_Label" / "CA_parameters.json").read_text())

    print(f"run : {run.name}\n")
    header = (f"{'#':>3} {'vars':>5} {'objective':>15} {'gold':>15} {'rel':>9}  verdict")
    print(header)

    detail = []
    for line in (run / "instances.jsonl").read_text().splitlines():
        record = json.loads(line)
        iid = record["instance_id"]
        if args.instance is not None and iid != args.instance:
            continue
        gold = record.get("gold_objective")
        logged_code = record.get("code_output")
        answers = [c[k] for c in record.get("calls", [])
                   for k in ("response", "output", "completion", "text")
                   if isinstance(c.get(k), str) and "```" in c.get(k, "")]
        if (not logged_code and not answers) or gold is None:
            print(f"{iid + 1:>3} {'-':>5} {'-':>15} {gold if gold else '-':>15} "
                  f"{'-':>9}  no answer logged  ({record.get('errors')})")
            continue

        # New runners persist the already extracted/path-resolved code. Use it
        # directly so diagnosis examines exactly what score_runs.py executed;
        # raw provider text remains a compatibility fallback for older runs.
        code = logged_code or extract(max(answers, key=len))
        env = {"gp": gp, "GRB": GRB}
        value = model = None
        # Set up front: a model that builds but ends INFEASIBLE or UNBOUNDED
        # never reaches the except branch, and reporting that needs a reason too.
        fail = "no gurobipy Model in the generated code"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, env)
            model = env.get("model")
            if model is not None:
                if model.Status == GRB.OPTIMAL:
                    value = model.ObjVal
                else:
                    fail = f"solver status {model.Status} (2 = OPTIMAL)"
        except Exception as exc:                              # noqa: BLE001
            value, model = None, None
            fail = f"{type(exc).__name__}: {exc}"

        if value is None:
            # An infeasible model whose parameters are all correct is a
            # formulation error, not a data error, and worth saying so here --
            # the two look identical in the results table.
            extra = ""
            if model is not None and model.Status in (3, 4):
                params = {}
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(code.split("model = gp.Model")[0], params)
                    truth = truths[iid]
                    off = [n for n in GROUPS
                           if n in truth and n in params
                           and compare(params[n], truth[n], n, truth)[0]]
                    extra = ("; every parameter matches, so the formulation "
                             "differs" if not off
                             else f"; wrong parameters in {', '.join(off)}")
                except Exception:                             # noqa: BLE001
                    pass
            print(f"{iid + 1:>3} {'-':>5} {'-':>15} {gold:>15,.2f} {'-':>9}  "
                  f"did not solve -- {fail}{extra}")
            continue

        rel = abs(value - gold) / max(abs(gold), 1.0)
        if rel < 1e-6:
            print(f"{iid + 1:>3} {model.NumVars:>5} {value:>15,.2f} {gold:>15,.2f} "
                  f"{rel:>9.0e}  correct")
            continue

        # Wrong number: find out which parameters differ.
        params = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code.split("model = gp.Model")[0], params)
        truth = truths[iid]
        problems = []
        for name in GROUPS:
            if name not in truth or name not in params:
                continue
            wrong, missing, first, note = compare(
                params[name], truth[name], name, truth)
            if wrong or missing:
                problems.append((name, len(wrong), len(missing), note))
        verdict = ("formulation differs; every parameter matches"
                   if not problems else
                   "; ".join(f"{n}: {w} wrong"
                             + (f", {m} missing" if m else "")
                             + (f" ({note})" if note else "")
                             for n, w, m, note in problems))
        print(f"{iid + 1:>3} {model.NumVars:>5} {value:>15,.2f} {gold:>15,.2f} "
              f"{rel:>9.1e}  {verdict}")
        detail.append((iid, problems))

    if detail and args.instance is None:
        print("\nRerun with --instance <n> for the first wrong entries of a case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
