#!/usr/bin/env python3
"""
classify_only.py
================
Run just the classification stage over the test set and report where each
instance would be routed. No modelling, no code generation, no solving.

    python verify/classify_only.py                       # as configured
    python verify/classify_only.py --refdata-mode trimmed
    python verify/classify_only.py --rows 70,72,77 --out verify/out/cls.csv

Why a separate runner
---------------------
The question this answers is not "how accurate is the pipeline" but "which
handler does each instance reach", and the second is a hundredth of the price
of the first. Classification is ~5 calls and ~8.4k prompt tokens per instance;
a full 101-instance pass costs about $2 here against $10 for run_full.sh, and
the number it produces is the one that moves.

That number: **how many of the 26 Mixture/Others instances reach
get_Others_response**. Mixture has no specialist of its own, and the ones that
get sent to a specialist do markedly worse. Measured on the run of 2026-08-23
(`...185057__4d5d26b5e733`):

    reached get_Others_response   13/26, of which 11 correct  (85%)
    sent to a type specialist     13/26, of which  5 correct  (38%)

The March run can be counted indirectly -- get_Others_response is the only
handler whose template tells the coder to read the CSV, and 22 of the 26 have
`read_csv` in their recorded code, 19 of them correct. So March routed 22/26
where this routes 13/26, at the same per-handler success rate. Closing that gap
is the point of the exercise.

--refdata-mode
--------------
Switches the classifier's retrieval corpus without editing any file, so both
arms of the comparison run from the same working tree:

    csvloader  (default)  every RefData column, empty ones included -- what
                          CSVLoader produced, and what the classifier saw
                          before the 2026-08-01 instrumentation
    trimmed               drop empty fields, the behaviour that instrumentation
                          introduced: it rewrote all 96 documents, losing
                          `Related` from 94 of them, `Label_Code` from 86 and
                          `Data_address` from 78

Run it once each way and diff the routing.
"""

from __future__ import annotations

import argparse
import contextlib
import csv as csvmod
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

# Keep this run's log out of runs/. load_config() reads LEANOPT_OUT_DIR, and it
# has to be set before the notebook's config cell calls it. Without this a
# classification-only pass leaves a directory that looks like a pipeline run:
# _latest_run.py would offer it to score_runs.py, which would find no code to
# execute and report every instance as wrong.
import os
os.environ.setdefault("LEANOPT_OUT_DIR", str(HERE / "verify/out/classify_runs"))

NOTEBOOK = HERE / "LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
DATA = HERE / "Test_Dataset/Large-scale-or/Large-scale-or-101.csv"


# --------------------------------------------------------------------------- #
# Notebook plumbing. Same approach as run_notebook.py: execute the setup cells,
# stop before the cell that would run the whole test set.
def code_cells(nb: dict):
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if src.strip():
            yield i, src


def setup_cells(nb: dict):
    """Cells up to, but not including, the first call of run_test()."""
    call = re.compile(r"^(?!\s*def\s)\s*(?:\w+\s*(?:,\s*\w+\s*)*=\s*)?"
                      r"run_test\s*\(", re.M)
    out = []
    for i, src in code_cells(nb):
        if call.search(src):
            return out
        out.append((i, src))
    raise SystemExit("no cell calls run_test(); cannot drive this notebook")


def extract_problem_type_fn(nb: dict):
    """Lift `extract_problem_type` out of run_test's body and compile it.

    It is defined inside run_test, so it cannot be imported, and copying the
    regex here would leave two versions to drift apart -- the whole point is to
    measure the classifier the pipeline actually uses. Pull the source instead.
    """
    for _, src in code_cells(nb):
        # [ \t]* rather than \s*: with re.M, \s* happily swallows the
        # preceding newline into group(1), which makes the measured indent one
        # too deep and the body scan stop on its own first line.
        m = re.search(r"^([ \t]*)def extract_problem_type\(", src, re.M)
        if not m:
            continue
        indent, start = len(m.group(1)), m.start()
        lines, body = src[start:].splitlines(keepends=True), []
        for n, line in enumerate(lines):
            if n and line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            body.append(line)
        code = "".join(body)
        code = "\n".join(l[indent:] if l.startswith(" " * indent) else l
                         for l in code.splitlines())
        ns = {"re": re}
        exec(compile(code, "<extract_problem_type>", "exec"), ns)  # noqa: S102
        return ns["extract_problem_type"]
    raise SystemExit("extract_problem_type not found in the notebook")


# --------------------------------------------------------------------------- #
#: Mirror of the if/elif chain in run_test (cell 23). Kept as data so the two
#: can be diffed by eye; if that chain gains a branch, add it here too.
ROUTES = [
    ("NRM handler", ("Network Revenue Management", "NRM",
                     "Network Revenue Management Problem")),
    ("RA handler", ("Resource Allocation", "RA", "Resource Allocation Problem")),
    ("TP handler", ("Transportation", "TP", "Transportation Problem")),
    ("FLP handler", ("Facility Location Problem", "FLP",
                     "Uncapacited Facility Location", "UFLP")),
    ("AP handler", ("Assignment Problem", "AP")),
]


def route_for(selected: str | None, has_csv: bool) -> str:
    """Which handler run_test would call. `None` falls through, as it does there."""
    if not has_csv:
        return "Others-without-CSV handler"
    for name, names in ROUTES:
        if selected in names:
            return name
    return "Others handler"          # the else branch, incl. Mixture and None


def group(gold: str) -> str:
    g = str(gold)
    if g.startswith("UFLP"):
        return "FLP"
    if g.startswith("Mixture") or g.startswith("Others"):
        return "Others"
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdata-mode", choices=("csvloader", "trimmed"),
                    default="csvloader")
    ap.add_argument("--rows", help="comma-separated row labels; default all")
    ap.add_argument("--out", type=Path,
                    default=HERE / "verify/out/classification.csv")
    ap.add_argument("--list", action="store_true",
                    help="show the plan and exit; spends nothing")
    args = ap.parse_args()

    import pandas as pd
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = setup_cells(nb)
    df = pd.read_csv(DATA)
    if args.rows:
        df = df.loc[[int(r) for r in args.rows.split(",") if r.strip()]]

    print(f"notebook     : {NOTEBOOK.name}")
    print(f"setup        : {len(cells)} cell(s), up to cell {cells[-1][0]}")
    print(f"instances    : {len(df)}")
    print(f"refdata mode : {args.refdata_mode}")
    if args.list:
        print("\n--list: nothing executed, nothing spent.")
        return 0

    import os
    os.chdir(HERE)                       # the notebook uses repo-relative paths

    ns: dict = {"__name__": "__main__"}
    for idx, src in cells:
        if src.strip().startswith(("!", "%")):
            continue
        exec(compile(src, f"<cell {idx}>", "exec"), ns)      # noqa: S102
        # Between the config cell and the cell that builds the classifier's
        # vector store, so the switch lands before the corpus is embedded.
        if "lx.load_config(" in src and args.refdata_mode == "trimmed":
            cfg = ns["CFG"]
            cfg.classification = dict(cfg.classification)
            cfg.classification["skip_empty_refdata_fields"] = True
            print("[classify_only] refdata: dropping empty fields")

    agent = ns.get("classification_agent")
    if agent is None:
        raise SystemExit("classification_agent not defined by the setup cells")
    extract = extract_problem_type_fn(nb)
    lx = ns["lx"]

    qcol = "Problem Type" if "Problem Type" in df.columns else "Type"
    log = ns.get("LOG")
    rows = []
    print(f"\n{'id':>4} {'gold':<10} {'predicted':<26} route")
    for i, row in df.iterrows():
        query = row["Query"]
        has_csv = "Dataset_address" in row.index and str(
            row.get("Dataset_address") or "").strip() != ""
        # Inside log.instance() so the calls are attributed and costed: the
        # tracker accumulates into the current record, and without one the
        # run closes with nothing to total.
        ctx = (log.instance(instance_id=i, query=str(query),
                            gold_type=row[qcol])
               if log is not None else contextlib.nullcontext())
        try:
            with ctx:
                with lx.stage("classification"):
                    resp = lx.call_with_retry(
                        agent.invoke, ns["CFG"],
                        f"What is the problem type of the text? text:{query}")
                    selected = extract(resp["output"])
        except Exception as e:                                # noqa: BLE001
            selected, resp = None, {"output": f"ERROR: {e}"}
        r = route_for(selected, has_csv)
        rows.append({"instance_id": i, "gold_type": row[qcol],
                     "group": group(row[qcol]), "predicted": selected or "",
                     "route": r})
        print(f"{i:>4} {str(row[qcol])[:10]:<10} {str(selected)[:26]:<26} {r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- the number this exists to produce ---------------------------------
    oth = [r for r in rows if r["group"] == "Others"]
    generic = sum(1 for r in oth if r["route"].startswith("Others"))
    print("\n" + "=" * 70)
    print(f"Mixture/Others reaching a generic handler: {generic}/{len(oth)}")
    print(f"   2026-08-23 run: 13/26     March (counted via read_csv): 22/26")
    if oth:
        print("\n   where the rest went:")
        for k, v in Counter(r["route"] for r in oth
                            if not r["route"].startswith("Others")).most_common():
            print(f"     {k:<28} {v}")

    print("\nrouting by gold type")
    print("-" * 70)
    by = defaultdict(Counter)
    for r in rows:
        by[r["group"]][r["route"]] += 1
    for g in sorted(by):
        tot = sum(by[g].values())
        top = ", ".join(f"{k.replace(' handler','')}={v}"
                        for k, v in by[g].most_common())
        print(f"  {g:<8} n={tot:<4} {top}")

    print(f"\nper-instance detail -> {args.out}")
    if log is not None:
        agg = log.close()
        # close() returns only {run_id, n_instances} when nothing was recorded
        if "total_cost_usd" in agg:
            print(f"cost: ${agg['total_cost_usd']:.4f} over "
                  f"{agg['n_instances']} instance(s)  ->  {log.dir}")
        else:
            print(f"no instances recorded (n={agg.get('n_instances', 0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
