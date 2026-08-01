#!/usr/bin/env python3
"""
link_air_nrm_labels.py
======================
Attach the Air-NRM ground-truth objectives to their query files.

    python link_air_nrm_labels.py --check     # verify alignment, write nothing
    python link_air_nrm_labels.py             # add the Label-objective column

Why this exists
---------------
Air-NRM has always had ground truth -- `SBLP_CA_Label/CA_answer.csv` and
`SBLP_NP_Flow_Label/np_obj.csv`, plus one `.lp` per instance. But the query
files that everything else reads carry only a `Query` column, so nothing could
reach it: `run_baseline.py` logged no `gold_objective`, `score_runs.py` had
nothing to compare against, and Air-NRM appeared in the tables as a dataset
with no measurable accuracy.

That is a wiring gap, not a missing measurement. This script closes it by
copying `Obj_label` into the query CSV as `Label-objective` -- the name
`exp_config.yaml:dataset_columns` already resolves.

Alignment is by row position, which is the only correspondence the files
offer, so it is verified rather than assumed: each query names a set of
departure times, and so does the matching label row's parameter dump. The two
sets must be equal for every row, and a deliberately shifted pairing must do
noticeably worse -- otherwise the check has no power to detect misalignment and
the run aborts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / "Test_Dataset" / "Air_NRM" / "small_scale"

# (query csv, label csv, column holding the objective, column holding the
#  parameter dump used for the alignment check)
PAIRS = [
    (BASE / "query_CA.csv",
     BASE / "SBLP_CA_Label" / "CA_answer.csv",
     "Obj_label", "information"),
    (BASE / "query_NP_Flow.csv",
     BASE / "SBLP_NP_Flow_Label" / "np_obj.csv",
     "Obj_label", "Information_label"),
]

TARGET_COL = "Label-objective"

TIME_IN_QUERY = re.compile(r"\d{2}:\d{2}")
TIME_IN_LABEL = re.compile(r"'(\d{2}:\d{2})'")


def check_alignment(q: pd.DataFrame, a: pd.DataFrame, info_col: str):
    """Return (n_matching, n_rows, shifted_score) for the row-position pairing."""
    def qt(i):
        return set(TIME_IN_QUERY.findall(str(q.iloc[i]["Query"])))

    def at(i):
        return set(TIME_IN_LABEL.findall(str(a.iloc[i][info_col])))

    n = min(len(q), len(a))
    match = sum(1 for i in range(n) if qt(i) and qt(i) == at(i))

    # Same comparison with the labels shifted by one. If shifting scores about
    # as well, the check cannot tell aligned from misaligned and proves nothing.
    shifted = sum(1 for i in range(n - 1) if qt(i) and qt(i) == at(i + 1))
    return match, n, shifted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify alignment and exit without writing")
    args = ap.parse_args()

    failures = 0
    for qp, ap_, obj_col, info_col in PAIRS:
        if not qp.exists() or not ap_.exists():
            print(f"  !! missing: {qp if not qp.exists() else ap_}")
            failures += 1
            continue

        q = pd.read_csv(qp)
        a = pd.read_csv(ap_)
        print(f"\n{qp.name}  <-  {ap_.name}")
        print(f"  rows: {len(q)} queries, {len(a)} labels")

        if len(q) != len(a):
            print(f"  !! row counts differ -- refusing to guess an alignment")
            failures += 1
            continue

        match, n, shifted = check_alignment(q, a, info_col)
        print(f"  departure-time sets match on {match}/{n} rows "
              f"(shifted by one: {shifted}/{n - 1})")

        if match != n:
            print("  !! not every row matches -- not writing")
            failures += 1
            continue
        if shifted >= match:
            print("  !! a shifted pairing scores as well, so this check cannot "
                  "detect misalignment -- not writing")
            failures += 1
            continue

        if args.check:
            print("  --check: alignment verified, nothing written")
            continue

        q[TARGET_COL] = a[obj_col].values
        q.to_csv(qp, index=False)
        print(f"  wrote {TARGET_COL} -> {qp.relative_to(HERE)}  "
              f"(first value {a[obj_col].iloc[0]})")

    if failures:
        print(f"\n{failures} pair(s) failed; nothing was written for those.",
              file=sys.stderr)
        return 1
    print("\nDone." if not args.check else "\nAll pairs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
