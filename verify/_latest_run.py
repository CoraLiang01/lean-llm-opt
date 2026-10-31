#!/usr/bin/env python3
"""
_latest_run.py
==============
Print the name of the newest run directory under runs/ that actually holds
instances.jsonl.

    RUN_ID="$(python3 verify/_latest_run.py)"

`ls -t runs | head -1` looks like it does this and does not: it happily returns
a stray .DS_Store, and on an empty runs/ it returns the empty string -- which,
passed to `score_runs.py --run ""`, is not "no runs" but "every run ever made",
re-executing several hundred generated models from unrelated datasets. Exiting
non-zero on nothing-found is the point of this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def main() -> int:
    if not RUNS.is_dir():
        print(f"no runs/ directory at {RUNS}", file=sys.stderr)
        return 1
    cands = [d for d in RUNS.iterdir()
             if d.is_dir() and (d / "instances.jsonl").exists()]
    if not cands:
        print(f"no run directory under {RUNS} contains instances.jsonl -- "
              f"the run produced nothing", file=sys.stderr)
        return 1
    print(max(cands, key=lambda d: (d / "instances.jsonl").stat().st_mtime).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
