#!/usr/bin/env python3
"""
array_lengths.py
================
Whether the modelling stage is still losing values out of its long arrays.

    python verify/array_lengths.py runs/<run_id>

This is the direct read-out for the CSVQA change, and it is worth more than the
accuracy on a small slice. The defect is specific: arrays that index the same
set come out of the modelling stage with different lengths, because the model
re-typed them and dropped a few. Instance 11 produced w=117, d=115, c=115 from
a 119-row file; instance 20, r=52, I=51, d=52 from 68 rows. Whether that stops
is a fact about the run, not a coin flip.

Arrays of genuinely different lengths are normal and are not counted: a
transportation instance has 10 supplies and 20 demands. The signal is arrays
whose lengths differ by 10% or less -- close enough that they were meant to
match. On the 2026-08-23 baseline that flagged 8 instances, which scored 1/8
against 44/54 for everything else. One of the 8 (instance 1, d=12 supplies vs
s=11 demands) is a false positive, so read the list, not just the count.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ARRAY_RE = re.compile(r'(\w+)\s*=\s*\[([^\]\[]*)\]')


def arrays(text: str) -> dict:
    out = {}
    for m in ARRAY_RE.finditer(text or ''):
        vals = [x for x in m.group(2).split(',') if x.strip()]
        if len(vals) >= 5:
            out.setdefault(m.group(1), len(vals))
    return out


def near_miss(lengths) -> bool:
    """Two arrays close in length but unequal: meant to match, one lost values."""
    L = sorted(set(lengths))
    return any(x < y and (y - x) / y <= 0.10 for x in L for y in L)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: array_lengths.py runs/<run_id>")
    d = Path(sys.argv[1])
    recs = {json.loads(l)['instance_id']: json.loads(l)
            for l in (d / 'instances.jsonl').read_text().splitlines() if l.strip()}
    scored = d / 'scored.csv'
    S = ({int(r['instance_id']): r for r in csv.DictReader(open(scored))}
         if scored.exists() else {})

    flagged, clean = [], []
    for i in sorted(recs):
        a = arrays(recs[i].get('model_output'))
        if len(a) < 2:
            continue
        (flagged if near_miss(a.values()) else clean).append((i, a))

    def ok(i):
        return (i in S) and str(S[i]['correct_optimal']).startswith('1')

    print("  arrays that should have matched and did not:")
    if not flagged:
        print("    none")
    for i, a in flagged:
        err = (S[i]['solve_error'] or '')[:34] if i in S else ''
        print(f"    {i:>4} {'OK' if ok(i) else '. ':>3}  "
              f"{str(a)[:58]:<58} {err}")

    nf, nc = len(flagged), len(clean)
    print(f"\n  flagged {nf}, clean {nc}")
    if S:
        if nf:
            print(f"  accuracy when flagged : {sum(ok(i) for i, _ in flagged)}/{nf}")
        if nc:
            print(f"  accuracy when clean   : {sum(ok(i) for i, _ in clean)}/{nc}")
    print("  (baseline over the full 101: 8 flagged at 1/8, 54 clean at 44/54)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
