#!/usr/bin/env python3
"""
compare_arms.py
===============
Put several runs side by side on the metrics that actually differ between them.

    python compare_arms.py                       # every run in scored_all.csv
    python compare_arms.py --run b533,b5a6,c5ba  # just these (substring match)
    python compare_arms.py --class Mixture       # add a per-class breakdown

Why this exists
---------------
`aggregate_runs.py` reports one configuration at a time, into its own table
directory. Comparing four arms then means opening eight CSVs and lining up
numbers by hand, which is where the mistakes come from. This reads the single
`scored_all.csv` that scoring already maintains and prints one table.

It reports two accuracies that move independently, and that is the point:

  cls_acc     did the classifier name the right problem class?
  e2e_acc     did the pipeline reach the right optimal objective?

They are not the same question and they do not have to move together. A class
can be misrouted and still solved -- 13 of the 18 Mixture instances were solved
correctly while being labelled something else -- so an improvement in `cls_acc`
can arrive with no change, or a fall, in `e2e_acc`. Reporting only one of them
hides that.

Read the differences against the noise floor, not against zero: on this
benchmark 19 of 101 instances flip between two runs of an *identical* config,
so a gap of a few instances is not a result. `--noise` prints the reminder with
the arithmetic done for the current n.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE))
from aggregate_runs import _canon_map, canonical_type   # noqa: E402


def load_cls_cfg(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (raw.get("common", {}) or {}).get("classification", {}) or {}


def arm_label(run_id: str) -> str:
    """Short, human-readable name for a run: the hash is what identifies it."""
    return run_id.rsplit("__", 1)[-1][:12]


def summarise(df: pd.DataFrame, canon: dict, others_prefix: bool,
              focus: str | None) -> pd.DataFrame:
    rows = []
    for run_id, g in df.groupby("run_id", sort=False):
        gold = g.gold_type.map(lambda v: canonical_type(v, canon, others_prefix))
        pred = g.pred_type.map(lambda v: canonical_type(v, canon, others_prefix))
        # A missing prediction is a wrong prediction, not an excluded instance:
        # the pipeline was asked and produced nothing usable.
        both = gold.notna()
        cls_ok = (gold == pred) & both
        e2e_ok = g.correct_optimal.fillna(0).astype(float) > 0

        row = {
            "arm": arm_label(str(run_id)),
            "n": len(g),
            "cls_acc_%": round(100 * cls_ok.sum() / max(1, both.sum()), 1),
            "e2e_acc_%": round(100 * e2e_ok.sum() / max(1, len(g)), 1),
            "e2e_n": int(e2e_ok.sum()),
        }
        if focus:
            sel = gold == focus
            row[f"{focus}_n"] = int(sel.sum())
            row[f"{focus}_pred"] = int((pred == focus).sum())
            row[f"{focus}_recall_%"] = (
                round(100 * (cls_ok & sel).sum() / max(1, sel.sum()), 1))
            row[f"{focus}_e2e_%"] = (
                round(100 * (e2e_ok & sel).sum() / max(1, sel.sum()), 1))
        row["run_id"] = str(run_id)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_all.csv")
    ap.add_argument("--config", default="exp_config.yaml")
    ap.add_argument("--run", help="comma-separated substrings of run_id; "
                                  "order is preserved in the output")
    ap.add_argument("--class", dest="focus", default="Mixture",
                    help="canonical class to break out; '' to disable")
    ap.add_argument("--noise", type=int, default=19,
                    help="instances observed to flip between identical runs "
                         "(default 19, measured on Large-Scale-OR n=101)")
    args = ap.parse_args()

    scored = Path(args.scored)
    if not scored.exists():
        raise SystemExit(f"{scored} not found -- run score_runs.py first")
    df = pd.read_csv(scored)

    if args.run:
        keys = [k.strip() for k in args.run.split(",") if k.strip()]
        parts = [df[df.run_id.str.contains(k, na=False)] for k in keys]
        missing = [k for k, p in zip(keys, parts) if p.empty]
        if missing:
            raise SystemExit(f"no runs matched: {', '.join(missing)}")
        df = pd.concat(parts, ignore_index=True)

    cls_cfg = load_cls_cfg(Path(args.config))
    canon = _canon_map(cls_cfg)
    others_prefix = bool(cls_cfg.get("others_prefix_counts_as_others", True))
    focus = args.focus or None

    t = summarise(df, canon, others_prefix, focus)
    show = [c for c in t.columns if c != "run_id"]
    print(t[show].to_string(index=False))

    print("\nrun_id:")
    for _, r in t.iterrows():
        print(f"  {r['arm']}  {r['run_id']}")

    n = int(t.n.max()) if len(t) else 0
    if n:
        # `noise` counts instances whose outcome changed between two runs of an
        # identical config. That is not the error bar: flips go both ways and
        # mostly cancel. Treating them as a random walk, the standard deviation
        # of the *net* difference is about sqrt(noise) instances -- so call a
        # gap meaningful only past roughly two of those.
        sd = args.noise ** 0.5
        print(f"\nNoise floor: {args.noise} of {n} instances changed outcome "
              f"between two runs of an identical configuration.")
        print(f"  Flips cancel, so the net difference has sd ~sqrt({args.noise})"
              f" = {sd:.1f} instances ({100 * sd / n:.1f} pp).")
        print(f"  Treat a gap as real only past ~{2 * sd:.0f} instances "
              f"({100 * 2 * sd / n:.1f} pp), and confirm it with "
              f"--repeat-index 1,2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
