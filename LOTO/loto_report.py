#!/usr/bin/env python3
"""
loto_report.py
==============
Assemble the LOTO result tables from a directory of runs.

    # after score_runs.py has produced scored.csv in each run directory
    python score_runs.py runs/
    python loto_report.py runs/ -o tables/

    # routing tables only -- works before scoring, costs nothing
    python loto_report.py runs/

Produces
--------
tables/loto_per_query.csv   one row per (instance, seed): what it was, what the
                            classifier called it, where it was routed, whether
                            the objective value came out right
tables/loto_routing.csv     table E4-2 -- held-out type x routed-to workflow
tables/loto_accuracy.csv    table E4-1 row -- per-seed accuracy, mean, sd

Notes
-----
* Reads ``holdout_type`` from each run's own ``config.json``, so a directory
  holding several folds is grouped correctly without being told which is which.
* ``correct_optimal`` comes from ``scored.csv``, which ``score_runs.py`` writes
  into each run directory. Without it the accuracy table is skipped and the
  routing tables are still produced.
* Standard deviation is the sample sd across seeds (ddof=1); with a single seed
  it is reported as empty rather than 0.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def read_config_holdout(run_dir: Path) -> str:
    """The fold this run belongs to, or 'none' for a full (A0) run."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return "unknown"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cls = cfg.get("common", {}).get("classification") or cfg.get("classification", {})
    return str(cls.get("holdout_type") or "none")


def read_scored(run_dir: Path) -> dict[int, int]:
    """instance_id -> correct_optimal (0/1), empty if the run is unscored."""
    scored = run_dir / "scored.csv"
    if not scored.exists():
        return {}
    df = pd.read_csv(scored)
    if "instance_id" not in df.columns or "correct_optimal" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        val = row["correct_optimal"]
        if pd.notna(val):
            out[int(row["instance_id"])] = int(val)
    return out


def collect(runs_dir: Path) -> pd.DataFrame:
    records = []
    run_dirs = sorted(p for p in runs_dir.iterdir() if (p / "instances.jsonl").exists())
    if not run_dirs:
        raise SystemExit(f"no run directories with instances.jsonl under {runs_dir}")

    n_scored = 0
    for run_dir in run_dirs:
        holdout = read_config_holdout(run_dir)
        scored = read_scored(run_dir)
        if scored:
            n_scored += 1
        for line in (run_dir / "instances.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            iid = r.get("instance_id")
            records.append({
                "holdout_type": holdout,
                "run_id": r.get("run_id", run_dir.name),
                "seed": r.get("repeat_index"),
                "instance_id": iid,
                "gold_type": r.get("gold_type"),
                "pred_type": r.get("pred_type"),
                # The gpt-oss pipeline collapses any answer outside its
                # category list to "Others with CSV" before the batch loop
                # sees it, so pred_type alone could never show the model
                # uttering a held-out type -- named_holdout would read 0 by
                # construction rather than by evidence. That pipeline logs the
                # pre-collapse answer as pred_type_raw; use it when present.
                "pred_type_raw": r.get("pred_type_raw"),
                "routed_to": r.get("routed_to"),
                "status": r.get("status"),
                "correct_optimal": scored.get(iid),
                "named_holdout": _named_holdout(
                    r.get("pred_type_raw") or r.get("pred_type"), holdout),
            })

    print(f"runs           : {len(run_dirs)} ({n_scored} scored)")
    if n_scored == 0:
        print("                 no scored.csv found -- run `python score_runs.py runs/` "
              "to fill in accuracy")
    return pd.DataFrame(records)


def _named_holdout(pred, holdout) -> bool:
    """Did the model utter the held-out type name despite it being removed?"""
    if not pred or holdout in ("none", "unknown"):
        return False
    return str(holdout).strip().lower() in str(pred).strip().lower()


def routing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table E4-2: held-out type x workflow actually routed to."""
    return (pd.crosstab(df["holdout_type"], df["routed_to"])
              .rename_axis(index="holdout_type", columns="routed_to"))


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Table E4-1: per-seed accuracy for each fold, plus mean and sd."""
    if df["correct_optimal"].isna().all():
        return None

    rows = []
    for holdout, g in df.groupby("holdout_type"):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            vals = gs["correct_optimal"].dropna()
            if len(vals):
                per_seed.append((seed, 100.0 * vals.mean(), len(vals)))
        if not per_seed:
            continue
        accs = [a for _, a, _ in per_seed]
        row = {
            "holdout_type": holdout,
            "n_instances": per_seed[0][2],
            "n_seeds": len(per_seed),
            "mean_acc_%": round(statistics.mean(accs), 1),
            "sd_acc_%": round(statistics.stdev(accs), 1) if len(accs) > 1 else "",
        }
        for seed, acc, _ in per_seed:
            row[f"seed{seed}_%"] = round(acc, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument("-o", "--out", default=None,
                    help="directory to write the CSVs into (default: print only)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    df = collect(runs_dir)

    df = df.sort_values(["holdout_type", "instance_id", "seed"])

    print(f"records        : {len(df)}")
    print(f"folds          : {sorted(df['holdout_type'].unique())}")
    print(f"seeds per fold : "
          f"{dict(df.groupby('holdout_type')['seed'].nunique())}")
    failed = (df["status"] != "ok").sum()
    if failed:
        print(f"failed         : {failed} instance-run(s) did not complete")

    print("\n" + "=" * 66)
    print("ROUTING  (table E4-2)")
    print("=" * 66)
    rt = routing_table(df)
    print(rt.to_string())

    print("\nper-instance routing stability across seeds:")
    for holdout, g in df.groupby("holdout_type"):
        for iid, gi in g.groupby("instance_id"):
            counts = Counter(gi["routed_to"])
            flag = "" if len(counts) == 1 else "   <-- unstable"
            summary = ", ".join(f"{k}x{v}" for k, v in counts.most_common())
            print(f"  [{holdout}] instance {iid}: {summary}{flag}")

    n_named = int(df["named_holdout"].sum())
    print(f"\nnamed_holdout  : {n_named} / {len(df)} predictions uttered the "
          f"removed type name")

    acc = accuracy_table(df)
    print("\n" + "=" * 66)
    print("ACCURACY  (table E4-1)")
    print("=" * 66)
    if acc is None:
        print("not available -- no scored.csv yet. Run:")
        print(f"    python score_runs.py {runs_dir.name}/")
        print("then re-run this script.")
    else:
        print(acc.to_string(index=False))
        print("\nnote: n=5 folds carry ~20pp per-instance granularity; "
              "report them as qualitative observations only (protocol section 6.3).")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "loto_per_query.csv", index=False)
        rt.to_csv(out / "loto_routing.csv")
        if acc is not None:
            acc.to_csv(out / "loto_accuracy.csv", index=False)
        written = ["loto_per_query.csv", "loto_routing.csv"]
        if acc is not None:
            written.append("loto_accuracy.csv")
        print(f"\nwrote {', '.join(written)} to {out}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
