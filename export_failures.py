#!/usr/bin/env python3
"""
export_failures.py
==================
Pull the instances a run got wrong into something a person can actually read.

    python export_failures.py --run 201027
    python export_failures.py --run 201027 -o failures/ --all

Everything needed is already in runs/<run_id>/instances.jsonl -- the query, the
model's answer, the generated code, the ground truth -- but it is one JSON
object per line with 10 KB strings inside, which nobody is going to read. This
writes the same content as:

    <out>/<run_id>/index.xlsx      one row per instance, sortable, with a
                                   failure category and short excerpts
    <out>/<run_id>/<id>.md         full detail for one instance: query,
                                   objectives, error, model output, code

The categories are the ones that turned out to matter when reading these by
hand -- "the code ran and gave the wrong number" is a modelling error, while
"IndexError" or "no code" is something else entirely, and lumping them together
hides where the remaining headroom is.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def text(value) -> str:
    """Empty cells reach us as NaN, not "" -- pandas fills missing values with
    float('nan'), which has no .strip(). Normalise once here rather than
    guarding at every use."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:        # NaN
        return ""
    return str(value)


def categorise(rec: dict, scored: dict | None) -> str:
    """One label per failure, chosen so the buckets suggest different actions."""
    err = text((scored or {}).get("solve_error")).strip()
    status = rec.get("status")

    if status == "failed":
        return "pipeline failed (no output at all)"
    if not text(rec.get("code_output")).strip():
        if "iteration limit" in text(rec.get("model_output")):
            return "agent hit the iteration / time limit"
        return "no code produced"
    if not err:
        return "code ran, objective does not match"
    head = err.split(":")[0].strip()
    if head in ("IndexError", "KeyError"):
        return f"{head} (usually inlined data shorter than the loop expects)"
    if head in ("SyntaxError",):
        return "SyntaxError (often a truncated literal)"
    if head in ("NameError",):
        return "NameError (used a variable it never defined)"
    if head.startswith("no gurobipy Model"):
        return "no model built -- see the code's own message in solve_error"
    return head or "other"


def load_scored(run_id: str) -> dict:
    """Per-instance scoring for this run, if score_runs.py has been run."""
    for p in (HERE / "scored_all.csv", HERE / "runs" / run_id / "scored.csv"):
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "run_id" in df.columns:
            df = df[df["run_id"].astype(str).str.contains(run_id, na=False)]
        if df.empty:
            continue
        return {str(r["instance_id"]): r.to_dict() for _, r in df.iterrows()}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="substring of the run_id to export")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "failures")
    ap.add_argument("--all", action="store_true",
                    help="export every instance, not only the failures")
    args = ap.parse_args()

    matches = sorted(d for d in (HERE / "runs").iterdir()
                     if d.is_dir() and args.run in d.name
                     and (d / "instances.jsonl").exists())
    if not matches:
        raise SystemExit(f"no run directory matches {args.run!r}")
    if len(matches) > 1:
        print("several runs match; exporting all of them:")
        for m in matches:
            print(f"  {m.name}")

    for run_dir in matches:
        run_id = run_dir.name
        scored = load_scored(run_id)
        out_dir = args.out / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Later record wins, same rule score_runs.py and aggregate_runs.py use.
        latest = {}
        for line in (run_dir / "instances.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                latest[str(d.get("instance_id"))] = d

        rows, written = [], 0
        for iid, rec in sorted(latest.items(),
                               key=lambda kv: int(kv[0]) if kv[0].isdigit()
                               else 1 << 30):
            sc = scored.get(iid)
            correct = None
            if sc is not None:
                try:
                    correct = float(sc.get("correct_optimal"))
                except (TypeError, ValueError):
                    correct = None
            if not args.all and correct == 1:
                continue

            query = text(rec.get("query"))
            model_output = text(rec.get("model_output"))
            code = text(rec.get("code_output"))
            err = text((sc or {}).get("solve_error"))
            cat = categorise(rec, sc)

            rows.append({
                "instance_id": iid,
                "verdict": {1.0: "correct", 0.0: "wrong"}.get(correct, "unscored"),
                "category": cat,
                "gold_type": rec.get("gold_type"),
                "pred_type": rec.get("pred_type"),
                "gold_objective": (sc or {}).get("gold_objective"),
                "model_objective": (sc or {}).get("objective"),
                "solve_error": err[:500],
                "n_vars": (sc or {}).get("n_vars"),
                "n_constrs": (sc or {}).get("n_constrs"),
                "query_excerpt": query[:400],
                "code_chars": len(code),
                "detail_file": f"{iid}.md",
            })

            md = [
                f"# Instance {iid} — {rows[-1]['verdict']}",
                "",
                f"- run: `{run_id}`",
                f"- category: **{cat}**",
                f"- ground-truth type: `{rec.get('gold_type')}`  "
                f"predicted: `{rec.get('pred_type')}`",
                f"- ground-truth objective: `{(sc or {}).get('gold_objective')}`  "
                f"model: `{(sc or {}).get('objective')}`",
            ]
            if err:
                md += ["", "## Error", "", "```", err, "```"]
            if rec.get("errors"):
                md += ["", "## Pipeline errors", "", "```",
                       json.dumps(rec["errors"], ensure_ascii=False, indent=2),
                       "```"]
            md += ["", "## Query", "", query or "_(empty)_"]
            md += ["", "## Model output", "",
                   model_output or "_(empty)_"]
            md += ["", "## Generated code", "", "```python",
                   code or "# (none)", "```", ""]
            (out_dir / f"{iid}.md").write_text("\n".join(md), encoding="utf-8")
            written += 1

        if not rows:
            print(f"{run_id}: nothing to export")
            continue

        df = pd.DataFrame(rows)
        xlsx = out_dir / "index.xlsx"
        try:
            df.to_excel(xlsx, index=False)
            where = xlsx
        except Exception as e:                            # noqa: BLE001
            df.to_csv(out_dir / "index.csv", index=False)
            where = out_dir / "index.csv"
            print(f"  (xlsx failed: {e}; wrote CSV instead)")

        print(f"\n{run_id}")
        print(f"  {written} instance(s) -> {out_dir}")
        print(f"  index -> {where}")
        print("\n  breakdown:")
        for cat, n in df["category"].value_counts().items():
            print(f"    {n:3d}  {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
