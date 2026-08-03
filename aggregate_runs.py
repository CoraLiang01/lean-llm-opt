#!/usr/bin/env python3
"""
aggregate_runs.py -- turn runs/*/instances.jsonl into the tables the reviewers
asked for.

    python aggregate_runs.py runs/ -o tables/
    python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv

Outputs
-------
tables/cost_table.csv / .tex     resource cost per (method, model, dataset)
                                 -> referee 2 Q3
tables/stage_breakdown.csv       tokens / calls / cost per pipeline stage
tables/classification.csv        classifier accuracy + confusion matrix
                                 -> referee 1 Q3
tables/per_instance.csv          flat table, one row per instance (for plots)

The optional --gold CSV supplies correctness labels; join key is
(dataset, instance_id). Recognised columns:
    correct_optimal      1/0  optimal value matches
    correct_formulation  1/0  manual exact-match verdict
    gold_type            str  ground-truth problem class
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RESOURCE_COLS = [
    "n_llm_calls", "n_tool_calls", "n_retriever_calls",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cached_prompt_tokens", "reasoning_tokens",
    "cost_usd", "llm_latency_s", "wall_s",
]


# --------------------------------------------------------------------------- #
def load_instances(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("instances.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            s = d.get("summary", {})
            row = {
                "run_id": d.get("run_id"),
                "method": d.get("method"),
                "model_profile": d.get("model_profile"),
                "dataset": d.get("dataset"),
                "repeat_index": d.get("repeat_index", 0),
                "instance_id": d.get("instance_id"),
                "status": d.get("status"),
                "gold_type": d.get("gold_type"),
                "pred_type": d.get("pred_type"),
                "n_retries": d.get("n_retries", 0),
                "n_llm_calls_failed": s.get("n_llm_calls_failed", 0),
                "n_estimated_token_calls": s.get("n_estimated_token_calls", 0),
                "objective_value": d.get("objective_value"),
            }
            for c in RESOURCE_COLS:
                row[c] = s.get(c) or 0
            for stage_name, sv in (s.get("by_stage") or {}).items():
                row[f"stage::{stage_name}::calls"] = sv.get("calls", 0)
                row[f"stage::{stage_name}::tool_calls"] = sv.get("tool_calls", 0)
                row[f"stage::{stage_name}::prompt_tokens"] = sv.get("prompt_tokens", 0)
                row[f"stage::{stage_name}::completion_tokens"] = sv.get("completion_tokens", 0)
                row[f"stage::{stage_name}::cost_usd"] = sv.get("cost_usd", 0.0)
                row[f"stage::{stage_name}::latency_s"] = sv.get("latency_s", 0.0)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no instances.jsonl found under {root}")
    return _dedupe(pd.DataFrame(rows))


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated records for the same instance within one run.

    Re-running a notebook's batch cell without starting a new run appends a
    second record for instances already in the log, so a run directory can
    hold several rows per instance. Averaging over the raw rows counts those
    instances more than once and quietly overstates the sample size -- on the
    first tables we produced, LEAN-LLM-OPT reported n_instances = 13 for what
    were 3 distinct problems.

    score_runs.py already collapses these (last record wins, since it is the
    most recent attempt); do the same here so the two tools agree, and say so
    when it happens.
    """
    if df.empty or "run_id" not in df or "instance_id" not in df:
        return df
    before = len(df)
    out = df.drop_duplicates(subset=["run_id", "instance_id"], keep="last")
    dropped = before - len(out)
    if dropped:
        print(f"[aggregate_runs] collapsed {dropped} duplicate record(s): "
              f"{before} rows -> {len(out)} (run, instance) pairs. "
              f"Re-running a batch cell into an existing run does this; the "
              f"last record for each instance is kept.", file=sys.stderr)
    return out.reset_index(drop=True)


#: Properties of the problem itself -- the same for every method that attempts
#: it, so they may be joined on (dataset, instance_id).
PER_INSTANCE_GOLD = ("gold_type", "gold_objective", "correct_formulation")

#: Properties of one attempt. Joining these on (dataset, instance_id) assigns
#: one run's result to every other run of the same instance.
PER_RUN_RESULT = ("correct_optimal", "objective", "n_vars", "n_constrs",
                  "n_nonzeros", "solver_status", "solve_error")


def attach_gold(df: pd.DataFrame, gold_path: Path | None) -> pd.DataFrame:
    """Join ground truth, keeping per-instance and per-run facts apart.

    `gold_labels.csv` mixes the two. `gold_type` and the human
    `correct_formulation` verdict belong to the instance. `correct_optimal`
    does not: it records whether *one particular run's* generated code hit the
    right objective. Joined on (dataset, instance_id) alone it lands on every
    method that touched that instance, so the last run scored silently
    overwrites everyone else's accuracy -- we measured Base-SingleCall at 3/3
    on Large-Scale-OR and then watched the table report 66.7% for it, which was
    LEAN-LLM-OPT's score on the same three instances.

    So per-run results come from `scored_all.csv`, which carries run_id, and
    are joined on (run_id, instance_id). Per-instance fields still come from
    the gold file.
    """
    for c in ("correct_optimal", "correct_formulation"):
        if c not in df.columns:
            df[c] = math.nan

    if gold_path is not None and Path(gold_path).exists():
        gold = pd.read_csv(gold_path)
        dropped = [c for c in PER_RUN_RESULT
                   if c in gold.columns and "run_id" not in gold.columns]
        if dropped:
            print(f"[aggregate_runs] {Path(gold_path).name} carries per-run "
                  f"column(s) {dropped} without a run_id; ignoring them and "
                  f"taking results from scored_all.csv instead.",
                  file=sys.stderr)
            gold = gold.drop(columns=dropped)
        keep = [c for c in gold.columns
                if c in ("dataset", "instance_id") + PER_INSTANCE_GOLD]
        gold = gold[keep]
        key = [c for c in ("dataset", "instance_id") if c in gold.columns]
        if key:
            df = df.merge(gold, on=key, how="left", suffixes=("", "_gold"))
            for c in PER_INSTANCE_GOLD:
                if f"{c}_gold" in df.columns:
                    df[c] = df[c].fillna(df[f"{c}_gold"]) \
                        if c in df.columns else df[f"{c}_gold"]

    scored = Path(gold_path).parent / "scored_all.csv" if gold_path \
        else Path("scored_all.csv")
    if scored.exists():
        s = pd.read_csv(scored)
        if {"run_id", "instance_id"} <= set(s.columns):
            cols = ["run_id", "instance_id"] + [c for c in PER_RUN_RESULT
                                                if c in s.columns]
            s = s[cols].drop_duplicates(subset=["run_id", "instance_id"],
                                        keep="last")
            df = df.merge(s, on=["run_id", "instance_id"], how="left",
                          suffixes=("", "_scored"))
            if "correct_optimal_scored" in df.columns:
                df["correct_optimal"] = df["correct_optimal_scored"]
            n = int(df["correct_optimal"].notna().sum())
            print(f"[aggregate_runs] per-run results joined from "
                  f"{scored.name} on (run_id, instance_id): "
                  f"{n}/{len(df)} rows scored")
        else:
            print(f"[aggregate_runs] {scored.name} has no run_id column; "
                  f"accuracy left unscored.", file=sys.stderr)
    else:
        print(f"[aggregate_runs] no scored_all.csv next to the gold file; "
              f"run score_runs.py first or accuracy stays NaN.",
              file=sys.stderr)
    return df


# --------------------------------------------------------------------------- #
def cost_table(df: pd.DataFrame, per_run: bool = False) -> pd.DataFrame:
    """The headline table: accuracy vs. resource cost, per method/model/dataset.

    With `per_run`, `run_id` joins the grouping keys. Merging every run of a
    method into one row is only meaningful when those runs are repeats of the
    same thing -- and they often are not. Our own first full table averaged a
    101-instance run together with four 3-instance smoke tests, two of which
    had been executed against an earlier version of the dataset, and reported
    the result as a single 74.07% over "113 instances". The per-run table makes
    that visible instead of dissolving it into an average.
    """
    keys = ["dataset", "method", "model_profile"]
    if per_run:
        keys = keys + ["run_id"]

    def agg(x: pd.DataFrame) -> pd.Series:
        n = len(x)
        out = {
            # n_instances counts executions (one per run x instance) -- that is
            # the right denominator for the means below. n_unique_instances is
            # how many distinct problems those executions covered, which is the
            # number a reader will read as the sample size. They differ
            # whenever a run is repeated, so report both rather than let the
            # larger one stand in for the smaller.
            "n_instances": n,
            "n_unique_instances": x["instance_id"].nunique(),
            "n_repeats": x["repeat_index"].nunique(),
            "acc_optimal_%": 100 * x["correct_optimal"].mean()
                             if x["correct_optimal"].notna().any() else math.nan,
            "acc_formulation_%": 100 * x["correct_formulation"].mean()
                                 if x["correct_formulation"].notna().any() else math.nan,
            "run_failure_%": 100 * (x["status"] != "ok").mean(),
            "llm_calls_mean": x["n_llm_calls"].mean(),
            "tool_calls_mean": x["n_tool_calls"].mean(),
            "prompt_tok_mean": x["prompt_tokens"].mean(),
            "completion_tok_mean": x["completion_tokens"].mean(),
            "total_tok_mean": x["total_tokens"].mean(),
            "cost_usd_mean": x["cost_usd"].mean(),
            "cost_usd_total": x["cost_usd"].sum(),
            "latency_s_mean": x["wall_s"].mean(),
            "latency_s_p95": x["wall_s"].quantile(0.95),
            "retries_mean": x["n_retries"].mean(),
            "est_tokens_%": 100 * (x["n_estimated_token_calls"] > 0).mean(),
        }
        # cost per additional correct instance vs. nothing (interpretability aid)
        if not math.isnan(out["acc_optimal_%"]) and out["acc_optimal_%"] > 0:
            out["usd_per_correct"] = x["cost_usd"].sum() / max(
                1e-9, x["correct_optimal"].sum())
        else:
            out["usd_per_correct"] = math.nan
        return pd.Series(out)

    # Built by iterating the groups rather than groupby.apply(): the
    # `include_groups` argument only exists in pandas >= 2.2, and this code has
    # to run on whatever pandas the notebook environment happens to have.
    rows = []
    for key, x in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(keys, key)), **agg(x).to_dict()})
    if not rows:
        return pd.DataFrame(columns=keys)
    return pd.DataFrame(rows).sort_values(keys).round(4)


def stage_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    stage_cols = [c for c in df.columns if c.startswith("stage::")]
    if not stage_cols:
        return pd.DataFrame()
    recs = []
    for (ds, m, mp), x in df.groupby(["dataset", "method", "model_profile"],
                                     dropna=False):
        stages = sorted({c.split("::")[1] for c in stage_cols})
        for s in stages:
            recs.append({
                "dataset": ds, "method": m, "model_profile": mp, "stage": s,
                "calls_mean": x.get(f"stage::{s}::calls", pd.Series([0])).mean(),
                "tool_calls_mean": x.get(f"stage::{s}::tool_calls", pd.Series([0])).mean(),
                "prompt_tok_mean": x.get(f"stage::{s}::prompt_tokens", pd.Series([0])).mean(),
                "completion_tok_mean": x.get(f"stage::{s}::completion_tokens", pd.Series([0])).mean(),
                "cost_usd_mean": x.get(f"stage::{s}::cost_usd", pd.Series([0])).mean(),
                "latency_s_mean": x.get(f"stage::{s}::latency_s", pd.Series([0])).mean(),
            })
    return pd.DataFrame(recs).round(4)


def classification_report(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    sub = df[df["gold_type"].notna() & df["pred_type"].notna()]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (ds, m, mp), x in sub.groupby(["dataset", "method", "model_profile"],
                                      dropna=False):
        hit = (x["gold_type"].str.strip().str.lower()
               == x["pred_type"].str.strip().str.lower())
        rec = {"dataset": ds, "method": m, "model_profile": mp,
               "n": len(x), "classification_acc_%": 100 * hit.mean()}
        # downstream accuracy conditional on classification correctness
        if x["correct_formulation"].notna().any():
            rec["form_acc_given_correct_cls_%"] = \
                100 * x.loc[hit, "correct_formulation"].mean() if hit.any() else math.nan
            rec["form_acc_given_wrong_cls_%"] = \
                100 * x.loc[~hit, "correct_formulation"].mean() if (~hit).any() else math.nan
        rows.append(rec)

        cm = pd.crosstab(x["gold_type"], x["pred_type"], dropna=False)
        cm.to_csv(out_dir / f"confusion__{ds}__{m}__{mp}.csv")
    return pd.DataFrame(rows).round(3)


def to_latex(t: pd.DataFrame, path: Path, caption: str, label: str):
    cols = ["method", "model_profile", "n_instances", "n_unique_instances",
            "acc_optimal_%",
            "acc_formulation_%", "llm_calls_mean", "tool_calls_mean",
            "prompt_tok_mean", "completion_tok_mean", "cost_usd_mean",
            "latency_s_mean"]
    cols = [c for c in cols if c in t.columns]
    sub = t[cols]

    def esc(x):
        if isinstance(x, float):
            return "--" if math.isnan(x) else f"{x:.2f}"
        return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    header = " & ".join(esc(c) for c in cols) + r" \\"
    lines = [" & ".join(esc(v) for v in row) + r" \\"
             for row in sub.itertuples(index=False, name=None)]
    body = ("\\begin{tabular}{" + "l" * len(cols) + "}\n\\toprule\n"
            + header + "\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n",
        encoding="utf-8")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("tables"))
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--run", default=None,
                    help="substring filter on run_id; use it to report a "
                         "single run instead of an average over several")
    ap.add_argument("--prune-empty", action="store_true",
                    help="delete run directories that hold no results "
                         "(left behind by re-running the config cell)")
    args = ap.parse_args()

    if args.prune_empty:
        import shutil
        removed = 0
        for d in sorted(args.runs_dir.iterdir()):
            if not d.is_dir():
                continue
            inst = d / "instances.jsonl"
            if not inst.exists() or inst.stat().st_size == 0:
                shutil.rmtree(d)
                removed += 1
        print(f"pruned {removed} empty run director"
              f"{'y' if removed == 1 else 'ies'}")

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_instances(args.runs_dir)
    if args.run:
        before = df["run_id"].nunique()
        df = df[df["run_id"].str.contains(args.run, na=False)]
        if df.empty:
            raise SystemExit(f"no run_id contains {args.run!r}")
        print(f"[aggregate_runs] --run {args.run!r}: "
              f"{df['run_id'].nunique()} of {before} run(s) kept")
    df = attach_gold(df, args.gold)
    df.to_csv(args.out / "per_instance.csv", index=False)

    ct = cost_table(df)
    ct.to_csv(args.out / "cost_table.csv", index=False)
    to_latex(ct, args.out / "cost_table.tex",
             "Accuracy and per-instance resource cost of LEAN-LLM-OPT, its "
             "ablations, and single-call baselines. Tokens and cost are "
             "averaged over instances; cost uses the provider price table "
             "recorded in the run configuration.",
             "tab:cost")

    # Always emit the per-run breakdown alongside the merged table: the merged
    # one hides whether its rows came from one run or from several that are not
    # comparable. Cheap to produce, and the first thing to check.
    ct_run = cost_table(df, per_run=True)
    ct_run.to_csv(args.out / "cost_table_per_run.csv", index=False)

    sb = stage_breakdown(df)
    if not sb.empty:
        sb.to_csv(args.out / "stage_breakdown.csv", index=False)

    cr = classification_report(df, args.out)
    if not cr.empty:
        cr.to_csv(args.out / "classification.csv", index=False)

    print(ct.to_string(index=False))

    if len(ct_run) > len(ct):
        cols = [c for c in ("run_id", "n_instances", "n_unique_instances",
                            "acc_optimal_%", "cost_usd_total", "latency_s_mean")
                if c in ct_run.columns]
        print("\nper run (the merged table above averages these together):")
        show = ct_run[cols].copy()
        show["run_id"] = show["run_id"].str.slice(-34)
        print(show.to_string(index=False))

    print(f"\nwritten to {args.out.resolve()}")


if __name__ == "__main__":
    main()
