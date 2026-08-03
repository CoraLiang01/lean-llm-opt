#!/usr/bin/env python3
"""
score_runs.py
=============
Execute the generated gurobipy code of every logged instance, compare the
objective value against the ground truth, and write the gold_labels.csv that
aggregate_runs.py needs to fill in its accuracy columns.

    python score_runs.py runs/                       # score everything
    python score_runs.py runs/ --run <run_id>        # one run only
    python score_runs.py runs/ -o gold_labels.csv

Output
------
gold_labels.csv           dataset, instance_id, gold_type, correct_optimal, ...
runs/<run_id>/scored.csv  per-instance detail for that run

What "correct_optimal" means
----------------------------
The generated code runs, produces an objective value, and that value matches
`Label-objective` within --tol relative error. This is the optimal-value
matching metric the paper already uses -- and, as Referee 1 points out, it
overstates correctness, because a structurally wrong model can still hit the
right number. `correct_formulation` is therefore left empty here: it needs
human judgement, and this script deliberately does not guess it.

Each solve runs in a separate process with a wall-clock limit, so an
infinite loop or a segfault in generated code cannot take the scorer down.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

CODE_FENCE = re.compile(r"^```(?:python)?|```$", re.IGNORECASE | re.M)


def clean_code(code: str) -> str:
    return CODE_FENCE.sub("", str(code)).strip()


# --------------------------------------------------------------------------- #
#: "Optimal objective 1.234e+05" as Gurobi prints it. One row of the 101-set
#: has a whole solver transcript pasted into the label cell instead of a number.
_GUROBI_OBJ_RE = re.compile(
    r"optimal\s+objective\s*[:=]?\s*"
    r"([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)", re.I)

#: Thousands separators, currency marks and stray whitespace. Not a general
#: number parser -- anything outside this set is left to fail loudly.
_NUMERIC_NOISE_RE = re.compile(r"[,\s$¥€£]")


def parse_gold(value) -> float | None:
    """Read a ground-truth objective that was entered by hand.

    `Label-objective` is human-maintained, so it holds things `float()` cannot
    take: `'11,258,129.67'`, `'$24,877.39 '`, and one cell containing an entire
    Gurobi transcript. A bare `float()` turned all five into "no ground truth",
    and an instance with no ground truth is dropped from the accuracy
    denominator -- so a formatting quirk silently shrank the sample instead of
    being reported.

    Recovery is deliberately narrow: strip separators and currency marks, and
    recognise Gurobi's own "Optimal objective ..." line. Anything else returns
    None rather than guessing, because inventing a number here would turn a
    visible gap into a wrong accuracy figure.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if v != v else v            # NaN -> None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    cleaned = _NUMERIC_NOISE_RE.sub("", s)
    try:
        return float(cleaned)
    except ValueError:
        pass
    m = _GUROBI_OBJ_RE.search(s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def data_dir_for(rec: dict) -> Path | None:
    """Directory the generated code should run in, from `dataset_address`.

    Methods differ in how their code reaches the data, and that difference was
    silently penalising one of them. The full pipeline retrieves values with a
    data agent and emits them inline (`r = [50, 500, ...]`) or reads through a
    `data_path` variable, so its code runs anywhere. The single-call baseline
    gets the same data pasted into its prompt but writes
    `pd.read_csv('products.csv')` -- a bare filename, because nothing ever told
    it a working directory. Executing that from the repo root raises
    FileNotFoundError, which then scored as a wrong answer.

    That penalty is one-directional: it makes the baseline look worse for a
    reason unrelated to its formulation -- exactly the baseline-fairness
    problem this instrumentation is meant to remove (referee comment 3.2). So
    resolve the instance's own data directory and run there.

    `dataset_address` may list several files, one per line; they share a
    directory, so the first one decides.
    """
    addr = (rec.get("dataset_address") or "").strip()
    if not addr:
        return None
    first = addr.splitlines()[0].strip()
    if not first:
        return None
    p = (HERE / first).resolve()
    d = p.parent if p.suffix else p
    return d if d.is_dir() else None


def _solve_worker(code: str, time_limit: float, q, workdir: str | None = None):
    """Runs in a child process; never trust generated code in-process."""
    import io
    import contextlib
    import os
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:                            # noqa: BLE001
        q.put({"ok": False, "error": f"gurobipy import failed: {e}"})
        return

    # Child process: this chdir is local to it and cannot affect the scorer.
    if workdir:
        try:
            os.chdir(workdir)
        except Exception:                             # noqa: BLE001
            pass                                      # keep the inherited cwd

    env = {"__builtins__": __builtins__, "gp": gp, "GRB": GRB}
    try:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf), \
                contextlib.redirect_stderr(buf):
            exec(code, env)                            # noqa: S102
            model = None
            for v in env.values():
                if isinstance(v, gp.Model):
                    model = v
            if model is None:
                # The generated code is written from a few-shot template that
                # wraps everything in `try: ... except Exception as e: print(e,
                # file=sys.stderr)`. So a failure does not propagate: the code
                # "succeeds", `m` is never bound, and the only symptom left is
                # a missing Model. Reporting just that discards the one thing
                # that explains it -- the message the code printed on its way
                # down, which this buffer is holding right now.
                tail = buf.getvalue().strip().replace("\n", " | ")[-300:]
                q.put({"ok": False,
                       "error": "no gurobipy Model in namespace"
                                + (f" -- code output: {tail}" if tail else
                                   " (and it printed nothing)")})
                return
            status = model.Status
            res = {
                "ok": True,
                "status": int(status),
                "objective": float(model.ObjVal) if status == GRB.OPTIMAL else None,
                "n_vars": int(model.NumVars),
                "n_constrs": int(model.NumConstrs),
                "n_nonzeros": int(model.NumNZs),
                "is_mip": bool(model.IsMIP),
                "runtime_s": float(model.Runtime),
            }
        q.put(res)
    except Exception as e:                             # noqa: BLE001
        q.put({"ok": False, "error": f"{type(e).__name__}: {e}"})


def solve_anywhere(code: str, time_limit: float, data_dir: Path | None) -> dict:
    """Execute the generated code, trying both plausible working directories.

    Nothing in any prompt tells the model what the working directory will be,
    so the generated code splits into two styles that need opposite setups:

      repo root      `pd.read_csv('Test_Dataset/.../DistanceMatrix.csv')`
      data directory `pd.read_csv('DistanceMatrix.csv')`

    Picking either one alone scores the other style as a wrong answer. That is
    not a modelling failure, and it is not evenly distributed: the full
    pipeline has a data agent and tends to emit the first style, the
    single-call baseline has only the pasted-in data and tends to emit the
    second. Committing to one directory therefore hands one method a
    systematic advantage -- the baseline-fairness issue behind referee comment
    3.2.

    So try the repo root first (the historical behaviour, keeps old numbers
    reproducible), and on a file-not-found retry from the instance's own data
    directory. Every method gets both attempts, so no style is penalised.
    `workdir_used` records which one produced the result.
    """
    r = solve(code, time_limit, workdir=HERE)
    if r.get("ok") or data_dir is None:
        r.setdefault("workdir_used", "repo_root")
        return r
    err = str(r.get("error") or "")
    if "FileNotFoundError" not in err and "No such file" not in err:
        r.setdefault("workdir_used", "repo_root")
        return r                      # a real failure; a retry proves nothing
    r2 = solve(code, time_limit, workdir=data_dir)
    if r2.get("ok"):
        r2["workdir_used"] = "data_dir"
        return r2
    r["workdir_used"] = "repo_root+data_dir both failed"
    return r


def solve(code: str, time_limit: float = 120.0, workdir=None) -> dict:
    if not code or not code.strip():
        return {"ok": False, "error": "no code"}
    q = mp.Queue()
    p = mp.Process(target=_solve_worker,
                   args=(code, time_limit, q, str(workdir) if workdir else None))
    p.start()
    p.join(time_limit)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"ok": False, "error": f"timeout after {time_limit}s"}
    try:
        return q.get_nowait()
    except Exception:
        return {"ok": False, "error": "worker produced no result (crash?)"}


# --------------------------------------------------------------------------- #
def score_run(run_dir: Path, tol: float, time_limit: float) -> pd.DataFrame:
    f = run_dir / "instances.jsonl"
    if not f.exists():
        return pd.DataFrame()

    rows = []
    lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    # A resumed run legitimately repeats an instance_id, and the later record
    # wins. But an id can also collide because two different batches were run
    # through the same logger after reset_index(drop=True) -- in that case the
    # earlier results are silently lost, which must not pass unnoticed.
    latest, seen = {}, {}
    # Ground-truth values that are present but unreadable; reported at the end
    # so an instance never leaves the accuracy denominator silently.
    unparsed: list[tuple] = []
    for line in lines:
        d = json.loads(line)
        iid = d.get("instance_id")
        seen[iid] = seen.get(iid, 0) + 1
        latest[iid] = d
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        print(f"  ! {run_dir.name}: {len(lines)} records but "
              f"{len(latest)} unique instance ids; only the last record of "
              f"{sorted(dupes)} is scored. If these were different batches, "
              f"do not use reset_index(drop=True) when slicing the test set.")

    for iid, d in sorted(latest.items(), key=lambda kv: (kv[0] is None, kv[0])):
        code = clean_code(d.get("code_output") or "")
        gold = d.get("gold_objective")
        gold_val = parse_gold(gold)
        if gold_val is None and gold not in (None, ""):
            # Present but unreadable. Say so: this instance is about to drop
            # out of the accuracy denominator, and that should never be quiet.
            unparsed.append((iid, str(gold)[:60].replace("\n", " ")))

        # Generated code addresses its data in one of two incompatible ways,
        # and neither is wrong -- nothing in the prompt ever fixes a working
        # directory. See solve_anywhere().
        r = (solve_anywhere(code, time_limit, data_dir_for(d)) if code
             else {"ok": False, "error": "no code"})
        obj = r.get("objective")
        correct = None
        if gold_val is not None and obj is not None:
            correct = int(abs(obj - gold_val) <= tol * max(1.0, abs(gold_val)))
        elif gold_val is not None:
            correct = 0                       # code failed => not correct

        rows.append({
            "dataset": d.get("dataset"),
            "instance_id": iid,
            "run_id": d.get("run_id"),
            "method": d.get("method"),
            "model_profile": d.get("model_profile"),
            "gold_type": d.get("gold_type"),
            "pred_type": d.get("pred_type"),
            "size_class": d.get("size_class"),
            "status": d.get("status"),
            "gold_objective": gold_val,
            "objective": obj,
            "correct_optimal": correct,
            "correct_formulation": "",        # human judgement, left blank
            "solver_status": r.get("status"),
            "n_vars": r.get("n_vars"),
            "n_constrs": r.get("n_constrs"),
            "n_nonzeros": r.get("n_nonzeros"),
            "is_mip": r.get("is_mip"),
            "solve_runtime_s": r.get("runtime_s"),
            "solve_error": r.get("error"),
            # which working directory the code turned out to need -- keep it
            # visible so the retry in solve_anywhere() stays auditable
            "workdir_used": r.get("workdir_used"),
        })

    if unparsed:
        print(f"  ! {run_dir.name}: {len(unparsed)} ground-truth value(s) could "
              f"not be read as a number, so those instances are excluded from "
              f"the accuracy denominator:")
        for iid, raw in unparsed[:5]:
            print(f"      [{iid}] {raw!r}")
        if len(unparsed) > 5:
            print(f"      ... and {len(unparsed) - 5} more")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(run_dir / "scored.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("gold_labels.csv"))
    ap.add_argument("--run", default=None, help="substring filter on run_id")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="relative tolerance for the objective match")
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="seconds per instance before the solve is killed")
    args = ap.parse_args()

    dirs = [d for d in sorted(args.runs_dir.iterdir())
            if d.is_dir() and (not args.run or args.run in d.name)]
    if not dirs:
        raise SystemExit(f"no run directories under {args.runs_dir}")

    all_df = []
    for d in dirs:
        df = score_run(d, args.tol, args.time_limit)
        if df.empty:
            continue
        all_df.append(df)
        ok = df["correct_optimal"].fillna(0).sum()
        n = len(df)
        err = df["solve_error"].notna().sum()
        print(f"{d.name[:66]:<68} {int(ok):>3}/{n:<3} correct  "
              f"({err} solve errors)")

    if not all_df:
        raise SystemExit("nothing scored")

    combined = pd.concat(all_df, ignore_index=True)

    # gold_labels.csv is keyed on (dataset, instance_id): the ground truth is a
    # property of the instance, not of the run that produced it.
    gold = (combined[combined["gold_objective"].notna()]
            .drop_duplicates(subset=["dataset", "instance_id", "method",
                                     "model_profile"])
            [["dataset", "instance_id", "gold_type", "correct_optimal",
              "correct_formulation"]])
    gold = gold.drop_duplicates(subset=["dataset", "instance_id"], keep="last")
    gold.to_csv(args.out, index=False)

    print(f"\n{len(combined)} instances scored across {len(all_df)} runs")
    print(f"gold labels -> {args.out}")

    # Written unconditionally. This used to sit inside the `if not sc.empty`
    # below, so a run in which nothing solved left the previous run's file in
    # place -- and the stale contents then read as if they belonged to the run
    # just scored. A run where everything failed is exactly when the detail is
    # most worth having.
    scored_all = args.out.with_name("scored_all.csv")
    combined.to_csv(scored_all, index=False)
    print(f"per-instance detail -> {scored_all}")

    # scale statistics -- Referee 2 Q2 asks for exactly this
    sc = combined[combined["n_vars"].notna()]
    if not sc.empty:
        print("\nmodel size of the instances that solved:")
        for col in ("n_vars", "n_constrs", "n_nonzeros"):
            s = sc[col]
            print(f"  {col:<12} min {s.min():>8.0f}  median {s.median():>8.0f}  "
                  f"max {s.max():>8.0f}")
    else:
        print("\nno instance produced a solvable model in this selection.")

    print("\nnext: python aggregate_runs.py runs/ -o tables/ "
          f"--gold {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
