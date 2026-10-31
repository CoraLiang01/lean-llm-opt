#!/usr/bin/env python3
"""
verify_report.py
================
Read one scored run and say whether the 2026-08-22 config revert worked.

    python verify/verify_report.py runs/<run_id>          # or any scored.csv

The revert put three things back to what the original notebooks did:

  1. embedding model   text-embedding-3-small  ->  text-embedding-ada-002
  2. classifier LLM    gpt-4.1-2025-04-14      ->  gpt-4
  3. load_refdata_docs() column fallback, two columns -> every column

The baseline it is measured against is the March run recorded in
Results/Large-scale-or-Results-0315-ALL.xlsx (LEAN GPT-4.1, 85.1% on the
optimal-value metric). That sheet's own grading was checked against
score_runs.py's 1e-4 relative criterion and agrees on 100 of 101 instances, so
the two numbers are comparable.

Eight instances -- 41, 50, 52, 58, 61, 70, 78, 99 -- were correct in March and
wrong in all seven post-packaging runs. They are the regression; everything
else moves around between runs. This script reports them separately from the
headline accuracy, because with ~40 of 101 instances flipping run to run, the
headline alone cannot tell a fix from a lucky draw.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent

#: Correct in March, wrong in all seven runs made after the packaging change.
REGRESSION_IDS = [41, 50, 52, 58, 61, 70, 78, 99]

BASELINE_XLSX = HERE / "Results" / "Large-scale-or-Results-0315-ALL.xlsx"
BASELINE_SHEET = "12月-CW-101汇总"
#: Column offsets in that sheet: ID is 1-based, the LEAN GPT-4.1 block starts
#: at 9 (model correctness) / 10 (objective correctness).
COL_ID, COL_TYPE, COL_MODEL_OK, COL_OBJ_OK = 0, 4, 9, 10


def load_baseline() -> pd.DataFrame | None:
    """March per-instance grading, 0-indexed to match the run logs."""
    if not BASELINE_XLSX.exists():
        return None
    try:
        import openpyxl
    except ImportError:
        print("! openpyxl not installed; skipping the March comparison\n")
        return None
    ws = openpyxl.load_workbook(BASELINE_XLSX, read_only=True,
                                data_only=True)[BASELINE_SHEET]
    rows = [r for r in list(ws.iter_rows(values_only=True))[2:]
            if r[COL_ID] is not None]
    return pd.DataFrame({
        # the sheet numbers instances from 1, the run logs from 0
        "instance_id": [int(r[COL_ID]) - 1 for r in rows],
        "gold_type": [str(r[COL_TYPE]) for r in rows],
        "march_ok": [int(r[COL_OBJ_OK]) if isinstance(r[COL_OBJ_OK], (int, float))
                     else None for r in rows],
    }).set_index("instance_id")


def load_scored(target: Path) -> pd.DataFrame:
    csv = target / "scored.csv" if target.is_dir() else target
    if not csv.exists():
        raise SystemExit(f"no scored.csv at {csv} -- run score_runs.py first")
    df = pd.read_csv(csv)
    df["instance_id"] = pd.to_numeric(df["instance_id"], errors="coerce")
    df = df.dropna(subset=["instance_id"])
    df["instance_id"] = df["instance_id"].astype(int)
    df["ok"] = (df["correct_optimal"].fillna(0) == 1).astype(int)
    return df.set_index("instance_id").sort_index()


def routed_to_specialist(gold: str, pred: str) -> bool:
    """Did the classifier send this instance to the agent that fits it?

    UFLP and FLP are the same specialist (`get_FLP_response`), so counting a
    UFLP instance routed to "Facility Location Problem" as a miss would report
    a routing failure that never happened. Mixture has no specialist at all --
    it falls through to the Others-with-CSV handler by design -- so it is left
    out of the rate rather than counted as a permanent miss.
    """
    g, p = str(gold).strip().lower(), str(pred).strip().lower()
    fam = {
        "nrm": "nrm", "network revenue management": "nrm",
        "network revenue management problem": "nrm",
        "ra": "ra", "resource allocation": "ra",
        "resource allocation problem": "ra",
        "tp": "tp", "transportation": "tp", "transportation problem": "tp",
        "ap": "ap", "assignment problem": "ap",
        "flp": "flp", "uflp": "flp", "facility location problem": "flp",
        "uncapacited facility location problem": "flp",
    }
    return fam.get(g) is not None and fam.get(g) == fam.get(p)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[3].strip())
    df = load_scored(Path(sys.argv[1]))
    base = load_baseline()

    n, ok = len(df), int(df["ok"].sum())
    run_id = str(df["run_id"].iloc[0]) if "run_id" in df else "?"

    print("=" * 74)
    print(f"run      : {run_id}")
    print(f"instances: {n}")
    print(f"accuracy : {ok}/{n} = {ok / n:.1%}   (optimal-value, tol 1e-4)")
    print("=" * 74)

    # ---- the eight deterministic regressions -------------------------------
    present = [i for i in REGRESSION_IDS if i in df.index]
    if present:
        print("\nThe 8 regression instances (right in March, wrong in all 7 "
              "post-packaging runs)")
        print("-" * 74)
        print(f"{'id':>4}  {'type':<10}  {'gold':>16}  {'produced':>18}  result")
        recovered = 0
        for i in present:
            r = df.loc[i]
            good = int(r["ok"]) == 1
            recovered += good
            produced = ("-" if pd.isna(r["objective"])
                        else f"{float(r['objective']):.6g}")
            note = "" if good or pd.isna(r.get("solve_error")) else \
                   f"  [{str(r['solve_error'])[:34]}]"
            print(f"{i:>4}  {str(r['gold_type'])[:10]:<10}  "
                  f"{float(r['gold_objective']):>16.6g}  {produced:>18}  "
                  f"{'RECOVERED' if good else 'still wrong'}{note}")
        print("-" * 74)
        print(f"recovered: {recovered}/{len(present)}")
        if recovered == len(present):
            print("=> the revert fixed every deterministic regression.")
        elif recovered == 0:
            print("=> nothing recovered. The three reverted settings are not "
                  "what broke these.")
        else:
            print(f"=> partial. {len(present) - recovered} instance(s) have a "
                  f"cause other than the three reverted settings.")

    # ---- per-type, against March ------------------------------------------
    if base is not None:
        j = df.join(base[["march_ok"]], how="left")
        print("\nAccuracy by problem type")
        print("-" * 74)
        print(f"{'type':<38} {'n':>4}  {'March':>7}  {'now':>7}  {'delta':>7}")
        for t, g in j.groupby(j["gold_type"].astype(str)):
            now = g["ok"].mean()
            mar = g["march_ok"].mean()
            d = "" if pd.isna(mar) else f"{now - mar:+.2f}"
            mar_s = "-" if pd.isna(mar) else f"{mar:.2f}"
            print(f"{t[:38]:<38} {len(g):>4}  {mar_s:>7}  {now:>7.2f}  {d:>7}")
        mar_all = j["march_ok"].mean()
        if not pd.isna(mar_all):
            print("-" * 74)
            print(f"{'ALL':<38} {len(j):>4}  {mar_all:>7.2f}  "
                  f"{j['ok'].mean():>7.2f}  {j['ok'].mean() - mar_all:>+7.2f}")

        flips = j[j["march_ok"].notna() & (j["march_ok"] != j["ok"])]
        newly_broken = sorted(flips[flips["march_ok"] == 1].index.tolist())
        newly_fixed = sorted(flips[flips["march_ok"] == 0].index.tolist())
        print(f"\nwrong now but right in March : {len(newly_broken)} "
              f"{newly_broken}")
        print(f"right now but wrong in March : {len(newly_fixed)} "
              f"{newly_fixed}")

    # ---- classifier routing ------------------------------------------------
    if "pred_type" in df:
        r = df[~df["gold_type"].astype(str).str.lower().str.startswith(
            ("mixture", "others"))]
        if len(r):
            hit = sum(routed_to_specialist(a, b)
                      for a, b in zip(r["gold_type"], r["pred_type"]))
            print(f"\nclassifier routing (specialist types only): "
                  f"{hit}/{len(r)} = {hit / len(r):.1%}")
            nrm = r[r["gold_type"].astype(str).str.upper().str.startswith("NRM")]
            if len(nrm):
                nh = sum(routed_to_specialist(a, b)
                         for a, b in zip(nrm["gold_type"], nrm["pred_type"]))
                # The 3/25 this used to cite was a property of the
                # text-embedding-3-small run, not of the classifier LLM.
                # Restoring ada-002 took it to 22/25 with the same gpt-4.1
                # classifier, so attributing it to the model was wrong.
                print(f"  of which NRM -> NRM agent: {nh}/{len(nrm)}  "
                      f"(3/25 with text-embedding-3-small; "
                      f"22-24/25 once ada-002 was restored)")

    # ---- things that are not modelling failures ----------------------------
    errs = df[df["solve_error"].notna()]
    if len(errs):
        print(f"\n{len(errs)} instance(s) failed to execute at all "
              f"(not a formulation result):")
        for i, r in errs.iterrows():
            print(f"  {i:>4}  {str(r['solve_error'])[:64]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
