#!/usr/bin/env python3
"""
run_original.py
===============
Run the pre-instrumentation notebook and score it with the current scorer.

    python verify/run_original.py --list             # plan only, spends nothing
    python verify/run_original.py                    # all 101, gpt-4.1 classifier
    python verify/run_original.py --classifier gpt-4 # verbatim, ~$35
    python verify/run_original.py --rows 49,70,72

This is the control the incremental config work never was. Instead of
restoring one setting at a time and re-measuring, it takes the notebook as it
stood at `a135715^` -- the commit before the 2026-08-01 instrumentation, and so
the code that produced the March numbers -- and runs it against the same data,
with the same scorer, as `verify/run_full.sh`. Whatever the difference between
then and now is, this measures all of it at once.

What is held fixed, and what is not
-----------------------------------
Only the notebook comes from the old commit. Test_Dataset, Large_Scale_Or_Files
and the scorer are the current ones, so the comparison against Run A
(`runs/...185057__4d5d26b5e733`, 73.3%) has the notebook as its single variable.

One substitution is unavoidable. At `a135715^`, Large_Scale_Or_Files/RefData.csv
is not a CSV -- it is an xlsx file carrying a .csv name (it starts `PK\\x03\\x04`),
and cell 4 hands it to CSVLoader, which raises. That commit does not run as
committed. The current RefData.csv is a real CSV of the same 96 rows, differing
in exactly one cell (`Sales-Based Linear Programming - CA` lost a space), so
this uses it and notes the substitution in the run log. leanopt_exp.load_refdata()
carries an xlsx-tolerance branch for the same reason; the instrumentation fixed
this file, it did not break it.

The classifier model
--------------------
The notebook hard-codes `model_name="gpt-4"` for the classification agent --
one occurrence, in cell 4; the other eight `model_name` settings are gpt-4.1
and belong to the modeller, data agent and coder. `--classifier` rewrites that
one, and defaults to gpt-4.1.

Defaulting to a substitution in a script called run_original is a deliberate
trade and the header says so on every run. gpt-4.1 is what Run A used, so it
keeps the notebook as the single variable against the number this is being
compared to, and it takes the pass from ~$35 to ~$10 (gpt-4 is $30/$60 per 1M
against $2/$8). `--classifier gpt-4` runs the cell verbatim; the choice is
recorded in config.json either way, and a run that asks for the substitution
and finds nothing to substitute fails rather than continuing.

Wall clock
----------
`time.sleep(15)` between instances adds about 25 minutes over the set;
`--no-sleep` removes it.

Nothing here enforces a budget: the old code has no budget guard, and this
deliberately does not add one.

The lost-instance bug
---------------------
The old run_test catches `requests.exceptions.RequestException` and `continue`s
without appending, so one network error leaves output_code shorter than the
test frame and every later row silently paired with the wrong query. That is
the bug the instrumented loop was written to fix. This refuses to score a run
where the lengths disagree rather than reporting a misaligned number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

#: The repo that still has the history. `lean-llm-opt-main` is a ZIP download.
GIT_REPO = Path.home() / "Desktop/lean-original-4check"
DEFAULT_COMMIT = "a135715^"
NOTEBOOK_PATH = "LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
DATA = HERE / "Test_Dataset/Large-scale-or/Large-scale-or-101.csv"
OUT_ROOT = HERE / "verify/out/original_runs"


def notebook_at(commit: str) -> dict:
    r = subprocess.run(["git", "-C", str(GIT_REPO), "show",
                        f"{commit}:{NOTEBOOK_PATH}"],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot read {NOTEBOOK_PATH} at {commit} from "
                         f"{GIT_REPO}:\n{r.stderr.decode()[:400]}")
    return json.loads(r.stdout.decode("utf-8"))


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
    raise SystemExit("no cell calls run_test()")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--rows", help="comma-separated row labels; default all 101")
    ap.add_argument("--data-commit", metavar="COMMIT",
                    help="take Large-scale-or-101.csv from this commit instead "
                         "of the working tree. 344804b is the 2026-03-16 "
                         "version, the last before the six August edits.")
    ap.add_argument("--classifier", choices=("gpt-4", "gpt-4.1"),
                    default="gpt-4.1",
                    help="model for the classification agent. The notebook "
                         "hard-codes gpt-4; the default here substitutes "
                         "gpt-4.1, which is the one thing Run A also uses and "
                         "cuts the pass from ~$35 to ~$10. Pass gpt-4 to run "
                         "the cell verbatim.")
    ap.add_argument("--no-sleep", action="store_true",
                    help="drop run_test's time.sleep(15); saves ~25 min, no "
                         "longer verbatim")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue into an existing run directory under "
                         "verify/out/original_runs/, skipping the instances "
                         "already logged there")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    nb = notebook_at(args.commit)
    cells = setup_cells(nb)

    # The test set is a variable in its own right: Large-scale-or-101.csv was
    # edited six times in August, changing the Query text of 10 instances and
    # the Label-objective of 3. The 85% and the external 82% predate all of
    # that, so comparing against them means holding the data still too.
    data_path, data_note = DATA, "working tree"
    if args.data_commit:
        rel = DATA.relative_to(HERE)
        r = subprocess.run(["git", "-C", str(GIT_REPO), "show",
                            f"{args.data_commit}:{rel}"], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"cannot read {rel} at {args.data_commit}:\n"
                             f"{r.stderr.decode()[:300]}")
        data_path = Path(tempfile.mkdtemp(prefix="leanopt-data-")) / DATA.name
        data_path.write_bytes(r.stdout)
        dsha = subprocess.run(["git", "-C", str(GIT_REPO), "rev-parse",
                               args.data_commit], capture_output=True,
                              text=True).stdout.strip()[:8]
        data_note = f"{args.data_commit} ({dsha})"
    df = pd.read_csv(data_path)
    if args.rows:
        df = df.loc[[int(r) for r in args.rows.split(",") if r.strip()]]

    sha = subprocess.run(["git", "-C", str(GIT_REPO), "rev-parse", args.commit],
                         capture_output=True, text=True).stdout.strip()[:12]
    tag = f"__data-{args.data_commit}" if args.data_commit else ""
    run_id = args.resume or (f"ORIGINAL__{sha}{tag}__"
                             f"{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir = OUT_ROOT / run_id
    if args.resume and not out_dir.exists():
        raise SystemExit(f"--resume: no such run directory: {out_dir}")

    print(f"notebook  : {NOTEBOOK_PATH} @ {args.commit} ({sha})")
    print(f"setup     : {len(cells)} cell(s), up to cell {cells[-1][0]}")
    print(f"instances : {len(df)}")
    print(f"test set  : {data_note}")
    if args.classifier == "gpt-4":
        print("classifier: gpt-4, verbatim            ~$0.27/instance, ~$35 total")
    else:
        print(f"classifier: {args.classifier}  <-- SUBSTITUTED, the notebook "
              f"hard-codes gpt-4   ~$10 total")
    print(f"sleep     : {'removed' if args.no_sleep else '15s between instances'}")
    print(f"out       : {out_dir}")
    if args.list:
        print("\n--list: nothing executed, nothing spent.")
        return 0

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if len(key) < 20:
        import getpass
        key = getpass.getpass("OPENAI_API_KEY (input hidden): ").strip()
    if len(key) < 20:
        raise SystemExit("no API key")

    os.chdir(HERE)                      # notebook uses repo-relative paths

    ns: dict = {"__name__": "__main__"}
    subs: list = []
    for idx, src in cells:
        if src.strip().startswith(("!", "%")):
            continue
        if args.no_sleep:
            src = re.sub(r"time\.sleep\(\s*15\s*\)", "pass", src)
        if args.classifier != "gpt-4":
            # `model_name="gpt-4"` occurs exactly once in this notebook, in
            # cell 4's classification agent. The eight `gpt-4.1` occurrences
            # are the modeller, data agent and coder; the \b and the closing
            # quote keep the pattern off them.
            src, n_sub = re.subn(r'(model_name\s*=\s*["\'])gpt-4(["\'])',
                                 lambda m: m.group(1) + args.classifier + m.group(2),
                                 src)
            if n_sub:
                subs.append((idx, n_sub))
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)      # noqa: S102
        except Exception as e:                                   # noqa: BLE001
            print(f"\n!! setup cell {idx} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise
        # cell 1 assigns the placeholder; every ChatOpenAI/OpenAIEmbeddings
        # built in later cells reads this name, so it has to be real before
        # cell 4 runs.
        if "user_api_key" in src:
            ns["user_api_key"] = key

    run_test = ns.get("run_test")
    agent = ns.get("classification_agent")
    if run_test is None or agent is None:
        raise SystemExit("setup finished but run_test/classification_agent "
                         "are not defined")

    # One row per run_test() call, and the record flushed before the next one
    # starts. run_test builds its three lists locally and returns them, doing
    # nothing else outside the loop, so a row at a time is the same pipeline --
    # but it means an hour of work survives a crash on instance 90 instead of
    # vanishing, and a resumed run can skip what is already on disk.
    #
    # This is not a hypothetical. The first attempt died an hour in: the laptop
    # changed networks, the openai SDK's pooled sockets were still bound to the
    # old local address, and every read blocked on a connection that could not
    # come back. Nothing had been written.
    #
    # It also removes the misalignment risk outright. The old run_test catches
    # RequestException and continues *without appending*, so over a whole frame
    # one network error shifts every later row onto the wrong query. Called one
    # row at a time, that shows up as an empty return for that row and is
    # recorded as a failure, not smeared across the rest.
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "instances.jsonl"
    gold_col = "Label-objective"
    type_col = "Problem Type" if "Problem Type" in df.columns else "Type"

    done = set()
    if jsonl.exists():
        done = {json.loads(l)["instance_id"]
                for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()}
        if done:
            print(f"resume: {len(done)} instance(s) already logged, skipping")

    print(f"\nrunning {len(df) - len(done)} instance(s) through the original "
          f"run_test(), one at a time\n")
    t0 = time.perf_counter()
    n_fail = 0
    with open(jsonl, "a", encoding="utf-8") as fh:
        for n, (i, row) in enumerate(df.iterrows(), 1):
            if int(i) in done:
                continue
            t1 = time.perf_counter()
            status, err = "ok", None
            try:
                om, oc, cl = run_test(df.loc[[i]], agent)
            except Exception as e:                            # noqa: BLE001
                om, oc, cl = [], [], []
                status, err = "failed", f"{type(e).__name__}: {e}"
                n_fail += 1
            if not om or not oc:
                # run_test swallowed the instance; record it rather than let it
                # shift the rest of the frame.
                status = "failed"
                err = err or "run_test returned nothing for this instance"
                om, oc, cl = [None], [None], [None]
                if status == "failed" and err.startswith("run_test returned"):
                    n_fail += 1
            fh.write(json.dumps({
                "run_id": run_id, "instance_id": int(i),
                "dataset": "Large-Scale-OR", "method": "LEAN-LLM-OPT-original",
                "model_profile": "gpt-4.1", "repeat_index": 0,
                "query": str(row["Query"]),
                "dataset_address": str(row.get("Dataset_address") or ""),
                "gold_type": str(row[type_col]),
                "gold_objective": str(row[gold_col]),
                "pred_type": cl[0] if cl else None,
                "size_class": str(row.get("Type by size") or ""),
                "status": status, "error": err,
                "model_output": om[0], "code_output": oc[0],
            }, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            mark = "ok " if status == "ok" else "FAIL"
            print(f"  [{n:>3}/{len(df)}] id={i:<4} {mark} "
                  f"{time.perf_counter() - t1:>5.1f}s"
                  + (f"  {err[:60]}" if err else ""))
    wall = time.perf_counter() - t0
    if n_fail:
        print(f"\n!! {n_fail} instance(s) failed; they are in the log with "
              f"status=failed and will score as wrong")
    json.dump({"run_id": run_id, "commit": sha, "notebook": NOTEBOOK_PATH,
               "n_instances": len(df), "wall_s": round(wall, 1),
               "no_sleep": args.no_sleep,
               "data_commit": args.data_commit, "data_source": data_note,
               "classifier_model": args.classifier,
               "classifier_substituted": args.classifier != "gpt-4",
               "refdata_substituted": "current real-CSV RefData.csv used; the "
                                      "committed one is xlsx and CSVLoader "
                                      "raises on it"},
              open(out_dir / "config.json", "w"), indent=1)

    if subs:
        print(f"\n[run_original] classifier patched to {args.classifier} in "
              f"cell(s) {[i for i, _ in subs]} "
              f"({sum(n for _, n in subs)} occurrence)")
    elif args.classifier != "gpt-4":
        raise SystemExit("--classifier asked for a substitution but no "
                         "model_name=\"gpt-4\" was found; the notebook changed")
    print(f"\n{len(df)} instance(s) in {wall:.0f}s -> {out_dir}")
    print("\nnext:")
    print(f"  python3 score_runs.py {OUT_ROOT} --run {run_id} "
          f"-o verify/out/gold_labels.csv")
    print(f"  python3 verify/verify_report.py {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
