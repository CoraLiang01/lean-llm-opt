#!/usr/bin/env python3
"""
run_notebook.py
===============
Run a pipeline notebook from the terminal, on a few instances.

    # see what would happen -- no API calls
    python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb --list

    # 2 instances of the CA query set
    python run_notebook.py LEAN_LLM_OPT_4.1_Air_NRM.ipynb \
        --data Test_Dataset/Air_NRM/small_scale/query_CA.csv --n 2

    # specific rows, as the notebook would index them
    python run_notebook.py LEAN_LLM_OPT_4.1_Large-scale-or.ipynb \
        --data Test_Dataset/Large-scale-or/Large-scale-or-101.csv --rows 15,45,81

Why this exists
---------------
Smoke-testing a notebook otherwise means opening it, restarting the kernel,
running fifteen setup cells by hand and remembering not to touch the cell that
runs the whole test set. That is slow and easy to get wrong -- one stray
Shift+Enter on the wrong cell spends real money on 101 instances.

This executes the notebook's setup cells (everything up to, but not including,
the first cell that *calls* the batch function) in one namespace, then calls
that batch function itself on the slice you asked for. Same code, same config
cell, same logging -- the only thing skipped is the cell that would have run
everything.

`jupyter nbconvert --execute` cannot do this: it runs every cell, including the
full test set.

Notes
-----
* Run from the repository root. The notebooks use repo-relative data paths.
* The config cell prompts for an API key if one is not in the environment,
  exactly as it does in Jupyter.
* `--list` stops before any cell executes, so it costs nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The two batch entry points across the notebook family. Air-NRM and the
# ablations expose Batch_Process_Queries(df); the large-scale notebooks expose
# run_test(df, agent). Both are detected at runtime rather than configured.
BATCH_FNS = ("Batch_Process_Queries", "run_test")
CALL_RE = re.compile(r"^(?!\s*def\s)\s*(?:\w+\s*(?:,\s*\w+\s*)*=\s*)?"
                     r"(" + "|".join(BATCH_FNS) + r")\s*\(", re.M)


def code_cells(nb: dict):
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if not src.strip():
            continue
        yield i, src


def split_at_first_call(nb: dict):
    """Setup cells, and the index of the first cell that runs the test set."""
    setup, stop = [], None
    for i, src in code_cells(nb):
        if CALL_RE.search(src):
            stop = i
            break
        setup.append((i, src))
    return setup, stop


def find_resumable_run(nb: dict, profile: str | None, repeat: int | None,
                       out_dir: Path) -> str | None:
    """
    Newest run directory that this invocation would be a continuation of.

    Run ids look like
        METHOD__PROFILE__DATASET__rN__TIMESTAMP__CONFIGHASH
    so method, profile, dataset and repeat can be matched straight off the
    directory name. The timestamp and the config hash deliberately are not:
    the timestamp is what makes a resumed run a *different* run id if we let
    it, and the hash may legitimately differ if the config was touched between
    the crash and the restart -- in which case the mismatch is worth seeing in
    config.json rather than silently starting over.
    """
    if not out_dir.exists():
        return None
    cell = next((s for _, s in code_cells(nb) if "lx.load_config(" in s), "")
    method = re.search(r"method\s*=\s*['\"]([^'\"]+)['\"]", cell)
    dataset = re.search(r"dataset\s*=\s*['\"]([^'\"]+)['\"]", cell)
    nb_profile = re.search(r"model_profile\s*=\s*['\"]([^'\"]+)['\"]", cell)
    nb_repeat = re.search(r"repeat_index\s*=\s*(\d+)", cell)

    want = [method.group(1) if method else None,
            profile or (nb_profile.group(1) if nb_profile else None),
            dataset.group(1) if dataset else None,
            f"r{repeat if repeat is not None else (nb_repeat.group(1) if nb_repeat else 0)}"]

    hits = []
    for d in out_dir.iterdir():
        if not (d / "instances.jsonl").exists():
            continue
        parts = d.name.split("__")
        if len(parts) < 4:
            continue
        if all(w is None or parts[i] == w for i, w in enumerate(want)):
            hits.append(d)
    if not hits:
        return None
    return max(hits, key=lambda d: d.stat().st_mtime).name


def patch_config_cell(src: str, profile: str | None, repeat: int | None,
                      resume_run_id: str | None = None,
                      dataset: str | None = None) -> str:
    """
    Override `model_profile=` / `repeat_index=` in the notebook's config cell.

    The notebooks hard-code both in the `lx.load_config(...)` call, which is
    fine in Jupyter and useless from a shell loop: running the same notebook at
    six repeat indices, or against a second model profile, would mean editing
    the file between runs -- and an edited notebook is a different notebook,
    which is exactly what the config hash is supposed to catch.

    Patching happens on an in-memory copy. The file on disk is never touched.
    """
    if profile:
        src, n = re.subn(r"model_profile\s*=\s*['\"][^'\"]*['\"]",
                         f"model_profile='{profile}'", src)
        if not n:
            raise SystemExit("--profile given but no model_profile= in the "
                             "config cell")
    if dataset:
        # Not cosmetic. cfg.dataset picks the path that gets fingerprinted into
        # config.json and names the run directory. The Air-NRM notebook is
        # hard-coded to the small-scale set, so running the large-scale queries
        # by only passing --data would log 15 large-scale answers under the
        # small-scale dataset's name and sha256 -- two different experiments,
        # indistinguishable afterwards.
        src, n = re.subn(r"dataset\s*=\s*['\"][^'\"]*['\"]",
                         f"dataset='{dataset}'", src)
        if not n:
            raise SystemExit("--dataset given but no dataset= in the config cell")
    if repeat is not None:
        # Some notebooks use a literal (repeat_index=0); others read the
        # default from LEANOPT_REPEAT with an expression containing commas.
        # Replace the complete keyword-argument line so both shapes work.
        src, n = re.subn(
            r"^(\s*)repeat_index\s*=.*,$",
            lambda m: f"{m.group(1)}repeat_index={repeat},",
            src,
            count=1,
            flags=re.M,
        )
        if not n:
            raise SystemExit("--repeat given but no repeat_index= in the "
                             "config cell")
    if resume_run_id:
        # RunLogger writes to out_dir/cfg.run_id and opens both jsonl files in
        # append mode, so pinning the id is the whole of "resume": the new
        # process writes into the same directory the crashed one was filling.
        # load_config() sets run_id itself (with a fresh timestamp), so this
        # has to land after that call and before RunLogger is constructed.
        src, n = re.subn(r"^(\s*)LOG\s*=\s*lx\.RunLogger\(",
                         lambda m: (f'{m.group(1)}CFG.run_id = '
                                    f'"{resume_run_id}"\n'
                                    f'{m.group(1)}print("[resume] appending to "\n'
                                    f'{m.group(1)}      "runs/{resume_run_id}")\n'
                                    + m.group(0)),
                         src, count=1, flags=re.M)
        if not n:
            raise SystemExit("--resume given but the config cell does not "
                             "construct LOG = lx.RunLogger(...)")
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--data", help="CSV to run; default = whatever the "
                                   "notebook's own run cell reads")
    ap.add_argument("--n", type=int, help="first N rows")
    ap.add_argument("--rows", help="comma-separated row labels, e.g. 15,45,81")
    ap.add_argument("--query-column", default="Query")
    ap.add_argument("--output-csv", help="write the returned result frame to "
                                         "this CSV path (relative to repo root)")
    ap.add_argument("--profile", help="model_profile to run with; overrides the "
                                      "one hard-coded in the notebook's config "
                                      "cell, e.g. openrouter-gpt-oss-20b")
    ap.add_argument("--dataset", help="dataset name from exp_config.yaml, e.g. "
                                      "Air-NRM-CA-LS. Overrides the notebook's "
                                      "own; with no --data the file is taken "
                                      "from the config's datasets: map")
    ap.add_argument("--repeat", type=int, help="repeat_index for this run; "
                                               "lands in the run_id as rN")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="RUN_ID",
                    help="continue an interrupted run instead of starting a "
                         "new one: reuse RUN_ID's directory and skip the "
                         "instances already logged there as ok. Bare "
                         "--resume picks the newest run matching this "
                         "method/profile/dataset/repeat.")
    ap.add_argument("--list", action="store_true",
                    help="show the plan and exit; executes nothing")
    args = ap.parse_args()

    nb_path = (HERE / args.notebook).resolve()
    if not nb_path.exists():
        raise SystemExit(f"notebook not found: {nb_path}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    resume_run_id = None
    if args.resume:
        resume_run_id = (args.resume if args.resume != "auto"
                         else find_resumable_run(nb, args.profile, args.repeat,
                                                 HERE / "runs"))
        if args.resume == "auto" and resume_run_id is None:
            print("--resume: nothing to continue, starting a fresh run")

    setup, stop = split_at_first_call(nb)
    if stop is None:
        raise SystemExit(
            f"no cell in {nb_path.name} calls any of {BATCH_FNS}; "
            f"this script does not know how to drive it")

    print(f"notebook : {nb_path.name}")
    print(f"setup    : {len(setup)} code cell(s), up to cell {setup[-1][0]}")
    print(f"skipping : cell {stop} onwards (that is the full-test-set cell)")
    if args.profile:
        print(f"profile  : {args.profile} (overriding the notebook's)")
    if args.repeat is not None:
        print(f"repeat   : r{args.repeat}")

    if args.list:
        print("\n--list: nothing executed, nothing spent.")
        return 0

    # The notebooks address their data relative to the repository root.
    import os
    os.chdir(HERE)

    ns: dict = {"__name__": "__main__"}
    for idx, src in setup:
        if "lx.load_config(" in src:
            src = patch_config_cell(src, args.profile, args.repeat,
                                    resume_run_id, args.dataset)
        stripped = src.strip()
        if stripped.startswith(("!", "%")):
            continue                       # shell / magic lines: notebook-only
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)   # noqa: S102
        except Exception as e:                                # noqa: BLE001
            print(f"\n!! setup cell {idx} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            raise

    fn_name = next((f for f in BATCH_FNS if callable(ns.get(f))), None)
    if fn_name is None:
        raise SystemExit(f"setup finished but none of {BATCH_FNS} is defined")
    fn = ns[fn_name]

    import pandas as pd
    cfg = ns.get("CFG")
    if not args.data and args.dataset and getattr(cfg, "dataset_path", None):
        args.data = cfg.dataset_path
        print(f"data     : {args.data} (from exp_config datasets:{args.dataset})")
    if args.data:
        df = pd.read_csv(HERE / args.data)
    else:
        src = "".join(nb["cells"][stop]["source"])
        m = re.search(r"read_csv\(\s*['\"]([^'\"]+)['\"]", src)
        if not m:
            raise SystemExit("could not infer the data file; pass --data")
        df = pd.read_csv(HERE / m.group(1))
        print(f"data     : {m.group(1)} (inferred from cell {stop})")

    if (args.dataset and args.data and cfg is not None
            and getattr(cfg, "dataset_path", None)
            and str(args.data) != str(cfg.dataset_path)):
        print(f"!! warning: --data {args.data} is not the file exp_config lists "
              f"for {args.dataset} ({cfg.dataset_path}); the run log will "
              f"fingerprint the file actually read", file=sys.stderr)

    if args.rows:
        labels = [int(r) for r in args.rows.split(",") if r.strip()]
        missing = [r for r in labels if r not in df.index]
        if missing:
            raise SystemExit(f"row labels not in the dataset: {missing}")
        df = df.loc[labels]          # .loc, not reset_index: instance ids must
    elif args.n:                     # stay unique across batches
        df = df.head(args.n)

    log = ns.get("LOG")
    if resume_run_id and log is not None:
        done = log.done_instance_ids()
        if done:
            before = len(df)
            df = df[~df.index.isin(done)]
            print(f"resume   : {len(done)} instance(s) already logged ok, "
                  f"{before - len(df)} skipped, {len(df)} left to run")
            if df.empty:
                print("nothing left to do for this run.")
                return 0

    print(f"running  : {len(df)} instance(s) through {fn_name}()\n")

    if fn_name == "Batch_Process_Queries":
        result = fn(df, query_column=args.query_column)
    else:
        agent = next((ns[k] for k in ("classification_agent", "agent",
                                      "classify_problem")
                      if callable(ns.get(k))), None)
        if agent is None:
            raise SystemExit("run_test needs a classification agent and none "
                             "was found in the notebook namespace")
        result = fn(df, agent)

    if log is not None:
        agg = log.close()
        print(f"\n{agg['n_instances']} instance(s) | "
              f"${agg['total_cost_usd']:.4f} | {agg['total_wall_s']:.0f}s")
        print(f"log: {log.dir}")

    if isinstance(result, tuple):
        result = result[0]
    if args.output_csv:
        output_path = (HERE / args.output_csv).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(result, "to_csv"):
            raise SystemExit("--output-csv requires the notebook runner to "
                             "return a dataframe")
        result.to_csv(output_path, index=False)
        print(f"result   : {output_path}")
    try:
        print(f"\nresult frame: {len(result)} row(s)")
    except Exception:                                        # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
