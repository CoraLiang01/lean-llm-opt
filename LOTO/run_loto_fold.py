#!/usr/bin/env python3
"""
run_loto_fold.py
================
Run one Leave-One-Type-Out (LOTO) fold end-to-end, from the terminal.

    # F5 -- Assignment Problem, 5 instances x 5 seeds
    python run_loto_fold.py --holdout "Assignment Problem"

    # F1 -- Network Revenue Management, 25 instances x 3 seeds
    python run_loto_fold.py --holdout "Network Revenue Management"

    # all five folds in sequence
    python run_loto_fold.py --all

    # show the plan, execute nothing, spend nothing
    python run_loto_fold.py --holdout "Transportation" --list

How it works
------------
Same approach as ``run_notebook.py``: the notebook's cells are executed
**in this Python process** with ``exec()``, not through a Jupyter kernel.

That matters for two reasons:

1. **Environment.** ``jupyter nbconvert --execute`` resolves the kernel from
   the notebook's ``kernelspec`` metadata (``"python3"``) via the *global*
   Jupyter registry, which on an anaconda machine usually points at ``base``
   -- not the ``leanopt`` env you activated. Running in-process means the
   notebook uses exactly the interpreter that launched this script.

2. **The API key prompt.** ``lx.ensure_api_keys(CFG)`` defaults to
   ``interactive=True`` and calls ``getpass``. In a terminal that prompts you
   to paste the key, exactly as it always has. Under nbconvert there is no
   stdin, the prompt cannot appear, and the run dies with
   "missing API key(s): OPENAI_API_KEY".

Cells 0-33 are executed: setup, agents, the LOTO-filtered test set, the four
``run_test`` batches and the final merge. Everything from cell 34 onwards
(NL4OPT, IndustryOR, MAMO) is skipped -- LOTO only concerns Large-Scale-OR.

Per run the script:
  1. patches ``exp_config.yaml`` with ``holdout_type``;
  2. patches cell 1's ``repeat_index=`` for the current seed;
  3. relaxes cell 1's ``load_dotenv(override=True)`` to ``override=False`` so
     a key you pasted at the prompt survives into later repeats instead of
     being wiped by the empty ``OPENAI_API_KEY=`` line in ``.env``;
  4. executes cells 0-33;
  5. restores ``exp_config.yaml`` to ``holdout_type: null`` when finished.

The notebook file on disk is never modified -- all patching happens on an
in-memory copy of the cell sources.

Prerequisites
-------------
- Run from the ``LOTO/`` directory, with the ``leanopt`` conda env active.
- A key in ``.env`` (``python set_key.py OPENAI_API_KEY``) or just paste it at
  the prompt on the first repeat.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import textwrap
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_NOTEBOOK = "LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
CONFIG = "exp_config.yaml"

# Where the Large-Scale-OR part of a notebook ends. Everything after it is
# NL4OPT / IndustryOR / MAMO, which LOTO does not touch, so the boundary is
# found by looking for the first cell that loads one of those benchmarks
# rather than by hard-coding a cell number: the GPT-4.1 and gpt-oss notebooks
# have different layouts (config cell 1 vs 0, boundary 33 vs 26), and a
# hard-coded index that is silently wrong on one of them either skips the run
# cell or spends money on three extra benchmarks.
SMALL_SCALE_MARKERS = ("NL4OPT", "IndustryOR", "MAMO", "Test_Dataset/Small-scale")

# Rough USD per instance-run, used only for the pre-flight estimate printed by
# --all. GPT-4.1 measured at $0.038; gpt-oss-20b via OpenRouter is ~25x cheaper
# per token and spends more of them on reasoning.
COST_PER_INSTANCE = {"gpt-4.1": 0.038, "openrouter-gpt-4.1": 0.038,
                     "openrouter-gpt-oss-20b": 0.0015, "gpt-oss-20b": 0.0}

# Fold definitions: (holdout_type, expected_n, default_repeats)
#
# The names here must match ``candidate_types`` in exp_config.yaml exactly,
# because ``load_refdata_docs`` and ``get_candidate_types`` compare strings
# directly against RefData's 'New Problem Type' column. "Transportation" and
# "Transportation Problem" are NOT interchangeable there -- the former removes
# zero RefData rows and the leak assertion in cell 4 then (correctly) fires.
# ``resolve_holdout`` below maps any alias onto the canonical spelling so a
# near-miss on the command line is fixed rather than punished.
FOLDS = [
    ("Network Revenue Management", 25, 3),   # F1
    ("Resource Allocation",        22, 3),   # F2
    ("Facility Location Problem",  14, 3),   # F3
    ("Transportation Problem",      9, 5),   # F4  small n -> more seeds
    ("Assignment Problem",          5, 5),   # F5  small n -> more seeds
    # ---- RefData-only folds -------------------------------------------------
    # F6/F7 have no type-tailored workflow to remove: Mixture and Others
    # already fall through to the generic handler (see _FULL_ROUTES in cell 23,
    # which holds only the five types above). Holding them out therefore
    # removes the retrieval corpus and the candidate label, but leaves the
    # model-generation path untouched.
    #
    # That makes them a control rather than a degenerate case: F1-F5 confound
    # "lost the workflow" with "lost the reference examples", and these two
    # folds isolate the second term. Mixture is the strongest version of that
    # test -- its 24 rows are a quarter of the whole 96-row corpus.
    ("Mixture",                    18, 3),   # F6  24 RefData rows (25%)
    ("Others",                      8, 5),   # F7  16 RefData rows; n=8
]

# Names that are valid holdouts even though they are not in candidate_types.
# RefData's 'New Problem Type' column spells the generic buckets "Mixture" and
# "Others", while candidate_types splits the latter into "Others with CSV" and
# "Others without CSV". holdout_type must carry the RefData spelling, because
# load_refdata_docs() filters that column by exact equality.
REFDATA_ONLY_NAMES = {"Mixture", "Others"}


def resolve_holdout(name: str) -> str:
    """
    Map a holdout name onto the exact spelling used in ``candidate_types``.

    ``load_refdata_docs()`` filters RefData by exact string equality on the
    'New Problem Type' column, so "Transportation" silently removes nothing
    while "Transportation Problem" removes seven rows. Rather than relying on
    everyone typing the canonical form, resolve through ``type_aliases``.
    """
    try:
        import yaml
    except ImportError:
        return name                      # can't check; let the assertion catch it

    cfg_path = HERE / CONFIG
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cls = cfg.get("common", {}).get("classification", {})
    candidates = [str(c).strip() for c in cls.get("candidate_types", [])]
    aliases = cls.get("type_aliases") or {}

    name = name.strip()

    # A declared fold name wins outright. "Others" and "Mixture" are RefData
    # spellings that do not appear verbatim in candidate_types; rewriting them
    # to "Others with CSV" would break the RefData filter, which is the whole
    # point of those folds.
    fold_names = {f[0] for f in FOLDS}
    if name in fold_names or name in candidates:
        return name

    lowered = name.lower()
    for _, group in aliases.items():
        group_l = [str(a).strip().lower() for a in group]
        if lowered not in group_l:
            continue
        # Prefer a declared fold name over a raw candidate entry.
        for fold in fold_names:
            if fold.lower() in group_l:
                print(f"[resolve] '{name}' -> '{fold}' (via type_aliases)")
                return fold
        for cand in candidates:
            if cand.lower() in group_l:
                print(f"[resolve] '{name}' -> '{cand}' (via type_aliases)")
                return cand

    raise SystemExit(
        f"holdout '{name}' is not a known type.\n"
        f"valid folds          : {sorted(fold_names)}\n"
        f"valid candidate_types: {candidates}"
    )


# ── exp_config.yaml patching (comment-preserving) ────────────────────────────

def patch_yaml_holdout(config_path: Path, holdout_type: str | None):
    """Set ``holdout_type`` in exp_config.yaml, leaving comments intact."""
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^(\s*holdout_type:\s*)(.*)$", text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find 'holdout_type:' in {config_path.name}")
    new_value = "null" if holdout_type is None else holdout_type
    text = text[:match.start()] + f"{match.group(1)}{new_value}" + text[match.end():]
    config_path.write_text(text, encoding="utf-8")


# ── Notebook loading & per-repeat patching ───────────────────────────────────

class NotebookPlan:
    """Which notebook to run, and the two cell indices that drive it."""

    def __init__(self, name: str, profile: str | None = None):
        self.name = name
        self.path = HERE / name
        if not self.path.exists():
            raise SystemExit(f"{name} not found in {HERE}; "
                             f"run this script from the LOTO/ directory")
        self.profile = profile
        nb = json.loads(self.path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

        self.config_cell = None
        self.last_cell = len(cells) - 1
        for i, c in enumerate(cells):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            if self.config_cell is None and "lx.load_config(" in src:
                self.config_cell = i
            if any(m in src for m in SMALL_SCALE_MARKERS):
                self.last_cell = i - 1
                break

        if self.config_cell is None:
            raise SystemExit(f"{name} has no cell calling lx.load_config(); "
                             f"this script does not know how to configure it")


def load_code_cells(nb_path: Path, last_cell: int) -> list[tuple[int, str]]:
    """Return [(cell_index, source)] for code cells 0..last_cell."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = []
    for i, c in enumerate(nb.get("cells", [])):
        if i > last_cell:
            break
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if src.strip():
            cells.append((i, src))
    return cells


def patch_config_cell(src: str, repeat_index: int,
                      profile: str | None = None) -> str:
    """
    Patch the configuration cell for one repeat:

    * ``repeat_index=<N>``           -> the seed we are running;
    * ``model_profile='...'``        -> ``--profile``, when given;
    * ``load_dotenv(override=True)`` -> ``override=False``.

    The second one is the subtle bit. ``.env`` normally carries an empty
    ``OPENAI_API_KEY=`` line because the key is pasted at the prompt rather
    than stored on disk. With ``override=True`` that empty value clobbers the
    key already in ``os.environ`` on every repeat after the first, so the run
    would stop and re-prompt (or fail) each time. ``override=False`` keeps a
    real key that is already in the environment and still lets a populated
    ``.env`` work normally.
    """
    if "repeat_index=" not in src:
        raise SystemExit("could not find 'repeat_index=' in the config cell")
    src = re.sub(r"repeat_index\s*=\s*\d+", f"repeat_index={repeat_index}", src)
    if profile:
        src, n = re.subn(r"model_profile\s*=\s*['\"][^'\"]*['\"]",
                         f"model_profile='{profile}'", src)
        if not n:
            raise SystemExit("--profile given but the config cell has no "
                             "model_profile=")
    src = re.sub(r"load_dotenv\(\s*override\s*=\s*True\s*\)",
                 "load_dotenv(override=False)", src)
    return src


# ── Execution ────────────────────────────────────────────────────────────────

def execute_cells(cells: list[tuple[int, str]], repeat_index: int,
                  plan: NotebookPlan) -> dict:
    """
    Execute the notebook cells in a fresh namespace, in *this* process.
    Returns the namespace so the caller can read LOG, CFG, etc.
    """
    ns: dict = {"__name__": "__main__"}
    for idx, src in cells:
        if idx == plan.config_cell:
            src = patch_config_cell(src, repeat_index, plan.profile)
        stripped = src.lstrip()
        if stripped.startswith(("!", "%")):
            continue                      # shell / magic lines are notebook-only
        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)   # noqa: S102
        except SystemExit:
            # Cell 33 calls sys.exit(1) if the merge fails -- do not let that
            # take down the whole multi-seed run.
            print(f"    cell {idx} called sys.exit(); continuing", file=sys.stderr)
        except Exception as e:                                # noqa: BLE001
            print(f"\n    !! cell {idx} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            raise
    return ns


def run_one_repeat(cells, repeat_index: int,
                   plan: NotebookPlan) -> tuple[str, float, str]:
    """Run a single seed. Returns (status, elapsed_seconds, log_dir)."""
    t0 = time.time()
    try:
        ns = execute_cells(cells, repeat_index, plan)
    except Exception:                                         # noqa: BLE001
        return "FAILED", time.time() - t0, ""

    elapsed = time.time() - t0
    log = ns.get("LOG")
    log_dir = ""
    if log is not None:
        try:
            agg = log.close()
            log_dir = str(log.dir)
            print(f"    {agg['n_instances']} instance(s) | "
                  f"${agg['total_cost_usd']:.4f} | {agg['total_wall_s']:.0f}s")
            print(f"    log: {log_dir}")
        except Exception as e:                                # noqa: BLE001
            print(f"    (could not close log: {e})")
    return "ok", elapsed, log_dir


# ── Fold driver ──────────────────────────────────────────────────────────────

def run_fold(holdout: str, repeats: int, plan: NotebookPlan,
             list_only: bool = False) -> int:
    holdout = resolve_holdout(holdout)
    expected_n = next((n for h, n, _ in FOLDS if h == holdout), None)
    if expected_n is None:
        print(f"warning: '{holdout}' is not in the standard fold list")

    print(f"\n{'=' * 62}")
    print(f"LOTO FOLD : {holdout}")
    if expected_n:
        print(f"instances : {expected_n} expected")
    print(f"seeds     : {repeats}")
    print(f"{'=' * 62}")

    cfg_path = HERE / CONFIG
    cells = load_code_cells(plan.path, plan.last_cell)
    print(f"notebook  : {plan.name}")
    if plan.profile:
        print(f"profile   : {plan.profile} (overriding the notebook's)")
    print(f"executing : {len(cells)} code cell(s), cells 0-{plan.last_cell} "
          f"(config cell {plan.config_cell})")
    print(f"skipping  : cell {plan.last_cell + 1} onwards "
          f"(NL4OPT / IndustryOR / MAMO)")

    if list_only:
        print("\n--list: nothing executed, nothing spent.")
        return 0

    patch_yaml_holdout(cfg_path, holdout)
    print(f"config    : holdout_type = {holdout}")

    results = []
    try:
        for ri in range(repeats):
            print(f"\n--- seed {ri + 1}/{repeats}  (repeat_index={ri}) "
                  f"{'-' * 28}")
            status, elapsed, _ = run_one_repeat(cells, ri, plan)
            results.append((status, elapsed))
            print(f"    {'done' if status == 'ok' else 'FAILED'} "
                  f"in {elapsed:.0f}s")
    finally:
        patch_yaml_holdout(cfg_path, None)
        print(f"\nconfig    : restored holdout_type = null")

    print(f"\n{'-' * 62}")
    print(f"SUMMARY: {holdout}")
    print(f"{'-' * 62}")
    for i, (status, elapsed) in enumerate(results):
        mark = "ok  " if status == "ok" else "FAIL"
        print(f"  seed {i}: {mark}  {elapsed:.0f}s")

    failed = sum(1 for s, _ in results if s != "ok")
    print(f"\n  {repeats - failed}/{repeats} seed(s) completed")
    return failed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Leave-One-Type-Out folds for LEAN-LLM-OPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            folds:
              F1  Network Revenue Management   25 instances   3 seeds
              F2  Resource Allocation          22 instances   3 seeds
              F3  Facility Location Problem    14 instances   3 seeds
              F4  Transportation                9 instances   5 seeds
              F5  Assignment Problem            5 instances   5 seeds
        """),
    )
    ap.add_argument("--holdout", help='e.g. "Assignment Problem"')
    ap.add_argument("--repeats", type=int, default=None,
                    help="override the default seed count for this fold")
    ap.add_argument("--all", action="store_true",
                    help="run all five folds in sequence")
    ap.add_argument("--skip", action="append", default=[], metavar="TYPE",
                    help="with --all, skip this fold (repeatable). Use it to "
                         "resume without re-running folds already finished, "
                         'e.g. --all --skip "Assignment Problem"')
    ap.add_argument("--notebook", default=DEFAULT_NOTEBOOK,
                    help=f"pipeline notebook to run (default: "
                         f"{DEFAULT_NOTEBOOK}). Use "
                         f"LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb for "
                         f"the gpt-oss-20b folds")
    ap.add_argument("--profile", help="model_profile override, e.g. "
                                      "openrouter-gpt-oss-20b")
    ap.add_argument("--list", action="store_true",
                    help="show the plan and exit; executes nothing")
    args = ap.parse_args()

    if not args.holdout and not args.all:
        ap.error("specify --holdout <type> or --all")

    if not (HERE / CONFIG).exists():
        raise SystemExit(f"{CONFIG} not found in {HERE}; "
                         f"run this script from the LOTO/ directory")

    # The notebook addresses its data relative to the LOTO directory.
    os.chdir(HERE)

    plan = NotebookPlan(args.notebook, args.profile)

    print(f"env       : {sys.prefix}")
    print(f"python    : {sys.executable}")

    if args.all:
        # Resolve --skip through the same alias map, so --skip "Transportation"
        # and --skip "Transportation Problem" both work.
        skip = {resolve_holdout(s).lower() for s in args.skip}
        todo = [f for f in FOLDS if f[0].lower() not in skip]
        skipped = [f[0] for f in FOLDS if f[0].lower() in skip]

        n_runs = sum(n * (args.repeats or dr) for _, n, dr in todo)
        rate = COST_PER_INSTANCE.get(args.profile or "gpt-4.1", 0.038)
        print(f"folds     : {len(todo)} to run"
              + (f", skipping {skipped}" if skipped else ""))
        if rate:
            print(f"runs      : {n_runs} instance-run(s), "
                  f"~${n_runs * rate:.2f} at ${rate}/instance")
        else:
            # A local model has no per-token price; the cost of this run is
            # GPU time, which is what hardware_note and the wall-clock columns
            # are for. Printing "$0.00" without saying why reads like a broken
            # estimator.
            print(f"runs      : {n_runs} instance-run(s), local model "
                  f"(no API cost; the cost here is GPU wall-clock)")

        total = 0
        for holdout, _, default_repeats in todo:
            total += run_fold(holdout,
                              args.repeats or default_repeats,
                              plan,
                              list_only=args.list)
        print(f"\n{'=' * 62}")
        print("ALL FOLDS COMPLETE" if not total
              else f"ALL FOLDS COMPLETE -- {total} seed(s) failed")
        print(f"{'=' * 62}")
        return 1 if total else 0

    repeats = args.repeats
    if repeats is None:
        repeats = next((dr for h, _, dr in FOLDS if h == args.holdout), 3)
    return 1 if run_fold(args.holdout, repeats, plan,
                         list_only=args.list) else 0


if __name__ == "__main__":
    sys.exit(main())
