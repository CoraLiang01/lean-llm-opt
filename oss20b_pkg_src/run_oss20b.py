#!/usr/bin/env python3
"""
run_oss20b.py
=============
One entry point for all three gpt-oss-20B experiments.

    # smoke first -- a few instances, ~10 minutes, graded automatically
    python run_oss20b.py --task large_scale_or --smoke

    # the real thing
    nohup python run_oss20b.py \\
      --task large_scale_or \\
      --repeats 3 \\
      --output-dir runs \\
      > large_scale_or.log 2>&1 &

Tasks
-----
    large_scale_or   101 instances x --repeats            ~6-9 h at 3 repeats
    loto             5 folds + Mixture, 307 instance-runs ~6-9 h
    air_nrm_ls       15 large SBLP instances x --repeats  ~2-4 h at 3 repeats

Interrupted? Re-run the same command.
  * large_scale_or and air_nrm_ls continue in the same run directory and skip
    the instances already logged ok, so only the lost work is repeated.
  * loto restarts the seed it was in the middle of (folds are 5-25 instances)
    and leaves finished seeds alone.

This script does not decide anything about the experiment -- it sets up the
directory, runs the same drivers you would run by hand, and packs the results.
Everything that defines the experiment lives in exp_config.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASKS = {
    "large_scale_or": {
        "dir": "large_scale_or",
        "notebook": "LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb",
        "dataset": "Large-Scale-OR",
        "data": "Test_Dataset/Large-scale-or/Large-scale-or-101.csv",
        # One instance of each of three different workflows, so a smoke test
        # exercises the router rather than one branch of it.
        "smoke_rows": "15,45,81",
        "n": 101,
        "hours": "6-9 h for 3 repeats",
    },
    "air_nrm_ls": {
        "dir": "air_nrm_ls",
        "notebook": "LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb",
        "dataset": "Air-NRM-CA-LS",
        "data": "Test_Dataset/Air_NRM/large_scale/query_CA.csv",
        # Smallest and largest instance: 112 and 421 variables. If size is
        # going to break something, it breaks at one of these two.
        "smoke_rows": "0,14",
        "n": 15,
        "hours": "2-4 h for 3 repeats",
    },
    "loto": {
        "dir": "loto",
        "notebook": "LEAN_LLM_OPT_gpt_oss_20b_LOTO.ipynb",
        "dataset": "Large-Scale-OR",
        "data": None,
        "smoke_rows": None,
        "n": 307,
        "hours": "6-9 h",
    },
}


def log(msg: str = "") -> None:
    """Timestamped, unbuffered -- this runs under nohup for hours."""
    if msg:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    else:
        print(flush=True)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def die(msg: str) -> "NoReturn":                                  # noqa: F821
    print(f"\nSTOPPED: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# --------------------------------------------------------------------------- #
def preflight(profile: str) -> None:
    step("preflight")
    try:
        import langchain_classic, langchain_ollama, faiss, pandas, yaml, gurobipy  # noqa
    except ImportError as e:
        die(f"missing package ({e.name}) -- run: "
            f"pip install -r requirements-server.txt")
    log(f"python {sys.version.split()[0]}  profile {profile}")

    if not profile.startswith("openrouter"):
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as r:
                tags = json.load(r)
        except (urllib.error.URLError, OSError) as e:
            die(f"no Ollama server at {base} ({e}) -- start it with: ollama serve")
        have = {m.get("name", "").split(":")[0]: m.get("name") for m in
                tags.get("models", [])}
        names = {m.get("name") for m in tags.get("models", [])}
        for want in ("gpt-oss:20b", "nomic-embed-text"):
            if want not in names and want.split(":")[0] not in have:
                die(f"model {want} not pulled -- run: ollama pull {want}")
        log(f"ollama at {base}: gpt-oss:20b and nomic-embed-text present")


def separate_smoke_data(task_dir: Path, out_dir: str) -> None:
    """
    Move any earlier (smoke) runs out of the way, once.

    The smoke test writes into the same directory. Left there, its instances
    are aggregated together with the real ones -- and for loto that means one
    fold counted twice under two seeds, quietly changing that fold's n. The
    marker file is what makes this safe to re-run: after the real run has
    started nothing is moved again, so an interruption is resumed, not reset.
    """
    runs = task_dir / out_dir
    runs.mkdir(parents=True, exist_ok=True)
    marker = runs / ".full_started"
    if marker.exists():
        log("real runs already started here -- continuing them, not restarting")
        return
    existing = [d for d in runs.iterdir() if d.is_dir()]
    if existing:
        aside = task_dir.parent / f"{task_dir.name}_runs_smoke"
        aside.mkdir(exist_ok=True)
        for d in existing:
            shutil.move(str(d), str(aside / d.name))
        log(f"moved {len(existing)} earlier run(s) to {aside.name}/")
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")


def run(cmd: list[str], cwd: Path, env: dict, dry: bool = False) -> int:
    # shlex.join so the printed line is one you can paste back verbatim;
    # a bare join loses the quotes around --holdout "Assignment Problem".
    log("$ " + shlex.join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one of the three gpt-oss-20B experiments.")
    ap.add_argument("--task", required=True, choices=sorted(TASKS),
                    help="which experiment to run")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats of the whole benchmark (default 3). "
                         "temperature=0 is not determinism: the GPT-4.1 arm "
                         "moved 73.3%%-78.2%% across repeats of an identical "
                         "config, so single-run numbers are not reportable. "
                         "Ignored by --task loto, whose seed counts are per "
                         "fold and fixed to match the GPT-4.1 panel.")
    ap.add_argument("--output-dir", default="runs",
                    help="where run directories are written, relative to the "
                         "task directory (default: runs)")
    ap.add_argument("--profile", default=os.environ.get("PROFILE", "gpt-oss-20b"),
                    help="model profile from exp_config.yaml (default: "
                         "gpt-oss-20b, the local Ollama one)")
    ap.add_argument("--smoke", action="store_true",
                    help="run only a couple of instances and grade them; "
                         "does not touch the real-run marker")
    ap.add_argument("--no-resume", action="store_true",
                    help="start a fresh run directory instead of continuing "
                         "an interrupted one")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit")
    args = ap.parse_args()

    spec = TASKS[args.task]
    task_dir = ROOT / spec["dir"]
    if not task_dir.exists():
        die(f"missing task directory: {task_dir}")

    print("=" * 68)
    print(f"  task        {args.task}")
    print(f"  profile     {args.profile}")
    print(f"  mode        {'SMOKE' if args.smoke else 'FULL RUN'}")
    if not args.smoke:
        print(f"  repeats     {args.repeats if args.task != 'loto' else 'per-fold (3/3/3/5/5/3)'}")
        print(f"  expect      {spec['hours']}")
    print(f"  output      {spec['dir']}/{args.output_dir}/")
    print(f"  started     {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68, flush=True)

    if not args.dry_run:
        preflight(args.profile)

    env = dict(os.environ)
    # Every driver picks the log directory up from here, so --output-dir works
    # without a separate flag on each of them.
    env["LEANOPT_OUT_DIR"] = args.output_dir
    env["PYTHONUNBUFFERED"] = "1"

    if not args.smoke and not args.dry_run:
        step("separating smoke data from the real runs")
        separate_smoke_data(task_dir, args.output_dir)

    t0 = time.time()
    rc = 0

    if args.task == "loto":
        step("LOTO folds")
        cmd = [sys.executable, "run_loto_fold.py",
               "--notebook", spec["notebook"], "--profile", args.profile]
        if args.smoke:
            # The Assignment Problem fold: 5 instances, the cheapest way to
            # prove the leakage machinery works end to end.
            cmd += ["--holdout", "Assignment Problem", "--repeats", "1"]
        else:
            # "Others" is skipped because the GPT-4.1 panel does not report it.
            cmd += ["--all", "--skip", "Others"]
        rc = run(cmd, task_dir, env, args.dry_run)
    else:
        repeats = 1 if args.smoke else args.repeats
        for r in range(repeats):
            step(f"repeat r{r}" if not args.smoke else "smoke")
            cmd = [sys.executable, "run_notebook.py", spec["notebook"],
                   "--profile", args.profile, "--repeat", str(r),
                   "--dataset", spec["dataset"], "--data", spec["data"]]
            if args.smoke:
                cmd += ["--rows", spec["smoke_rows"]]
            elif not args.no_resume:
                cmd += ["--resume"]
            code = run(cmd, task_dir, env, args.dry_run)
            rc = rc or code
            if code:
                log(f"repeat r{r} exited {code}; continuing with the next one")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    elapsed = time.time() - t0
    step(f"finished in {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m "
         f"(exit {rc})")

    # ---- grade, or pack --------------------------------------------------
    sys.path.insert(0, str(ROOT))
    if args.smoke:
        step("verdict")
        try:
            import check_smoke
            rc = check_smoke.grade(args.task, args.output_dir)
        except Exception as e:                                    # noqa: BLE001
            log(f"could not grade automatically: {type(e).__name__}: {e}")

    step("packing results")
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    kind = "smoke" if args.smoke else "runs"
    tarball = ROOT / f"{kind}_{args.task}_{stamp}.tar.gz"
    subprocess.run(["tar", "czf", str(tarball), "-C", str(ROOT),
                    f"{spec['dir']}/{args.output_dir}"], check=False)
    if tarball.exists():
        log(f"packed: {tarball} ({tarball.stat().st_size / 1e6:.1f} MB)")

    print(f"""
Send back:
  1. {tarball.name}
  2. this log file
  3. hardware line: GPU model, driver/CUDA, `ollama --version`, quantisation
     (`ollama show gpt-oss:20b`)
""", flush=True)
    if args.smoke:
        print("Do not start the full runs until we have looked at the above.\n",
              flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
