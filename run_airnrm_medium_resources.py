#!/usr/bin/env python3
"""Run the missing medium Air-NRM baselines using the existing LS protocol."""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "Test_Dataset/Air_NRM/large_scale/run_agentic_benchmark.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "gpt-5.2", "gpt-5.4",
                                             "gemini-3.1-pro", "gemini-3-pro"),
                        default="all")
    parser.add_argument("--family", choices=("all", "ca", "np"), default="all")
    parser.add_argument("--workflow", choices=("all", "single", "two-stage-repair"),
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-run-cost-usd", type=float, default=30.0)
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args()
    if args.max_run_cost_usd < 0:
        parser.error("--max-run-cost-usd must be nonnegative")
    models = ["gpt-5.2", "gemini-3.1-pro"] if args.model == "all" else [args.model]
    families = ["ca", "np"] if args.family == "all" else [args.family]
    workflows = ["single", "two-stage-repair"] if args.workflow == "all" else [args.workflow]
    jobs = []
    for model in models:
        for workflow in workflows:
            for family in families:
                jobs.append((model, [sys.executable, str(RUNNER), model,
                    "--scale", "medium", "--dataset", family,
                    "--workflow", workflow, "--repeat-index", str(args.repeat_index),
                    "--max-run-cost-usd", str(args.max_run_cost_usd)]))
    # Validate every selected job before requesting a key or starting paid work.
    for _, command in jobs:
        result = subprocess.run(command + ["--dry-run"], cwd=ROOT)
        if result.returncode:
            return result.returncode
    if args.dry_run:
        print(f"Validated {len(jobs)} runs; no API requests were made.", flush=True)
        return 0
    run_env = os.environ.copy()
    suffix = "medium_resources_" + uuid.uuid4().hex[:12]
    run_env["LEANOPT_RUN_SUFFIX"] = suffix
    for model in models:
        variable = "GOOGLE_API_KEY" if model.startswith("gemini") else "OPENAI_API_KEY"
        if not run_env.get(variable):
            key = getpass.getpass(f"{variable} (hidden, process only): ").strip()
            if not key:
                raise SystemExit(f"Empty {variable}")
            run_env[variable] = key
    failures = 0
    print(f"Starting {len(jobs)} runs; suffix={suffix}", flush=True)
    for _, command in jobs:
        result = subprocess.run(command, cwd=ROOT, env=run_env)
        failures += result.returncode != 0
    print(f"Finished; {failures} failed commands. Reports:", flush=True)
    for run_dir in sorted((ROOT / "runs").glob(f"*__u{suffix}__*")):
        print(run_dir / "report.md")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
