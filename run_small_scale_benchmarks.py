#!/usr/bin/env python3
"""Run the four small-scale datasets with the current GPT-4.1 LEAN pipeline.

Examples (run with the project's Python environment):
    python run_small_scale_benchmarks.py --list
    python run_small_scale_benchmarks.py --preflight
    python run_small_scale_benchmarks.py --limit 2
    python run_small_scale_benchmarks.py
    python run_small_scale_benchmarks.py --repeats 3

Every invocation uses a separate output directory. No paid calls are made by
--list or --preflight. Normal runs preflight all four datasets before starting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {
    "IndustryOR": "Test_Dataset/Small-scale/IndustryOR_fixedV2.csv",
    "MAMO-easy": "Test_Dataset/Small-scale/MAMO_EasyLP_fixed.csv",
    "MAMO-complex": "Test_Dataset/Small-scale/MAMO_ComplexLP_fixed.csv",
    "NL4OPT": "Test_Dataset/Small-scale/NL4OPT NEW.csv",
}


def run_command(arguments: list[str], env: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-u", *arguments], cwd=HERE, env=env, check=True)


def write_reports(output: Path, env: dict[str, str]) -> None:
    runs = output / "runs"
    if not any(runs.glob("*/instances.jsonl")):
        return
    tables = output / "tables"
    run_command(["aggregate_runs.py", str(runs), "-o", str(tables)], env)

    import pandas as pd

    frame = pd.read_csv(tables / "per_instance.csv")
    rows = []
    for (dataset, method, profile), group in frame.groupby(
        ["dataset", "method", "model_profile"], sort=False
    ):
        rows.append({
            "dataset": dataset,
            "method": method,
            "model_profile": profile,
            "n_instances": len(group),
            "n_unique_instances": group["instance_id"].nunique(),
            "n_repeats": group["repeat_index"].nunique(),
            "n_failed": int(group["status"].ne("ok").sum()),
            "runtime_s_total": group["wall_s"].sum(),
            "runtime_s_mean": group["wall_s"].mean(),
            "runtime_s_p95": group["wall_s"].quantile(0.95),
            "api_cost_usd_total": group["cost_usd"].sum(),
            "api_cost_usd_mean": group["cost_usd"].mean(),
            "llm_calls_total": group["n_llm_calls"].sum(),
            "llm_calls_mean": group["n_llm_calls"].mean(),
            "tool_calls_total": group["n_tool_calls"].sum(),
            "tool_calls_mean": group["n_tool_calls"].mean(),
            "retriever_calls_total": group["n_retriever_calls"].sum(),
            "prompt_tokens_total": group["prompt_tokens"].sum(),
            "completion_tokens_total": group["completion_tokens"].sum(),
            "total_tokens": group["total_tokens"].sum(),
            "n_estimated_token_calls": group["n_estimated_token_calls"].sum(),
        })
    summary = pd.DataFrame(rows).round(6)
    summary.to_csv(output / "resource_summary.csv", index=False)
    print("\nResource summary (includes failed instances):", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSummary: {output / 'resource_summary.csv'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="show the plan; no API calls")
    mode.add_argument("--preflight", action="store_true", help="validate setup and all rows; no API calls")
    parser.add_argument("--limit", type=int, help="first N rows of each dataset")
    parser.add_argument("--repeats", type=int, default=1, help="independent runs per dataset (default: 1)")
    parser.add_argument("--out", type=Path, help="new output directory (default: a unique directory under outputs/)")
    args = parser.parse_args()
    if args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--repeats and --limit must be positive")

    env = dict(os.environ)
    commands = []
    total = 0
    for dataset, relative in DATASETS.items():
        with (HERE / relative).open(encoding="utf-8-sig", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        selected = min(count, args.limit) if args.limit else count
        total += selected * args.repeats
        print(f"{dataset}: {selected}/{count} rows x {args.repeats} repeat(s)", flush=True)
        base = ["run_v2_101.py", "--dataset", dataset, "--data", relative,
                "--profile", "gpt-4.1", "--method", "LEAN-LLM-OPT-V2"]
        if args.limit is not None:
            base += ["--limit", str(args.limit)]
        commands.append((dataset, base))
    print(f"Total planned executions: {total}\n", flush=True)

    # Preflight parses every row, including special NL4OPT labels, and executes
    # setup definitions only. No model, embedding, or solver calls are made.
    for _, base in commands:
        run_command([*base, "--list" if args.list else "--preflight"], env)
    if args.list or args.preflight:
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = (args.out or HERE / "outputs" / "small_scale_resources" / stamp).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env["LEANOPT_OUT_DIR"] = str(output / "runs")
    env["LEANOPT_RUN_SUFFIX"] = stamp
    (output / "plan.json").write_text(json.dumps({
        "datasets": DATASETS, "profile": "gpt-4.1", "method": "LEAN-LLM-OPT-V2",
        "limit_per_dataset": args.limit, "repeats": args.repeats,
        "planned_executions": total, "python": sys.executable,
        "runtime_scope": "per-instance classification, retrieval, modeling, code generation and solving",
        "cost_scope": "logged LLM and embedding usage priced using exp_config.yaml",
    }, indent=2), encoding="utf-8")
    print(f"\nOutput: {output}\n", flush=True)
    started = time.perf_counter()
    try:
        for repeat in range(args.repeats):
            for dataset, base in commands:
                run_command([
                    *base, "--repeat", str(repeat), "--run-dir-marker",
                    str(output / f"latest_{dataset}_r{repeat}.txt"),
                ], env)
    finally:
        (output / "session_timing.json").write_text(json.dumps({
            "session_wall_s": round(time.perf_counter() - started, 3),
            "scope": "runner processes including setup and pipeline output; excludes preflight and aggregation",
        }, indent=2), encoding="utf-8")
        write_reports(output, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
