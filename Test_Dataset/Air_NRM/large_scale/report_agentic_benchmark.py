#!/usr/bin/env python3
"""Score one logged agentic run and emit machine- and human-readable reports."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from score_runs import score_run  # noqa: E402


def read_latest_records(run_dir: Path) -> list[dict]:
    latest: dict[object, dict] = {}
    path = run_dir / "instances.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record.get("instance_id")] = record
    return [latest[key] for key in sorted(latest, key=lambda x: (x is None, x))]


def objective_correct(df: pd.DataFrame, tolerance: float) -> pd.Series:
    def verdict(row):
        gold, objective = row.get("gold_objective"), row.get("objective")
        if pd.isna(gold):
            return math.nan
        if pd.isna(objective):
            return 0
        return int(abs(objective - gold) <= tolerance * max(1.0, abs(gold)))
    return df.apply(verdict, axis=1)


def accuracy(df: pd.DataFrame) -> dict:
    valid = df["correct_optimal"].dropna()
    correct = int(valid.sum()) if len(valid) else 0
    return {
        "correct": correct,
        "scored": int(len(valid)),
        "accuracy": round(correct / len(valid), 6) if len(valid) else None,
    }


def aggregate_resources(records: list[dict]) -> dict:
    keys = [
        "n_llm_calls", "n_llm_calls_failed", "n_tool_calls",
        "n_tool_calls_failed", "prompt_tokens", "completion_tokens",
        "cached_prompt_tokens", "reasoning_tokens", "total_tokens",
        "cost_usd", "llm_latency_s", "tool_latency_s", "wall_s",
    ]
    totals = {
        key: round(sum((r.get("summary") or {}).get(key) or 0 for r in records), 8)
        for key in keys
    }
    totals["n_instances"] = len(records)
    totals["n_failed_instances"] = sum(r.get("status") != "ok" for r in records)
    return totals


def build_per_instance(records: list[dict], scored: pd.DataFrame) -> pd.DataFrame:
    resources = []
    for record in records:
        summary = record.get("summary") or {}
        resources.append({
            "instance_id": record.get("instance_id"),
            "run_status": record.get("status"),
            "extraction_error": record.get("extraction_error", ""),
            "repair_attempted": bool(record.get("repair_attempted", False)),
            "repaired_successfully": bool(
                record.get("repaired_successfully", False)
            ),
            "final_local_execution_ok": (record.get("final_execution") or {}).get(
                "ok"
            ),
            "cost_limit_usd": record.get("cost_limit_usd"),
            "llm_calls": summary.get("n_llm_calls", 0),
            "failed_llm_calls": summary.get("n_llm_calls_failed", 0),
            "tool_calls": summary.get("n_tool_calls", 0),
            "prompt_tokens": summary.get("prompt_tokens", 0),
            "cached_prompt_tokens": summary.get("cached_prompt_tokens", 0),
            "completion_tokens": summary.get("completion_tokens", 0),
            "reasoning_tokens": summary.get("reasoning_tokens", 0),
            "total_tokens": summary.get("total_tokens", 0),
            "cost_usd": summary.get("cost_usd", 0),
            "llm_latency_s": summary.get("llm_latency_s", 0),
            "wall_s": summary.get("wall_s", 0),
        })
    resource_df = pd.DataFrame(resources)
    return scored.merge(resource_df, on="instance_id", how="left")


def write_markdown(run_dir: Path, config: dict, report: dict,
                   per_instance: pd.DataFrame) -> None:
    a4 = report["scores"]["relative_tolerance_1e-4"]
    a6 = report["scores"]["relative_tolerance_1e-6"]
    res = report["resources"]
    dataset = config.get("dataset") or "Optimization"
    lines = [
        f"# {dataset} Benchmark Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Model profile: `{config.get('model_profile')}`",
        f"- Model: `{config.get('models', {}).get('modeler', {}).get('model')}`",
        f"- Instances: {res['n_instances']} ({res['n_failed_instances']} failed)",
        f"- Accuracy @ 1e-4: {a4['correct']}/{a4['scored']} "
        f"({100 * a4['accuracy']:.2f}%)" if a4["accuracy"] is not None else
        "- Accuracy @ 1e-4: not scored",
        f"- Accuracy @ 1e-6: {a6['correct']}/{a6['scored']} "
        f"({100 * a6['accuracy']:.2f}%)" if a6["accuracy"] is not None else
        "- Accuracy @ 1e-6: not scored",
        f"- Calls: {int(res['n_llm_calls'])} LLM, "
        f"{int(res['n_tool_calls'])} execution/tool attempts",
        f"- Tokens: {int(res['prompt_tokens'])} input "
        f"({int(res['cached_prompt_tokens'])} cached), "
        f"{int(res['completion_tokens'])} output, "
        f"{int(res['reasoning_tokens'])} reasoning, "
        f"{int(res['total_tokens'])} total",
        f"- Estimated API cost: ${res['cost_usd']:.6f}",
        f"- LLM latency: {res['llm_latency_s']:.2f}s; "
        f"end-to-end wall time: {res['wall_s']:.2f}s",
        "",
        "## Per-instance results",
        "",
        "| ID | Status | Objective | Gold | @1e-4 | @1e-6 | LLM/exec | Repair | Tokens | Cost / limit | Solve error |",
        "|---:|:---|---:|---:|---:|---:|:---:|:---:|---:|---:|:---|",
    ]
    if report.get("safety_stop"):
        lines[-4:-4] = [f"- Safety stop: {report['safety_stop']}", ""]
    for _, row in per_instance.iterrows():
        def show_number(value):
            return "-" if pd.isna(value) else f"{value:.6g}"
        limit_value = row.get("cost_limit_usd")
        limit_text = ("-" if pd.isna(limit_value)
                      else f"${float(limit_value):.4f}")
        if row.get("repair_attempted"):
            repair_text = "success" if row.get("repaired_successfully") else "failed"
        else:
            repair_text = "not needed"
        lines.append(
            f"| {row['instance_id']} | {row.get('run_status', '-')} | "
            f"{show_number(row.get('objective'))} | {show_number(row.get('gold_objective'))} | "
            f"{int(row['correct_optimal']) if not pd.isna(row['correct_optimal']) else '-'} | "
            f"{int(row['correct_optimal_1e-6']) if not pd.isna(row['correct_optimal_1e-6']) else '-'} | "
            f"{int(row.get('llm_calls') or 0)}/{int(row.get('tool_calls') or 0)} | "
            f"{repair_text} | "
            f"{int(row.get('total_tokens') or 0)} | "
            f"${float(row.get('cost_usd') or 0):.6f} / "
            f"{limit_text} | "
            f"{str(row.get('solve_error') or '').replace('|', '/')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_existing_aggregate(run_dir: Path, scored_1e4: pd.DataFrame) -> None:
    # aggregate_runs expects these conventional files next to the gold input.
    scored_1e4.to_csv(run_dir / "scored_all.csv", index=False)
    gold = scored_1e4[[
        "dataset", "instance_id", "gold_type", "correct_optimal",
        "correct_formulation",
    ]]
    gold.to_csv(run_dir / "gold_labels.csv", index=False)
    cmd = [
        sys.executable, str(ROOT / "aggregate_runs.py"), str(ROOT / "runs"),
        "-o", str(run_dir / "report_tables"),
        "--run", run_dir.name,
        "--gold", str(run_dir / "gold_labels.csv"),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "aggregate_report.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        print(f"warning: aggregate_runs.py failed; see {run_dir / 'aggregate_report.log'}")


def run_parameter_diagnosis(run_dir: Path) -> None:
    cmd = [sys.executable, str(HERE / "diagnose_run.py"), "--run", run_dir.name]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (run_dir / "parameter_diagnosis.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        print(f"warning: parameter diagnosis failed; see {run_dir / 'parameter_diagnosis.txt'}")


def generate_report(run_dir: Path, timeout: float = 300.0,
                    diagnose: bool = True) -> dict:
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = read_latest_records(run_dir)
    run_summary_path = run_dir / "summary.json"
    run_summary = (json.loads(run_summary_path.read_text(encoding="utf-8"))
                   if run_summary_path.exists() else {})

    # One solve is sufficient: both metrics use the same objective and differ
    # only in their relative tolerance.
    scored_1e6 = score_run(run_dir, tol=1e-6, time_limit=timeout)
    scored_1e6 = scored_1e6.copy()
    scored_1e6["correct_optimal"] = objective_correct(scored_1e6, 1e-6)
    scored_1e6.to_csv(run_dir / "scored_1e-6.csv", index=False)
    scored_1e4 = scored_1e6.copy()
    scored_1e4["correct_optimal"] = objective_correct(scored_1e4, 1e-4)
    scored_1e4.to_csv(run_dir / "scored_1e-4.csv", index=False)
    scored_1e4.to_csv(run_dir / "scored.csv", index=False)

    per_instance = build_per_instance(records, scored_1e4)
    strict_verdict = scored_1e6.set_index("instance_id")["correct_optimal"]
    per_instance["correct_optimal_1e-6"] = per_instance["instance_id"].map(
        strict_verdict
    )
    per_instance.to_csv(run_dir / "per_instance_report.csv", index=False)
    report = {
        "run_id": config.get("run_id", run_dir.name),
        "dataset": config.get("dataset"),
        "method": config.get("method"),
        "model_profile": config.get("model_profile"),
        "scores": {
            "relative_tolerance_1e-4": accuracy(scored_1e4),
            "relative_tolerance_1e-6": accuracy(scored_1e6),
        },
        "resources": aggregate_resources(records),
        "pricing": config.get("models", {}).get("modeler", {}),
        "safety_stop": run_summary.get("safety_stop"),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    write_markdown(run_dir, config, report, per_instance)
    run_existing_aggregate(run_dir, scored_1e4)
    if diagnose:
        run_parameter_diagnosis(run_dir)
    print(f"report: {run_dir / 'report.md'}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-diagnosis", action="store_true")
    args = ap.parse_args()
    generate_report(args.run_dir, args.timeout, not args.no_diagnosis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
