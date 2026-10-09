"""One-pass evaluations with immutable inputs, attempts, logs and versioned results.

Run from the project environment. No code/solver repair or outcome-based rerun exists.
The v5 user policy permits discarded ReAct protocol attempts within the case deadline.
The reference objective is read only by the local scorer, never placed in a prompt.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/react_revision_20261006"
NAMES = {
    "full": "LEAN_LLM_OPT_4.1_Large-scale_1006.ipynb",
    "rag_only": "Ablation_Study_Large_Scale_Or_RAG_Only.ipynb",
    "few_shot_only": "Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb",
    "examples_only": "LOTO_Examples_Only_GPT4.1_Large-scale.ipynb",
    "examples_and_route": "LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb",
}
NS = None
EXTERNAL_CASE_DEADLINE_SECONDS = 1800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_full_definitions(left, right):
    """Allow descriptive Markdown edits and the reviewed 606 switch; code is fixed."""
    a, b = json.loads(Path(left).read_text()), json.loads(Path(right).read_text())
    if len(a["cells"]) != len(b["cells"]):
        return False
    return all(x["cell_type"] == y["cell_type"] and
               (x["cell_type"] == "markdown" or x["source"] == y["source"])
               for i, (x, y) in enumerate(zip(a["cells"], b["cells"])) if i != 39)


def recorded_query_only_alias_correction(frozen, original, method):
    """Allow only the audited no-CSV alias correction for the preserved CSV pass."""
    control_path = OUT / "delivery_control.json"
    if not control_path.exists():
        return False
    update = json.loads(control_path.read_text()).get("query_only_alias_corrections", {}).get(method)
    if not update or sha(frozen) != update["evaluated_frozen_sha256"] or sha(original) != update["delivered_sha256"]:
        return False
    a, b = json.loads(frozen.read_text()), json.loads(original.read_text())
    expected = "".join(a["cells"][39]["source"]).replace(update["old_fragment"], update["new_fragment"])
    return (len(a["cells"]) == len(b["cells"]) and expected == "".join(b["cells"][39]["source"]) and
            all(x["source"] == y["source"] for i,(x,y) in enumerate(zip(a["cells"],b["cells"])) if i != 39))


def namespace(path, method):
    ns = {"__name__": "__evaluation__"}
    book = json.loads(path.read_text())
    for i, cell in enumerate(book["cells"]):
        if cell["cell_type"] != "code":
            continue
        # Execute definitions, never an experiment/preflight cell.
        if i > 35 and method == "full":
            # Additional dataset loader functions share a switch cell; switches are false.
            if i not in (45, 47):
                continue
        if method.startswith("examples") and i in (43, 45):
            continue
        exec(compile("".join(cell["source"]), f"{path}:cell{i}", "exec"), ns)
    ns["NOTEBOOK_PATH"] = path
    return ns


def initialize(path, method):
    global NS
    os.chdir(ROOT)
    os.environ["OMP_NUM_THREADS"] = "1"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        NS = namespace(Path(path), method)
        NS["gp"].setParam("Threads", 2)
    NS["EVALUATION_METHOD"] = method


def worker(item):
    ns = NS
    case = item["case"]
    folder = Path(item["folder"])
    started = time.monotonic()
    timed_out = threading.Event()
    def interrupt_at_deadline():
        timed_out.set()
        os.kill(os.getpid(), signal.SIGINT)
    deadline = threading.Timer(EXTERNAL_CASE_DEADLINE_SECONDS, interrupt_at_deadline)
    deadline.daemon = True
    deadline.start()
    with (folder / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            if ns["EVALUATION_METHOD"].startswith("examples"):
                fold = ns["set_loto_fold"](case["true_label"])
                (folder / "fold_manifest.json").write_text(json.dumps(fold, indent=2))
            record = ns["execute_pipeline_case"](case)
        except BaseException as exc:
            if timed_out.is_set():
                original = exc
                exc = TimeoutError(f"External case deadline of {EXTERNAL_CASE_DEADLINE_SECONDS}s exceeded; no rerun")
                exc.pipeline_context = getattr(original, "pipeline_context", {})
            elif not isinstance(exc, Exception):
                raise
            traceback.print_exc()
            record = ns["error_record"](case, exc)
            if ns["EVALUATION_METHOD"].startswith("examples"):
                record.update(ns["loto_record_fields"]())
            record["fatal_service_error"] = type(exc).__name__ in {"AuthenticationError", "PermissionDeniedError"} or "insufficient_quota" in str(exc)
        finally:
            deadline.cancel()
        if "react_protocol_record_fields" in ns:
            record.update(ns["react_protocol_record_fields"]())
        record = ns["finalize_record"](record)
        record.update(seconds=round(time.monotonic() - started, 3),
                      classification_source="final_full_cache" if ns["EVALUATION_METHOD"] in {"rag_only", "few_shot_only"} else "computed",
                      external_case_deadline_seconds=EXTERNAL_CASE_DEADLINE_SECONDS,
                      external_deadline_exceeded=timed_out.is_set(),
                      base_notebook_sha256=item["base_sha"], notebook_source_sha256=item["notebook_sha"],
                      cache_fingerprint=ns["case_fingerprint"](case, source_fingerprint=item["source_fingerprint"]),
                      evaluated_utc=datetime.now(timezone.utc).isoformat())
        # Each worker writes only to its own case folder. Parent assembles shared CSVs.
        ns["save_record"](folder / "results.csv", record)
        (folder / "result.json").write_text(json.dumps(record, indent=2, default=str))
        (folder / "attempt.json").write_text(json.dumps({"state": "finished", "case_id": case["problem_id"]}))
    return item["group"], record


def summarize(ns, groups, root):
    import pandas as pd
    rows, details = [], []
    for group, cases in groups.items():
        path = root / group / "results.csv"
        records = ns["load_records"](path)
        by_id = {r["problem_id"]: r for r in records}
        denomin = len(cases)
        rows.append({"dataset": group, "expected": denomin, "recorded": len(records),
                     "classification_correct": sum(r.get("classification_correct") is True for r in records),
                     "solved": sum(r.get("final_ok") is True for r in records),
                     "objective_match": sum(r.get("solution_correct") is True for r in records),
                     "objective_accuracy": sum(r.get("solution_correct") is True for r in records) / denomin})
        for case in cases.to_dict("records"):
            record = by_id.get(case["problem_id"], {})
            details.append({"dataset": group, "problem_id": case["problem_id"],
                            "true_label": case["true_label"], "recorded": bool(record),
                            **{k: record.get(k) for k in ("predicted_label", "assigned_route", "classification_correct",
                               "final_ok", "solution_correct", "final_objective", "label_objective", "pipeline_stage",
                               "execution_error_type", "execution_error", "csvqa_status", "repair_count", "retry_count",
                               "fallback_count", "api_retry_count", "api_retry_events", "cache_source",
                               "protocol_retry_count", "protocol_retry_events",
                               "disabled_workflow_route", "route_allowed")}})
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(details).to_csv(root / "case_summary.csv", index=False)
    full = [r for r in details if r["dataset"] == "automatic"]
    if full:
        table = pd.DataFrame(full).groupby("true_label").agg(total=("problem_id", "size"),
            classification_correct=("classification_correct", lambda s: s.eq(True).sum()),
            solved=("final_ok", lambda s: s.eq(True).sum()), objective_match=("solution_correct", lambda s: s.eq(True).sum()))
        table.to_csv(root / "accuracy_by_class.csv")
    (root / "progress.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=NAMES, default="full")
    ap.add_argument("--version", default="v5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect-only", action="store_true", help="Collect a stopped frozen version without any inference")
    args = ap.parse_args()
    root = OUT / (args.method + "_" + args.version)
    root.mkdir(exist_ok=True)
    frozen = root / "frozen_notebook.ipynb"
    original = ROOT / NAMES[args.method]
    if not frozen.exists():
        frozen.write_bytes(original.read_bytes())
    if not args.collect_only and sha(frozen) != sha(original) and not (
            (args.method == "full" and same_full_definitions(frozen, original)) or
            recorded_query_only_alias_correction(frozen, original, args.method)):
        raise ValueError("Notebook changed: select a new version; never overwrite a frozen experiment")
    with contextlib.redirect_stdout(io.StringIO()):
        ns = namespace(frozen, args.method)
    if args.method == "full":
        groups = {"automatic": ns["load_benchmark"](),
                  "variants": ns["load_variants_for_baseline"](ROOT / "benchmark_dataset/questions.csv")}
        groups.update({"columns/" + k: v for k, v in ns["load_redundant_sheets_for_baseline"](
            ROOT / "redundancy_complete/redundant_instances.xlsx",
            [f"{p}-S{s}" for p in ("50pct", "100pct", "200pct") for s in (1, 2, 3)]).items()})
    elif args.method.startswith("examples"):
        frame, folds = ns["loto_preflight"]()
        groups = {"automatic": frame}
        folds.to_csv(root / "preflight_folds.csv", index=False)
    else:
        groups = {"automatic": ns["attach_cached_classifications"](ns["load_benchmark"]())}
    if recorded_query_only_alias_correction(frozen, original, args.method):
        assert all(frame["dataset_address"].astype(str).str.strip().ne("").all() for frame in groups.values()), "The scoped alias correction permits reuse only for CSV cases"
    inputs = {}
    for path in (ROOT / "Large_Scale_Or_Files").rglob("*.csv"):
        inputs[str(path)] = sha(path)
    for path in [ns["BENCHMARK_PATH"], ROOT / "benchmark_dataset/questions.csv", ROOT / "redundancy_complete/redundant_instances.xlsx"]:
        inputs[str(path)] = sha(path)
    for frame in groups.values():
        assert frame["problem_id"].is_unique
        assert frame["label_objective"].notna().all()
        for address in frame["dataset_address"]:
            for raw in address.splitlines():
                inputs[raw] = sha(raw)
    source = ns["_source_fingerprint"]()
    full_frozen = OUT / ("full_" + args.version) / "frozen_notebook.ipynb"
    if args.method in {"rag_only", "few_shot_only"}:
        full_frozen = Path(ns["CLASSIFICATION_RESULTS_PATH"]).parents[1] / "frozen_notebook.ipynb"
    if full_frozen.exists():
        assert same_full_definitions(full_frozen, ROOT / NAMES["full"]) or args.collect_only, "Full pipeline changed; choose a new baseline"
    base_sha = sha(full_frozen if full_frozen.exists() else ROOT / NAMES["full"])
    manifest = {"method": args.method, "version": args.version, "notebook_sha256": sha(frozen),
                "base_sha256": base_sha, "source_fingerprint": source, "inputs": inputs,
                "sizes": {k: len(v) for k, v in groups.items()}, "model": ns["MODEL_SNAPSHOT"],
                "embedding_model": ns["EMBEDDING_MODEL"], "temperature": 0, "top_p": 1,
                "rel_tol": 1e-4, "abs_tol": 1e-4, "api_max_retries": ns.get("API_MAX_RETRIES", 0),
                "model_repair": False, "code_repair": False, "truncation_retry": False,
                "react_protocol_restarts": "invoke_react_protocol" in ns,
                "react_protocol_restart_reasons": (["missing_required_csvqa", "react_output_format", "incomplete_final_answer"]
                    if "invoke_react_protocol" in ns else []),
                "csvqa_minimum_calls": 1 if "invoke_react_protocol" in ns else "exactly one",
                "multiple_csvqa_calls_allowed": "invoke_react_protocol" in ns,
                "parsing_correction": False, "gurobi_threads": 2, "time_limit": "unchanged",
                "python": sys.version, "executable": sys.executable,
                "csvqa_modes": ns["CSVQA_MODE_BY_ROUTE"], "workers": args.workers,
                "canonical_csv_formulation_protocol": "ReAct",
                "few_shot_observation_protocol": "ReAct without tools"}
    mp = root / "manifest.json"
    if mp.exists():
        previous = json.loads(mp.read_text())
        if args.collect_only:
            manifest = previous
            base_sha = previous["base_sha256"]
        else:
            assert previous == manifest, "Inputs/settings changed; use a new version directory"
    else:
        mp.write_text(json.dumps(manifest, indent=2))
    if not args.collect_only:
        control = {"external_case_deadline_seconds": EXTERNAL_CASE_DEADLINE_SECONDS,
                   "action": "SIGINT; preserve interrupted or unavailable results as failures; no rerun",
                   "gurobi_time_limit": "unchanged",
                   "scope": "Every full, ablation and LOTO case"}
        control_path = root / "runtime_control.json"
        if control_path.exists():
            assert json.loads(control_path.read_text()) == control, "Execution controls changed; select a new version"
        else:
            control_path.write_text(json.dumps(control, indent=2))
        harness = root / "frozen_evaluation_runner.py"
        if not harness.exists():
            harness.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"preflight": "PASS", "method": args.method, "sizes": manifest["sizes"],
                      "total": sum(manifest["sizes"].values()), "output": str(root)}, indent=2), flush=True)
    if not args.run and not args.collect_only:
        return
    if args.run:
        ns["require_api_key"]()
    tasks = []
    for group, frame in groups.items():
        for case in frame.to_dict("records"):
            # Drop unused gold model text before it can enter an inference worker.
            case = {k: v for k, v in case.items() if k not in {"Label-model", "Label-objective", "Problem Type"}}
            folder = root / group / "attempts" / ns["quote"](case["problem_id"], safe="")
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "result.json"
            if result_path.exists():
                row = json.loads(result_path.read_text())
                ns["save_record"](root / group / "results.csv", row)
                continue
            if (folder / "attempt.json").exists():
                # An interrupted attempt is a failure, never a license to rerun the case.
                row = ns["error_record"](case, RuntimeError("Interrupted previous attempt; no rerun permitted"))
                row.update(pipeline_stage="interrupted", base_notebook_sha256=base_sha)
                row = ns["finalize_record"](row)
                ns["save_record"](root / group / "results.csv", row)
                result_path.write_text(json.dumps(row, indent=2, default=str))
                continue
            if args.collect_only:
                continue
            tasks.append({"case": case, "folder": str(folder), "group": group,
                          "base_sha": base_sha, "notebook_sha": sha(frozen), "source_fingerprint": source})
    # Bounded dispatch: a service configuration failure stops new submissions.
    total = len(tasks)
    if args.collect_only:
        print(json.dumps(summarize(ns, groups, root), indent=2))
        (root / "run_status.json").write_text(json.dumps({"complete": False, "diagnostic_version_stopped": True}, indent=2))
        return
    completed = 0
    iterator = iter(tasks)
    fatal = False
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    (root / "coordinator_pid.txt").write_text(str(os.getpid()))
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(frozen), args.method)) as pool:
        pending = {}
        def submit(item):
            Path(item["folder"], "attempt.json").write_text(json.dumps({"state": "started", "case_id": item["case"]["problem_id"]}))
            pending[pool.submit(worker, item)] = item
        for _ in range(min(args.workers, total)):
            submit(next(iterator))
        while pending:
            future = next(as_completed(pending))
            item = pending.pop(future)
            group, row = future.result()
            ns["save_record"](root / group / "results.csv", row)
            completed += 1
            print(f"[{completed}/{total}] {group} {row['problem_id']} class={row.get('classification_correct')} solved={row['final_ok']} match={row.get('solution_correct')} {row.get('execution_error_type') or ''}", flush=True)
            summarize(ns, groups, root)
            fatal = fatal or bool(row.get("fatal_service_error"))
            stop = stop or (root / "STOP").exists()
            if not fatal and not stop:
                next_item = next(iterator, None)
                if next_item:
                    submit(next_item)
    print(json.dumps(summarize(ns, groups, root), indent=2), flush=True)
    (root / "run_status.json").write_text(json.dumps({"complete": not fatal and not stop and completed == total,
        "new_attempts": completed, "fatal_service_error": fatal, "diagnostic_version_stopped": stop}, indent=2))


if __name__ == "__main__":
    main()
