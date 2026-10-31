#!/usr/bin/env python3
"""Run the V2 Large-Scale-OR notebook on a configured benchmark.

The supplied V2 notebook is intentionally kept as the source of pipeline
logic.  This driver executes its setup cells, stops before its experiment
cells (which would otherwise run 101 automatic plus 606 forced-route cases),
and adds the repository's existing per-instance accounting around the
automatic run.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import leanopt_exp as lx
from run_notebook import split_at_first_call


HERE = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK = HERE / "LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"
DEFAULT_DATA = HERE / "Test_Dataset/Large-scale-or/Large-scale-or-101.csv"
DEFAULT_MARKER = HERE / "outputs/large_scale_or_test_final/latest_run_dir.txt"
DATA_PATH_RE = re.compile(
    r'''["']((?:Test_Dataset|Large_Scale_Or_Files)/[^"']+)["']'''
)


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _parse_rows(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    rows: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            if end < start:
                raise SystemExit(f"invalid row range: {part}")
            rows.extend(range(start, end + 1))
        else:
            rows.append(int(part))
    if len(rows) != len(set(rows)):
        raise SystemExit("--rows contains duplicates")
    return rows


def _find_resume_run(out_dir: Path, method: str, profile: str,
                     dataset: str, repeat: int) -> Path | None:
    if not out_dir.exists():
        return None
    matches: list[Path] = []
    for directory in out_dir.iterdir():
        config_path = directory / "config.json"
        instances_path = directory / "instances.jsonl"
        if not config_path.is_file() or not instances_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identity = (
            config.get("method"), config.get("model_profile"),
            config.get("dataset"), int(config.get("repeat_index", 0)),
        )
        if identity == (method, profile, dataset, repeat):
            matches.append(directory)
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _execute_setup(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup, stop = split_at_first_call(notebook)
    if stop is None:
        raise SystemExit("could not find the V2 experiment cell that calls run_test")

    namespace: dict[str, Any] = {"__name__": "__main__"}
    os.chdir(HERE)
    os.environ.setdefault("LEAN_LLM_OPT_ROOT", str(HERE))
    for index, source in setup:
        stripped = source.strip()
        if stripped.startswith(("!", "%")):
            continue
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            print(
                f"setup cell {index} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            raise
    if not callable(namespace.get("run_test")):
        raise SystemExit("V2 setup completed, but run_test was not defined")
    namespace["NOTEBOOK_PATH"] = notebook_path
    return namespace


def _install_accounting(namespace: dict[str, Any], cfg: lx.ExpConfig,
                        log: lx.RunLogger) -> None:
    """Attach the existing tracker without changing V2 pipeline decisions."""
    tracker = lx.TRACKER.bind(cfg)

    chat_openai = namespace["ChatOpenAI"]
    reject_truncated = namespace["RejectTruncatedOutput"]

    def tracked_make_llm(*, timeout=180):
        return chat_openai(
            model_name=namespace["MODEL_SNAPSHOT"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.0,
            top_p=1,
            n=1,
            max_retries=0,
            timeout=timeout,
            callbacks=[tracker, reject_truncated()],
        )

    namespace["make_llm"] = tracked_make_llm

    # V2 has one direct OpenAIEmbeddings construction in addition to its
    # make_embeddings() helper. Replacing the constructor in the notebook
    # namespace covers both while retaining the notebook's arguments.
    raw_embeddings = namespace["OpenAIEmbeddings"]

    def tracked_embeddings(*args, **kwargs):
        inner = raw_embeddings(*args, **kwargs)
        return lx.CountingEmbeddings(inner, cfg, cfg.embedding_price_per_1m)

    namespace["OpenAIEmbeddings"] = tracked_embeddings

    # Agent-level callbacks are what expose actual Tool executions. LLM-level
    # callbacks alone count tokens but cannot see FileQA/CSVQA/ORLM_QA calls.
    raw_initialize_agent = namespace["initialize_agent"]

    def tracked_initialize_agent(*args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        if tracker not in callbacks:
            callbacks.append(tracker)
        kwargs["callbacks"] = callbacks
        return raw_initialize_agent(*args, **kwargs)

    namespace["initialize_agent"] = tracked_initialize_agent

    def stage_wrap(name: str, stage_name: str) -> None:
        original = namespace[name]

        def wrapped(*args, **kwargs):
            with lx.stage(stage_name):
                return original(*args, **kwargs)

        wrapped.__name__ = getattr(original, "__name__", name)
        wrapped.__doc__ = getattr(original, "__doc__", None)
        namespace[name] = wrapped

    for function_name, stage_name in (
        ("invoke_classifier", "classification"),
        ("invoke_formulation", "modeling"),
        ("get_code", "codegen"),
        ("get_csv_code", "codegen"),
        ("execute_code", "solver"),
    ):
        stage_wrap(function_name, stage_name)

    finalize_record = namespace["finalize_record"]
    raw_execute_case = namespace["execute_pipeline_case"]

    def attach_record(rec: lx.InstanceRecord, result: dict[str, Any]) -> None:
        final = finalize_record(result)
        solution_correct = final.get("solution_correct")
        rec.set(
            pred_type=final.get("predicted_label"),
            model_output=final.get("generated_model"),
            code_output=final.get("solve_code"),
            dataset_address=final.get("dataset_address"),
            gold_objective=final.get("label_objective"),
            objective_value=final.get("final_objective"),
            classification_correct=final.get("classification_correct"),
            correct_optimal=(int(solution_correct)
                             if solution_correct is not None else None),
            pipeline_record_status=final.get("record_status"),
            assigned_route=final.get("assigned_route"),
        )

    def tracked_execute_case(case, *, forced_route=None):
        instance_id = case.get("__instance_id", case.get("problem_id"))
        with log.instance(
            instance_id=instance_id,
            query=str(case.get("Query", "")),
            gold_type=case.get("true_label"),
            gold_label_raw=case.get("__gold_label_raw"),
        ) as rec:
            try:
                result = raw_execute_case(case, forced_route=forced_route)
            except BaseException as exc:  # preserve partial code/model output
                partial = getattr(exc, "pipeline_context", None)
                if isinstance(partial, dict):
                    attach_record(rec, partial)
                raise
            attach_record(rec, result)
            return result

    namespace["execute_pipeline_case"] = tracked_execute_case

    # A spending-cap exception originates when the instance context closes.
    # V2 normally converts every Exception into a row and continues, which
    # would neutralize the cap. Let this one exception stop the run.
    raw_error_record = namespace["error_record"]

    def guarded_error_record(case, exc, forced_route=None):
        if isinstance(exc, lx.BudgetExceeded):
            raise exc
        return raw_error_record(case, exc, forced_route)

    namespace["error_record"] = guarded_error_record


def _write_cumulative_summary(run_dir: Path) -> None:
    """Make summary.json cumulative after a resumed run."""
    instance_path = run_dir / "instances.jsonl"
    if not instance_path.exists():
        return
    latest: dict[Any, dict[str, Any]] = {}
    for line in instance_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        latest[record.get("instance_id")] = record
    records = list(latest.values())
    if not records:
        return
    keys = (
        "n_llm_calls", "n_llm_calls_failed", "n_tool_calls",
        "n_tool_calls_failed", "n_retriever_calls", "prompt_tokens",
        "cached_prompt_tokens", "completion_tokens", "reasoning_tokens",
        "total_tokens", "cost_usd", "llm_latency_s", "tool_latency_s",
        "wall_s",
    )
    summary = {
        "run_id": records[0].get("run_id"),
        "method": records[0].get("method"),
        "dataset": records[0].get("dataset"),
        "model_profile": records[0].get("model_profile"),
        "n_instances": len(records),
        "n_failed": sum(record.get("status") != "ok" for record in records),
    }
    for key in keys:
        total = sum((record.get("summary") or {}).get(key) or 0 for record in records)
        summary[f"total_{key}"] = round(total, 6)
        summary[f"mean_{key}"] = round(total / len(records), 4)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rows", help="row labels, e.g. 0,2-4,100")
    parser.add_argument("--limit", type=int, help="first N selected rows")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--profile", default="gpt-4.1")
    parser.add_argument("--method", default="LEAN-LLM-OPT-V2")
    parser.add_argument("--dataset", default="Large-Scale-OR")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None, metavar="RUN_ID",
        help="continue RUN_ID; bare --resume selects the newest matching run",
    )
    parser.add_argument("--run-dir-marker", type=Path, default=DEFAULT_MARKER)
    parser.add_argument("--list", action="store_true",
                        help="validate the benchmark plan without API calls")
    parser.add_argument(
        "--preflight", action="store_true",
        help="check the V2 run dependencies and data; makes no API calls",
    )
    args = parser.parse_args()

    notebook_path = args.notebook.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    if not notebook_path.is_file():
        raise SystemExit(f"notebook not found: {notebook_path}")
    if not data_path.is_file():
        raise SystemExit(f"dataset not found: {data_path}")
    row_count = _csv_row_count(data_path)
    selected_rows = _parse_rows(args.rows)
    planned = len(selected_rows) if selected_rows is not None else row_count
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        planned = min(planned, args.limit)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup, stop = split_at_first_call(notebook)
    print(f"notebook : {notebook_path.name}")
    print(f"dataset  : {data_path} ({row_count} rows)")
    print(f"setup    : {len(setup)} code cells; experiment cells start at {stop}")
    print(f"plan     : {planned} automatic case(s); forced-route cells are skipped")
    if args.dataset == "Large-Scale-OR" and row_count != 101 and selected_rows is None:
        raise SystemExit(f"expected the 101-case dataset, found {row_count} rows")
    if args.list:
        print("\n--list: no notebook code executed and no API calls made.")
        return 0

    namespace = _execute_setup(notebook_path)
    cfg = lx.load_config(
        HERE / "exp_config.yaml",
        model_profile=args.profile,
        method=args.method,
        dataset=args.dataset,
        repeat_index=args.repeat,
    )
    if cfg.dataset_path and (HERE / cfg.dataset_path).resolve() != data_path:
        raise SystemExit(
            f"--data does not match the configured path for {args.dataset}: "
            f"{cfg.dataset_path}"
        )

    frame = namespace["read_csv_compat"](data_path)
    frame["__instance_id"] = frame.index
    if "Label" in frame:
        frame["__gold_label_raw"] = frame["Label"]
        # NL4OPT contains 16 cases labelled 'No Best Solution'. They still
        # belong in the resource benchmark, but have no numeric optimum to
        # compare. Preserve the source label and leave objective match blank.
        no_best = frame["Label"].astype(str).str.strip().eq("No Best Solution")
        if no_best.any():
            frame.loc[no_best, "Label"] = float("nan")
            print(f"labels   : {int(no_best.sum())} case(s) have no numeric optimum; "
                  "resource metrics will still be recorded")
    frame = namespace["prepare_cases"](frame)
    if selected_rows is not None:
        missing_rows = [row for row in selected_rows if row not in frame.index]
        if missing_rows:
            raise SystemExit(f"row labels not in the dataset: {missing_rows}")
        frame = frame.loc[selected_rows]
    if args.limit is not None:
        frame = frame.head(args.limit)

    if args.preflight:
        notebook_text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        missing = sorted(
            relative for relative in set(DATA_PATH_RE.findall(notebook_text))
            if not (HERE / relative).exists()
        )
        if missing:
            raise SystemExit(
                "V2 notebook references missing input file(s):\n  "
                + "\n  ".join(missing)
            )
        lx.ensure_api_keys(cfg, interactive=False)
        required = (
            "run_test", "execute_pipeline_case", "finalize_record",
            "report_results", "load_records",
        )
        absent = [name for name in required if not callable(namespace.get(name))]
        if absent:
            raise SystemExit(f"V2 setup is missing callable(s): {absent}")
        print("\nV2 preflight passed:")
        print("  - notebook setup cells import and compile")
        print("  - all notebook input paths exist")
        print(f"  - {args.dataset}: {row_count} rows, {len(frame)} selected and parsed")
        print("  - the gpt-4.1 configuration and OPENAI_API_KEY are available")
        print("  - no API calls were made")
        return 0
    lx.ensure_api_keys(cfg, interactive=True)

    out_root = Path(cfg.out_dir)
    if not out_root.is_absolute():
        out_root = HERE / out_root
    if args.resume:
        resume_dir = (
            _find_resume_run(out_root, args.method, args.profile,
                             args.dataset, args.repeat)
            if args.resume == "auto" else out_root / args.resume
        )
        if resume_dir is None or not resume_dir.is_dir():
            raise SystemExit("--resume found no matching run directory")
        cfg.run_id = resume_dir.name
        print(f"resume   : {resume_dir}")

    log = lx.RunLogger(cfg)
    _install_accounting(namespace, cfg, log)

    done = log.done_instance_ids()
    if done:
        before = len(frame)
        frame = frame.loc[~frame.index.isin(done)]
        print(f"resume   : {before - len(frame)} completed case(s) skipped; "
              f"{len(frame)} left")

    run_dir = out_root / cfg.run_id
    args.run_dir_marker = args.run_dir_marker.expanduser().resolve()
    args.run_dir_marker.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir_marker.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    if frame.empty:
        print(f"nothing left to run; existing results are in {run_dir}")
        return 0

    pipeline_csv = log.dir / "pipeline_results.csv"
    print(f"running  : {len(frame)} case(s)")
    print(f"log      : {log.dir.resolve()}\n")

    failure: BaseException | None = None
    try:
        namespace["run_test"](
            frame,
            output_csv=pipeline_csv,
            reuse_completed=False,
            continue_on_error=True,
        )
        all_records = namespace["load_records"](pipeline_csv)
        namespace["report_results"](all_records, log.dir / "v2_accuracy")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            if log._started:  # close files and write an initial summary
                log.close()
            _write_cumulative_summary(run_dir)
        except Exception as close_exc:
            if failure is None:
                raise
            print(f"could not finalize summary: {close_exc}", file=sys.stderr)

    print(f"\nrun_dir  : {run_dir.resolve()}")
    print("next     : score generated models, then aggregate the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
