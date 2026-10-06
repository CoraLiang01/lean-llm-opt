"""Pair the original GPT-4.1 V2 notebooks with RA-structured V3 copies.

Default mode performs read-only preflight. Pass --run to make model calls.
The sample is the intersection of cases historically assigned to RA in both
GPT-4.1 LOTO conditions, then evaluated in all three original/V3 pairs.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/loto_cross_model_audit_20261004/gpt41_audit"
OUT = ROOT / "outputs/gpt41_ra_structured_v3_validation"
PAIRS = {
    "full": (
        "LEAN_LLM_OPT_4.1_Large-scale_Model_Interface_V2.ipynb",
        "LEAN_LLM_OPT_4.1_Large-scale_RA_Structured_V3.ipynb",
        False,
    ),
    "examples_only": (
        "LOTO_Examples_Only_GPT4.1_Large-scale_Model_Interface_V2.ipynb",
        "LOTO_Examples_Only_GPT4.1_Large-scale_RA_Structured_V3.ipynb",
        True,
    ),
    "examples_and_route": (
        "LOTO_Examples_And_Route_GPT4.1_Large-scale_Model_Interface_V2.ipynb",
        "LOTO_Examples_And_Route_GPT4.1_Large-scale_RA_Structured_V3.ipynb",
        True,
    ),
}


def sample_ids():
    only = json.loads((AUDIT / "examples_only_cases.json").read_text())
    both = json.loads((AUDIT / "examples_and_route_cases.json").read_text())
    only_ra = {row["problem_id"]: row for row in only if row.get("assigned_route") == "RA"}
    both_ra = {row["problem_id"]: row for row in both if row.get("assigned_route") == "RA"}
    ids = sorted(only_ra.keys() & both_ra.keys())
    if not ids:
        raise ValueError("No shared historical RA-routed cases were found")
    return ids, only_ra, both_ra


def load_namespace(notebook_path, loto):
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": "__ra_structured_comparison__"}
    last_cell = 41 if loto else 35
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code" and index <= last_cell:
                source = "".join(cell["source"])
                compile(source, f"{notebook_path.name}:cell{index}", "exec")
                exec(source, namespace)
    # Shared persistence-only compatibility fix: pandas may serialize nullable
    # integer metadata as "1.0". All six paired runs use this identical reader.
    def read_integral_csv_value(value):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"Expected an integral numeric field, got {value!r}")
        return int(number)

    for field in ("solver_status", "model_count", "classification_repair_count",
                  "code_repair_count", "service_retry_count"):
        namespace["CSV_VALUE_READERS"][field] = read_integral_csv_value
    return namespace


def prepare(path, loto, ids):
    namespace = load_namespace(path, loto)
    frame = namespace["load_benchmark"]()
    selected = frame.loc[frame["problem_id"].isin(ids)].copy()
    if len(selected) != len(ids):
        raise ValueError(f"{path.name}: selected {len(selected)} of {len(ids)} requested cases")
    if loto:
        namespace["loto_preflight"]()
    namespace["_source_fingerprint"]()
    return namespace, frame, selected


def preflight_inputs(namespace, selected):
    reference = ROOT / namespace["RAG_EXAMPLES_ALL_PATH"]
    payload = reference.read_bytes()
    if payload.startswith(b"%TSD"):
        raise PermissionError(f"Protected reference CSV: {reference}")
    for address in selected["dataset_address"]:
        for raw in str(address).splitlines():
            if raw.strip():
                path = Path(raw.strip()).expanduser()
                path = path if path.is_absolute() else ROOT / path
                if path.read_bytes().startswith(b"%TSD"):
                    raise PermissionError(f"Protected benchmark CSV: {path}")


def score(row):
    solved = row.get("final_ok") is True or row.get("final_ok") == "True"
    correct = row.get("solution_correct") is True or row.get("solution_correct") == "True"
    return solved, correct


def run_case_pair(batch, ids, run):
    reports, pending = [], []
    for variant, (before_name, after_name, loto) in PAIRS.items():
        for condition, notebook_name in (("before", before_name), ("ra_structured", after_name)):
            path = ROOT / notebook_name
            entry = {"variant": variant, "condition": condition, "notebook": notebook_name,
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "requested_ids": ids, "loto": loto, "end_to_end_run": False}
            try:
                namespace, frame, selected = prepare(path, loto, ids)
                preflight_inputs(namespace, selected)
                namespace["RESULTS_DIR"] = OUT / "results" / variant / condition / batch
                entry.update(preflight="PASS", output_dir=str(namespace["RESULTS_DIR"]))
                pending.append((entry, namespace, frame, selected, loto))
            except Exception as exc:
                entry.update(preflight="BLOCKED", error=f"{type(exc).__name__}: {exc}")
            reports.append(entry)

    if run and len(pending) == len(PAIRS) * 2:
        for entry, namespace, frame, selected, loto in pending:
            namespace["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)
            log_path = namespace["RESULTS_DIR"] / "run.log"
            entry["run_log"] = str(log_path)
            indices = frame.index[frame["problem_id"].isin(ids)].tolist()
            existing = read_results(entry, loto)
            case_indices = dict(zip(frame.loc[indices, "problem_id"], indices))
            errors = []
            for case_id in ids:
                if case_id in existing:
                    continue
                index = case_indices[case_id]
                try:
                    with log_path.open("a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        if loto:
                            namespace["run_loto"](rows=[index], reuse_completed=False)
                        else:
                            namespace["run_test"](
                                frame.loc[[index]],
                                output_csv=namespace["RESULTS_DIR"] / "results.csv",
                                reuse_completed=False,
                            )
                except Exception as exc:
                    errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
                    # Preserve this case even if a notebook-level report/aggregation
                    # raises after the normal per-case handler has been bypassed.
                    existing = read_results(entry, loto)
                    if case_id not in existing:
                        case = frame.loc[[index]].iloc[0].to_dict()
                        try:
                            if loto:
                                namespace["set_loto_fold"](case["true_label"])
                            record = namespace["error_record"](case, exc)
                            if loto:
                                record.update(**namespace["loto_record_fields"]())
                            record["pipeline_stage"] = "runner"
                            record["runner_error"] = True
                            record = namespace["finalize_record"](record)
                            record["cache_fingerprint"] = namespace["case_fingerprint"](
                                case, source_fingerprint=namespace["_source_fingerprint"]())
                            output_csv = (namespace["RESULTS_DIR"] / str(case["true_label"]) / "results.csv"
                                          if loto else namespace["RESULTS_DIR"] / "results.csv")
                            namespace["save_record"](output_csv, record)
                        except Exception as save_exc:
                            errors.append(f"{case_id}: fallback-save failed: {type(save_exc).__name__}: {save_exc}")
                # A LOTO report may fail after the per-case CSV was already saved.
                existing = read_results(entry, loto)
            entry.update(evaluated=len(existing), solved=sum(score(row)[0] for row in existing.values()),
                         objective_match=sum(score(row)[1] for row in existing.values()),
                         end_to_end_run=len(existing) == len(ids))
            if errors:
                entry["case_errors"] = errors

    return reports


def read_results(entry, loto):
    base = Path(entry["output_dir"])
    paths = list(base.glob("*/results.csv")) if loto else [base / "results.csv"]
    records = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row.get("problem_id")
                if key in records:
                    raise ValueError(f"Duplicate result for {entry['variant']}/{entry['condition']}/{key}")
                records[key] = row
    return records


def compare(reports, ids, only_ra, both_ra):
    pairs, summary = [], []
    for variant in PAIRS:
        found = {}
        for condition in ("before", "ra_structured"):
            entry = next((item for item in reports if item["variant"] == variant and item["condition"] == condition), None)
            found[condition] = read_results(entry, entry["loto"]) if entry and entry.get("preflight") == "PASS" else {}
        common = 0
        ra_before = ra_after = 0
        variant_pairs = []
        for case_id in ids:
            old, new = found["before"].get(case_id), found["ra_structured"].get(case_id)
            old_solved, old_match = score(old or {})
            new_solved, new_match = score(new or {})
            before_route = (old or {}).get("executed_route") or (old or {}).get("assigned_route") or (old or {}).get("selected_route")
            after_route = (new or {}).get("executed_route") or (new or {}).get("assigned_route") or (new or {}).get("selected_route")
            if old and new:
                common += 1
            ra_before += before_route == "RA"
            ra_after += after_route == "RA"
            paired = {"variant": variant, "problem_id": case_id,
                          "true_label": (old or new or {}).get("true_label"),
                          "historical_examples_only_route": only_ra.get(case_id, {}).get("assigned_route"),
                          "historical_examples_and_route_route": both_ra.get(case_id, {}).get("assigned_route"),
                          "before_route": before_route, "before_solved": old_solved if old else None,
                          "before_match": old_match if old else None,
                          "structured_route": after_route, "structured_solved": new_solved if new else None,
                          "structured_match": new_match if new else None}
            pairs.append(paired)
            variant_pairs.append(paired)
        ra_pairs = [row for row in variant_pairs
                    if row["before_route"] == "RA" and row["structured_route"] == "RA"
                    and row["before_solved"] is not None and row["structured_solved"] is not None]
        summary.append({"variant": variant, "sample_cases": len(ids), "paired_recorded": common,
                        "before_solved_all": sum(row["before_solved"] is True for row in variant_pairs if row["before_solved"] is not None),
                        "before_match_all": sum(row["before_match"] is True for row in variant_pairs if row["before_match"] is not None),
                        "structured_solved_all": sum(row["structured_solved"] is True for row in variant_pairs if row["structured_solved"] is not None),
                        "structured_match_all": sum(row["structured_match"] is True for row in variant_pairs if row["structured_match"] is not None),
                        "paired_RA_route_records": len(ra_pairs),
                        "before_RA_solved": sum(row["before_solved"] is True for row in ra_pairs),
                        "before_RA_objective_match": sum(row["before_match"] is True for row in ra_pairs),
                        "structured_RA_solved": sum(row["structured_solved"] is True for row in ra_pairs),
                        "structured_RA_objective_match": sum(row["structured_match"] is True for row in ra_pairs),
                        "before_routed_to_RA": ra_before, "structured_routed_to_RA": ra_after})
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"sample_ids": ids, "summary": summary, "paired_cases": pairs,
              "note": "Report RA-route counts separately; outcomes from cases routed elsewhere are not evidence about the RA formulation path.",
              "shared_persistence_fix": "Integral result metadata is parsed from both integer strings and pandas float-formatted integers (e.g. 1 and 1.0); applied identically to all six runs and does not alter inference."}
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ["# GPT-4.1 RA structured-data transfer: paired sample", "",
             "The sample is the intersection of cases historically assigned to RA in both LOTO conditions. Each V2 notebook is paired with its RA-structured V3 copy; the comparison reruns the same case IDs.", "",
             "| Condition | Paired records | RA in both runs | Before RA solved | Before RA objective match | Structured RA solved | Structured RA objective match | Before RA route | Structured RA route |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['paired_recorded']}/{len(ids)} | {row['paired_RA_route_records']} | {row['before_RA_solved']}/{row['paired_RA_route_records']} | {row['before_RA_objective_match']}/{row['paired_RA_route_records']} | {row['structured_RA_solved']}/{row['paired_RA_route_records']} | {row['structured_RA_objective_match']}/{row['paired_RA_route_records']} | {row['before_routed_to_RA']}/{len(ids)} | {row['structured_routed_to_RA']}/{len(ids)} |")
    lines += ["", "Overall sample totals across every route (descriptive only):", "",
              "| Condition | Before solved | Before objective match | Structured solved | Structured objective match |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['before_solved_all']}/{row['paired_recorded']} | {row['before_match_all']}/{row['paired_recorded']} | {row['structured_solved_all']}/{row['paired_recorded']} | {row['structured_match_all']}/{row['paired_recorded']} |")
    lines += ["", "Only cases routed to RA in both paired runs contribute to the RA-specific comparison. The all-route totals also reflect route changes.", "",
              "| Condition | Problem ID | Gold type | Historical Examples Only route | Historical Examples and Route route | Before route | Before solved | Before match | Structured route | Structured solved | Structured match |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in pairs:
        lines.append("| " + " | ".join(str(row.get(k) if row.get(k) is not None else "—") for k in (
            "variant", "problem_id", "true_label", "historical_examples_only_route",
            "historical_examples_and_route_route", "before_route", "before_solved", "before_match",
            "structured_route", "structured_solved", "structured_match")) + " |")
    (OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    return result


def main(run, resume_batch=None):
    ids, only_ra, both_ra = sample_ids()
    batch = resume_batch or "ra_sample_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    reports = run_case_pair(batch, ids, run)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_status.json").write_text(json.dumps(
        {"batch": batch, "sample_ids": ids, "reports": reports}, ensure_ascii=False, indent=2))
    if run and all(row.get("end_to_end_run") for row in reports):
        compare(reports, ids, only_ra, both_ra)
    print(json.dumps({"batch": batch, "sample_ids": ids, "reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make model calls after preflight")
    parser.add_argument("--resume-batch", help="continue an existing ra_sample_YYYYMMDD_HHMMSS batch")
    args = parser.parse_args()
    main(args.run, args.resume_batch)
