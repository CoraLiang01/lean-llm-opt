"""Report every attempt with fixed denominators and explicit uncertainty."""
from collections import Counter
import argparse
import contextlib
import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/react_revision_20261006"
CLASSES = ["AP", "FLP", "NRM", "RA", "TP", "Others", "Mixture"]
SIZES = {"AP": 5, "FLP": 14, "NRM": 25, "RA": 22, "TP": 9, "Others": 8, "Mixture": 18}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def attempts(folder):
    return [(p, json.loads(p.read_text())) for p in folder.glob("**/attempts/*/result.json")]


def failure_stage(path, r):
    if r.get("solution_correct"):
        return "none", "objective_matched", "confirmed"
    if r.get("external_deadline_exceeded") or path.with_name("external_interrupt.json").exists():
        return "execution", "external_case_deadline_exceeded", "confirmed; no optimal result within the common execution deadline"
    raw = r.get("pipeline_stage", "setup")
    error = r.get("execution_error", "") or ""
    log = path.with_name("run.log").read_text(errors="replace") if path.with_name("run.log").exists() else ""
    if "context_length_exceeded" in error or "maximum context length" in error:
        return "data_transfer", "context_length_exceeded", "confirmed by API rejection; complete input exceeds the fixed model context"
    if raw == "route_selection":
        return "classification", "disabled_route_proposal_blocked", "confirmed; modeling was prevented by the route guard"
    if raw == "classification":
        return "classification", r.get("execution_error_type", "classification_failure"), "confirmed"
    if raw == "data_extraction":
        return "data_extraction", r.get("execution_error_type", "extraction_failure"), "confirmed"
    if raw == "code_generation" or (raw == "formulation" and "[Gurobi Pipeline Step 3/3]" in log):
        return "code_generation", r.get("execution_error_type", "generation_failure"), "confirmed"
    if raw == "result_extraction":
        return "result_extraction", r.get("execution_error_type", "model_interface_failure"), "confirmed"
    if raw == "formulation":
        if "CSVQA exactly once; observed 0" in error:
            return "data_extraction", "mandatory_data_tool_skipped", "confirmed"
        return "mathematical_modeling", r.get("execution_error_type", "invalid_formulation"), "confirmed"
    if not r.get("final_ok"):
        if r.get("execution_error_type") in {"SyntaxError", "IndentationError", "NameError"}:
            return "code_generation", r["execution_error_type"], "confirmed invalid generated Python syntax/name"
        if any(s in error for s in ("unexpected keyword", "unhashable type", "Duplicate keys", "string indices", "object has no attribute 'casefold'", "TempConstr", "not function", "Constraint has no bool value")):
            return "code_generation", r.get("execution_error_type", "invalid_generated_api_or_index"), "confirmed by execution"
        return "execution", r.get("execution_error_type", "solver_or_data_access_failure"), "confirmed; upstream semantic cause may be unresolved"
    trace = json.loads(r.get("csvqa_trace") or "{}")
    conditions = [c for t in (trace.get("plan") or {}).get("tables", [])
                  for c in t.get("filters", {}).get("conditions", [])]
    if not trace.get("fallback_count") and any("such as" in c.get("evidence", "").lower() for c in conditions):
        return "data_extraction", "illustrative_names_used_as_subset", "confirmed from saved plan"
    return "mathematical_modeling_or_code_generation", "objective_mismatch", "unresolved; solver optimality does not prove formulation correctness"


def main(version="v7", output_dir=None):
    summary, per_class, failures, comparisons = [], [], [], []
    all_cases = []
    annotation_path = OUT / "failure_annotations.json"
    annotations = json.loads(annotation_path.read_text()) if annotation_path.exists() else {}
    root = OUT / ("full_" + version)
    full_pairs = attempts(root)
    full_rows = [r for _, r in full_pairs if "/automatic/" in str(_)]
    for label in CLASSES:
        group = [r for r in full_rows if r.get("true_label") == label]
        per_class.append({"class": label, "expected": SIZES[label], "recorded": len(group),
                          "classification_correct": sum(r.get("classification_correct") is True for r in group),
                          "solved": sum(r.get("final_ok") is True for r in group),
                          "objective_match": sum(r.get("solution_correct") is True for r in group)})
    for method in ["full", "rag_only", "few_shot_only", "examples_only", "examples_and_route"]:
        experiment_version = version
        folder = OUT / (method + "_" + experiment_version)
        if not folder.exists():
            comparisons.append({"method": method, "expected": 101, "recorded": 0, "status": "not_run"})
            continue
        pairs = attempts(folder)
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        grouped = {}
        for path, r in pairs:
            group = str(path.relative_to(folder)).split("/attempts/")[0]
            grouped.setdefault(group, []).append(r)
            stage, kind, evidence = failure_stage(path, r)
            annotation = annotations.get(f"{method}_{experiment_version}/{group}/{r['problem_id']}", annotations.get(f"{method}/{group}/{r['problem_id']}", {}))
            stage = annotation.get("failure_stage", stage)
            kind = annotation.get("failure_type", kind)
            evidence = annotation.get("cause_certainty", evidence)
            case = {"method": method, "version": experiment_version, "dataset": group, **{k:r.get(k) for k in (
                "problem_id", "true_label", "predicted_label", "assigned_route", "classification_correct",
                "final_ok", "solution_correct", "final_objective", "label_objective", "pipeline_stage",
                "execution_error_type", "execution_error", "csvqa_status", "repair_count", "retry_count",
                "api_retry_count", "api_retry_events", "fallback_count", "cache_source", "seconds",
                "protocol_retry_count", "protocol_retry_events",
                "classification_source",
                "held_out_type", "disabled_workflow_route", "route_allowed")},
                "failure_stage": stage, "failure_type": kind, "cause_certainty": evidence,
                "confirmed_cause": annotation.get("confirmed_cause", ""),
                "external_deadline_exceeded": bool(r.get("external_deadline_exceeded") or path.with_name("external_interrupt.json").exists()),
                "artifact_folder": str(path.parent.relative_to(OUT))}
            trace = json.loads(r.get("csvqa_trace") or "{}")
            missing_trace = (method != "few_shot_only" and r.get("assigned_route") in {"NRM","RA","TP","AP","FLP","Others"}
                             and r.get("input_kind") == "external_csv"
                             and r.get("pipeline_stage") == "formulation" and not trace.get("status"))
            case["csvqa_evidence_unavailable_at_failure"] = missing_trace
            if missing_trace:
                case["fallback_count"] = None
            all_cases.append(case)
            if not r.get("solution_correct"):
                failures.append(case)
        for group, expected in manifest.get("sizes", {}).items():
            rows = grouped.get(group, [])
            summary.append({"method": method, "version": experiment_version, "dataset": group, "expected": expected, "recorded": len(rows),
                            "classification_correct": sum(r.get("classification_correct") is True for r in rows),
                            "solved": sum(r.get("final_ok") is True for r in rows),
                            "objective_match": sum(r.get("solution_correct") is True for r in rows),
                            "objective_accuracy": sum(r.get("solution_correct") is True for r in rows)/expected,
                            "repair_count": sum(int(r.get("repair_count",0)) for r in rows),
                            "pipeline_retry_count": sum(int(r.get("retry_count",0))-int(r.get("protocol_retry_count",0)) for r in rows),
                            "protocol_retry_count": sum(int(r.get("protocol_retry_count",0)) for r in rows),
                            "http_retry_count": sum(int(r.get("api_retry_count",0)) for r in rows),
                            "recorded_full_data_fallback_count": sum(int(r.get("fallback_count",0)) for r in rows),
                            "csvqa_evidence_unavailable_cases": sum(r["csvqa_evidence_unavailable_at_failure"] for r in all_cases if r["method"]==method and r["dataset"]==group)})
        automatic = grouped.get("automatic", [])
        matched = sum(r.get("solution_correct") is True for r in automatic)
        full_matched = sum(r.get("solution_correct") is True for r in full_rows)
        comparisons.append({"method": method, "version": experiment_version, "expected": 101, "recorded": len(automatic),
                            "classification_correct": sum(r.get("classification_correct") is True for r in automatic),
                            "solved": sum(r.get("final_ok") is True for r in automatic), "objective_match": matched,
                            "delta_matches_vs_full": matched-full_matched if len(automatic)==len(full_rows)==101 else None,
                            "delta_percentage_points": 100*(matched-full_matched)/101 if len(automatic)==len(full_rows)==101 else None,
                            "status": "complete" if len(automatic)==101 else "incomplete"})
    stage_counts = [dict(method=m, failure_stage=stage, count=count)
                    for (m, stage), count in sorted(Counter((r["method"],r["failure_stage"]) for r in failures).items())]
    order = {name:i for i,name in enumerate(["full","rag_only","few_shot_only","examples_only","examples_and_route"])}
    def case_order(row):
        digits = re.findall(r"\d+",row["problem_id"])
        return (order[row["method"]],row.get("dataset","automatic"),int(digits[-1]) if digits else 0)
    all_cases.sort(key=case_order)
    failures.sort(key=case_order)
    output = Path(output_dir) if output_dir is not None else OUT / ("report_" + version)
    output.mkdir(exist_ok=True)
    full_by_id = {r["problem_id"]: r for r in all_cases if r["method"] == "full" and r["dataset"] == "automatic"}
    paired = []
    for row in all_cases:
        baseline = full_by_id.get(row["problem_id"])
        if row["method"] == "full" or not baseline:
            continue
        before, after = bool(baseline["solution_correct"]), bool(row["solution_correct"])
        paired.append({"method": row["method"], "problem_id": row["problem_id"],
                       "version": row["version"], "true_label": row["true_label"], "full_match": before, "experiment_match": after,
                       "change": "regression" if before and not after else "improvement" if after and not before else "unchanged",
                       "failure_stage": row["failure_stage"], "failure_type": row["failure_type"],
                       "cause_certainty": row["cause_certainty"], "artifact_folder": row["artifact_folder"]})
    paired.sort(key=case_order)
    folds = []
    for method in ["rag_only", "few_shot_only", "examples_only", "examples_and_route"]:
        for label in CLASSES:
            rows = [r for r in all_cases if r["method"] == method and r["true_label"] == label]
            base = [r for r in per_class if r["class"] == label][0]
            folds.append({"method": method, "class": label, "expected": SIZES[label], "recorded": len(rows),
                          "classification_correct": sum(r["classification_correct"] is True for r in rows),
                          "solved": sum(r["final_ok"] is True for r in rows),
                          "objective_match": sum(r["solution_correct"] is True for r in rows),
                          "delta_matches_vs_full": sum(r["solution_correct"] is True for r in rows)-base["objective_match"] if len(rows)==SIZES[label] else None,
                          "disabled_route_proposals": sum(r["assigned_route"] == r["disabled_workflow_route"] for r in rows if r["disabled_workflow_route"]),
                          "disabled_route_executions": sum(r["assigned_route"] == r["disabled_workflow_route"] and r["pipeline_stage"] != "route_selection" for r in rows if r["disabled_workflow_route"])})
    for name, rows in [("all_case_results", all_cases), ("failures", failures), ("metrics", summary),
                       ("full_101_by_class", per_class), ("experiment_comparison", comparisons),
                       ("paired_changes", paired), ("experiments_by_class", folds), ("failure_stage_counts", stage_counts)]:
        for row in rows:
            if "expected" in row:
                for metric in ["classification_correct", "solved", "objective_match"]:
                    if metric in row:
                        row[metric+"_percent"] = round(100*row[metric]/row["expected"], 3)
        write_csv(output / (name+".csv"), rows)
    full_summary = [r for r in summary if r["method"]=="full"]
    required = {"complete_101": len(full_rows)==101,
                "overall_101_at_least_93": sum(r.get("solution_correct") is True for r in full_rows)>=93,
                "category_minimum_92_percent": all(r["objective_match"]/r["expected"]>=.92 for r in per_class if r["class"] in {"AP","FLP","NRM","RA","TP"}),
                "variants_at_least_32": any(r["dataset"]=="variants" and r["recorded"]==36 and r["objective_match"]>=32 for r in full_summary),
                "all_nine_sheets_checked": sum(r["dataset"].startswith("columns/") and r["recorded"]==35 for r in full_summary)==9}
    goals = {r["dataset"]:{"matches":r["objective_match"],"recorded":r["recorded"],
                           "at_least_31":r["objective_match"]>=31,"at_least_90_percent":r["objective_match"]>=32}
             for r in full_summary if r["dataset"].startswith("columns/")}
    gate = {"required":required,"passed":all(required.values()),"redundant_column_goals":goals,
            "basis":"One complete prespecified pass; no best-of-round selection; fixed tolerances"}
    (output / "gate.json").write_text(json.dumps(gate,indent=2))
    # Keep the necessary-floor report distinct from the user's overall launch criterion.
    launch_gate = None
    if version.startswith('v') and version[1:].isdigit() and int(version[1:]) >= 10:
        from check_react_launch_gate import check
        with contextlib.redirect_stdout(io.StringIO()):
            launch_gate = check(version)
    overall_rows = []
    if launch_gate and launch_gate.get('overall') and launch_gate.get('baseline_overall'):
        for label, denominator, before, after in zip(
                ['101', 'Variants', 'Redundant columns (all nine sheets)'], [101, 36, 315],
                launch_gate['baseline_overall']['matches'], launch_gate['overall']['matches']):
            overall_rows.append({'group': label, 'expected': denominator, 'baseline_match': before,
                                 'current_match': after, 'delta_matches': after - before,
                                 'baseline_accuracy_percent': 100 * before / denominator,
                                 'current_accuracy_percent': 100 * after / denominator})
        write_csv(output / 'overall_comparison.csv', overall_rows)
    delivery_path = OUT / "delivery_control.json"
    delivery = json.loads(delivery_path.read_text()) if delivery_path.exists() else {}
    full_manifest_path = root / "manifest.json"
    retry_policy = json.loads(full_manifest_path.read_text()).get("react_protocol_restarts", False) if full_manifest_path.exists() else False
    all_csv_react = (json.loads(full_manifest_path.read_text()).get("csvqa_modes", {}).get("Others") == "planned"
                    if full_manifest_path.exists() else False)
    architecture_description = (
        "All six CSV workflows use ZERO_SHOT_REACT_DESCRIPTION with required CSVQA. "
        "Others/Mixture use the same symbolic-model and code-generation boundary. "
        "The original query-only ORLM_QA route remains."
        if all_csv_react else
        "NRM, RA, TP, AP and FLP use ZERO_SHOT_REACT_DESCRIPTION with CSVQA. "
        "Others with CSV retains its two-chain schema workflow; query-only ORLM_QA remains.")
    protocol_description = (
        "The agent must execute CSVQA at least once; multiple calls for different CSV files are allowed. "
        "Missing calls, malformed ReAct output and an incomplete Final Answer restart only the agent "
        "within the common case deadline, as explicitly requested by the user. Intermediate protocol "
        "attempts are logged and are not scored as separate failed cases. The first protocol-valid "
        "completion is retained; objective mismatches and generated-code errors never trigger retries."
        if retry_policy else
        "There is one agent invocation and exactly one executed CSVQA call. Skipped tools, duplicate "
        "requests and parsing failures are recorded as failures; no protocol restart exists.")
    def table(rows, fields):
        return "| " + " | ".join(fields) + " |\n|"+"|".join("---" for _ in fields)+"|\n" + "\n".join("| "+" | ".join(str(r.get(k,"")) for k in fields)+" |" for r in rows)+"\n"
    text = f"""# ReAct architecture evaluation: {version}

Status: {'overall launch gate passed' if launch_gate and launch_gate['passed'] else 'overall launch gate not passed' if launch_gate else 'necessary full-model gate passed' if gate['passed'] else 'full-model gate not yet passed'}.
606 enabled: {delivery.get('RUN_FORCED_ROUTES', False)}; executed: {delivery.get('606_executed', False)}.

{architecture_description}
The agent chooses CSVQA and receives its Observation before its Final Answer.
{protocol_description}
Few-shot Only uses the same ReAct agent class with an already supplied complete Python
Observation and no data tools.

All failed/error/truncated/timeout cases stay in the prescribed denominators.
Objective Match requires a finite objective from an optimal returned Gurobi model,
with unchanged rel_tol=abs_tol=1e-4. Classification and successful solve are separate.
A partial run is marked incomplete and is not a complete benchmark result.

## Overall comparison under the confirmed policy

{table(overall_rows,['group','expected','baseline_match','current_match','delta_matches','baseline_accuracy_percent','current_accuracy_percent']) if launch_gate else 'Historical policy applies to this preserved version.'}
{('Equal mean of the three group accuracies: baseline ' + str(round(100 * launch_gate['baseline_overall']['macro_accuracy'], 4)) + '%; current ' + str(round(100 * launch_gate['overall']['macro_accuracy'], 4)) + '%; change ' + str(round(launch_gate['macro_delta_percentage_points'], 4)) + ' percentage points. Pooled matches: ' + str(launch_gate['baseline_overall']['pooled_matches']) + '/452 to ' + str(launch_gate['overall']['pooled_matches']) + '/452.') if launch_gate and launch_gate.get('overall') and launch_gate.get('baseline_overall') else ''}
The current launch policy permits individual dataset, sheet and class declines.
It still requires 101 >= 93/101, AP/FLP/NRM/RA/TP >= 92% each, Variants >= 32/36,
unchanged inputs and complete evaluation of all nine redundancy sheets. Redundancy
targets remain improvement goals. All declines are reported. Partial totals do not
establish improvement. New studies are blocked until the complete launch gate passes.

## Full 101 by class

{table(per_class,['class','expected','recorded','classification_correct','solved','objective_match'])}
## Full datasets and every sheet

{table(full_summary,['dataset','expected','recorded','classification_correct','solved','objective_match'])}
## Ablations and LOTO against this ReAct baseline

{table(comparisons,['method','version','recorded','classification_correct','solved','objective_match','delta_matches_vs_full','delta_percentage_points','status'])}

Ablations reuse only the newly validated ReAct full classification cache. LOTO
reclassifies after target-reference removal. Examples Only retains every route;
Examples And Route bans the target workflow and checks availability before modeling.
The original semantic-label accuracy is zero by design in the route-removal experiment.
Only fold grouping/exclusion and the scorer use the true class; no gold model/answer
is used to choose a replacement route or generate code.

## Failures

{table(stage_counts,['method','failure_stage','count'])}
{table(failures,['method','dataset','problem_id','failure_stage','failure_type','cause_certainty'])}

Unresolved objective mismatches remain unresolved until the saved model/program/source
records support an explanation. Missing CSVQA evidence is explicitly marked, and
fallback totals are lower bounds when evidence is unavailable. No trace is fabricated.

## Repairs, retries and fallbacks

{table(summary,['method','dataset','repair_count','pipeline_retry_count','protocol_retry_count','http_retry_count','recorded_full_data_fallback_count','csvqa_evidence_unavailable_cases'])}

HTTP retries retain the original SDK transport policy and are recorded separately.
ReAct protocol restarts follow this version's recorded user policy. Code and solver
repair and objective-based retries remain disabled. Each worker uses Gurobi Threads=2;
MIPGap=1e-4 and a common external 1,800-second deadline remain unchanged.

## Preserved controls and reproduction

The earlier direct-call results remain under outputs/minimal_revision_20261005.
Their 97/101 is not a verified score of the restored ReAct architecture. Current
notebooks are backed up before restoration under backups_direct_call_version.
The source-data extraction, exact identifiers, axis validation, generic solver
interfaces and immutable failed-case cache are retained. No question-ID, objective,
or reference-answer branch is added to generation.

Use /opt/miniconda3/envs/lean_llm_opt_4_1/bin/python with:

```bash
python scripts/evaluate_react_revision_20261006.py --method full --version {version} --workers 6 --run
python scripts/evaluate_react_revision_20261006.py --method rag_only --version {version} --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version {version} --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_only --version {version} --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version {version} --workers 3 --run
python scripts/report_react_revision_20261006.py --version {version}
```

Every case/version has one scored pipeline attempt. This version's protocol restarts
are logged within that attempt. Resuming preserves failures as well as successes.
Changed code or data requires a fresh result directory. Every manifest freezes the
notebook, input hashes, fixed model snapshot and execution settings. Source files,
model/code artifacts and logs are retained. 606 stays disabled until this revision
passes the required full-model gate and all nine sheets have been reviewed.
"""
    (output / "REPORT.md").write_text(text)
    print(json.dumps(gate,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v7")
    parser.add_argument("--output-dir", type=Path, help="Save a separate report without replacing a historical report")
    args = parser.parse_args()
    main(args.version, args.output_dir)
