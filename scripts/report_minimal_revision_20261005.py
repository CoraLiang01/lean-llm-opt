"""Report every attempt with fixed denominators and explicit uncertainty."""
from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/minimal_revision_20261005"
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


def main(version="v4"):
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
        experiment_version = "v5" if method == "few_shot_only" else version
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
                "classification_source",
                "held_out_type", "disabled_workflow_route", "route_allowed")},
                "failure_stage": stage, "failure_type": kind, "cause_certainty": evidence,
                "confirmed_cause": annotation.get("confirmed_cause", ""),
                "external_deadline_exceeded": bool(r.get("external_deadline_exceeded") or path.with_name("external_interrupt.json").exists()),
                "artifact_folder": str(path.parent.relative_to(OUT))}
            trace = json.loads(r.get("csvqa_trace") or "{}")
            missing_trace = (method != "few_shot_only" and r.get("assigned_route") in {"NRM","RA","TP","AP","FLP"}
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
                            "pipeline_retry_count": sum(int(r.get("retry_count",0)) for r in rows),
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
    output = OUT / "report"
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
    delivery_path = OUT / "delivery_control.json"
    delivery = json.loads(delivery_path.read_text()) if delivery_path.exists() else {}
    def table(rows, fields):
        return "| " + " | ".join(fields) + " |\n|"+"|".join("---" for _ in fields)+"|\n" + "\n".join("| "+" | ".join(str(r.get(k,"")) for k in fields)+" |" for r in rows)+"\n"
    text = f"""# Evaluation report: {version}

Status: {'necessary full-model gate passed' if gate['passed'] else 'full-model gate not yet passed'}.
606 switch enabled: {delivery.get('RUN_FORCED_ROUTES', False)}; 606 executed: {delivery.get('606_executed', False)}.
All rates use the complete prespecified dataset size. Failed, truncated, erroneous,
interrupted and unavailable results are never removed from the denominator. Pending
cases are marked incomplete; a partial run is not a complete benchmark result.
Objective Match uses unchanged `rel_tol=1e-4, abs_tol=1e-4` and requires a finite
objective from an optimal live Gurobi model. Classification accuracy compares the
seven semantic labels; solve success requires completed execution/result extraction
and an optimal live Gurobi model. An optimal printed value followed by a program error
is not a valid returned result and remains a failed case.
Neither classification correctness nor solver optimality implies Objective Match.

## Full model: 101 cases by class

{table(per_class,['class','expected','recorded','classification_correct','solved','objective_match'])}
## Full model: every dataset and sheet

{table(full_summary,['dataset','expected','recorded','classification_correct','solved','objective_match'])}
## Ablations and LOTO compared with the same final full definitions

{table(comparisons,['method','version','recorded','classification_correct','solved','objective_match','delta_matches_vs_full','delta_percentage_points','status'])}
Route-removal LOTO gold-label classification accuracy is expected to be zero:
the held-out label's workflow is deliberately unavailable. Selected-route validity
is a separate constraint, audited before formulation and reported in case records.
The true type is used only for fold grouping/exclusion, never to choose a replacement route.
`experiments_by_class.csv` reports every class/fold; `paired_changes.csv` identifies
each improvement and regression against this exact frozen full baseline.

## Failure evidence

{table(stage_counts,["method","failure_stage","count"])}

{table(failures,['method','dataset','problem_id','failure_stage','failure_type','cause_certainty'])}
Objective mismatches whose semantic origin is unresolved are explicitly marked.
The saved original query, Observation/plan, mathematical model, program and execution
log allow individual review without feeding any reference model or target value to generation.
Some failed mathematical-model requests return before CSVQA evidence is attached to
the case record. Their `csvqa_evidence_unavailable_at_failure` flag is explicit; their
fallback counts are unknown, not claimed zero. Recorded fallback totals are lower bounds
where this occurs. No missing plan or Observation is fabricated or regenerated.

## Retry, fallback and repair records

{table(summary,["method","dataset","repair_count","pipeline_retry_count","http_retry_count","recorded_full_data_fallback_count","csvqa_evidence_unavailable_cases"])}
HTTP retries are the original SDK transport policy, not a new model repair or best-of-output run.
When CSVQA evidence is unavailable, its fallback total is a lower bound. Full-model final cases have complete CSVQA evidence.

## Changes and controls

1. Reuse the existing planned CSVQA data path for AP, TP and FLP. Python invokes
   the CSVQA Tool once before one modeling call. This fixes actual skipped-tool
   failures without protocol correction, model reruns or generated-code repairs.
2. Preserve every original field value and row position. Validate matrix axes and
   expose an optional bijection only for declared axes with complete unique matching
   numeric suffix strings. Leading zeros are preserved; unresolved axes fail the plan
   validation and use the existing logged full-source fallback.
3. Reject filters justified only by illustrative examples, separate categorical-code
   prefixes from arbitrary substring matching, and prevent demonstration/sample
   entity counts from narrowing the current data. These rules never inspect a case ID,
   gold objective or reference model.
4. Keep common model-return, Gurobi naming, tuple indexing and logical-constraint
   interface guidance independent of retrieved examples. The query-only Others
   modeling route and its reference source remain; only single-pass parser/client
   controls are shared. No model repair, truncation rerun or solver/code retry exists.
5. Restore the original SDK maximum of two HTTP transport retries and share a
   connection pool after a recorded TLS EOF. Actual HTTP retries are logged through
   `x-stainless-retry-count`, separately from zero pipeline/model retries.
   A common external limit of 1,800 seconds per case prevents indefinitely blocked
   experiments. It was adopted after a saved makespan run exceeded ten million
   branch-and-bound nodes without proof-gap progress. All previously completed final
   cases were below that limit. Solver parameters and matching tolerances stay unchanged;
   interrupted/nonoptimal results count as failures and are never regenerated. The
   adoption record and each method's `runtime_control.json` preserve this decision.
6. RAG Only removes retrieved model/Observation demonstrations for NRM, RA, TP,
   AP and FLP, Others Abstract Model Plan/code demonstrations, retrieved code examples,
   inserted query-only demonstrations and its three fixed Question/Final Answer triggers.
   Query-only ORLM_QA retrieval, classification FileQA evidence and current
   CSVQA loading/profiling, plans, Python extraction/validation and full-data fallback
   remain, as does the Others CSV schema/statistics/runtime reading path.
   Both ablations use the full model's recorded predictions, so classification
   retrieval is retained in definitions but is not rerun during these comparisons.
7. Few-shot Only retains route model/code examples. Python supplies every current
   CSV field directly as text to mathematical modeling. There is no CSVQA Tool,
   planner, LLM extraction/summary, row selection or data rewrite. Code generation
   gets the symbolic variable/objective/constraint sections, query, paths and column names; complete current CSV
   Observations are not forwarded to it. Python removes echoed markdown tables and parameter/Data Mapping
   sections before transfer, without summarizing or rewriting source values. The generated program reads source files.
   This boundary was added after inspecting saved v4 models that echoed complete coefficient tables.
   The v4 Few-shot Only pass (80/101) remains preserved as a preliminary boundary-violating round;
   the complete v5 pass is the final comparison regardless of whether its score improves.
8. Both LOTO notebooks filter the target semantic type from the effective classifier
   reference (`RAG_Examples_All`, which the baseline actually uses), route example
   library and fixed classifier demonstrations. Generic class definitions and model
   interfaces remain. Examples Only retains all routes; Examples And Route reclassifies
   among remaining labels and checks route availability before any formulation.
   The separate query-only reference library is also filtered by its original
   problem-type field, and target-type fixed query-only triggers are removed. The
   full notebook's query-only route remains intact. These query-only exclusions were
   checked offline; the supplied 101-case benchmark has external CSV inputs throughout.

`RUN_AUTOMATIC`, `RUN_VARIANTS`, `RUN_REDUNDANT_COLUMNS`, `RUN_API`, and `RUN_LOTO`
are false in delivered notebooks until explicitly enabled. The 606 switch is only
enabled after the necessary full-model gate and nine-sheet review. Redundant-column
90% is an improvement goal (32/35); each sheet's goal status is in `gate.json`.
The delivered full notebook differs from its evaluated frozen copy only in the
606 explanation/switch cells (38 and 39); all modeling/execution definitions and inputs
are identical. The two delivered LOTO notebooks also correct only the unused query-only
reference filter in cell 39: Knapsack Problem is classified as RA for target-reference exclusion.
All other LOTO definitions match their evaluated copies, and all 101 inputs have CSV files,
so no evaluated execution uses that filter. The corrected no-CSV filter is verified offline
(`query_only_alias_boundary_verification.json`); its API performance is not evaluated.
Experiment provenance and cached classifications refer to the frozen validated copies. `delivery_control.json` records both hashes and the gate snapshot.

## Reproduction and preserved history

Final comparison folders are `full_v4`, `rag_only_v4`, `few_shot_only_v5`, `examples_only_v4` and `examples_and_route_v4`.
`few_shot_v5_adoption.json` preserves the boundary correction and the mandatory complete new-pass policy.
`rag_only_removed_example_candidates.json` lists all 15 effective reference rows (their source indices, types and prompts)
whose model/Observation/code demonstrations are removed; the three query-only fixed triggers are INTEGER, MULTI-PERIOD FLOW and LOGIC+BINARY.
`component_manifest.json` lists retained retrieval functions and separate query-only reference coverage.

Run with `/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python`:

```bash
python scripts/evaluate_minimal_revision_20261005.py --method full --version {version} --workers 10 --run
python scripts/evaluate_minimal_revision_20261005.py --method rag_only --version {version} --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method few_shot_only --version v5 --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method examples_only --version {version} --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method examples_and_route --version {version} --workers 3 --run
python scripts/report_minimal_revision_20261005.py
```

The runner freezes notebook bytes, code/data hashes, paths, settings, model snapshot,
solver threads and matching tolerances before inference. A narrowly checked delivery exception
allows the preserved CSV LOTO pass to be read/resumed after the no-CSV alias-only correction;
changed notebook hashes, replacement fragments and zero query-only cases are recorded.
A fresh experiment version freezes the current corrected notebook. It strips unused gold model
text from worker inputs. Each case has an attempt marker, result, artifacts and log;
every recorded failure is preserved during resume. An interrupted attempt is recorded
as failed and never regenerated. New versions and sheets use distinct directories.
`environment.json` records the interpreter and library versions; original inputs remain
unchanged, with hashes in each manifest. `backups/` contains the five files before editing.

Historical 96/101 and 32/36 results did not match the initial current source fingerprint.
Only the historical additional-benchmark 200pct-S1 records matched it (31/35).
Those historical figures are context, not current-version results. Diagnostic v1/v2/v3
stopped new submissions after evidence of general failures; their attempts and state remain separate,
and none is substituted into {version}. A single final pass does not prove repeat-run stability.
"""
    (output / "REPORT.md").write_text(text)
    print(json.dumps(gate,indent=2))


if __name__ == "__main__":
    main()
