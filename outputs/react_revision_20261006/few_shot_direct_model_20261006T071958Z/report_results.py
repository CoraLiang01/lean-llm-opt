"""Report one completed revised Few-shot Only pass without inference or rescoring."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/react_revision_20261006"
RUN = OUT / "few_shot_only_v7_direct_model_20261006T071958Z"
REPORT = RUN / "report"
REPORT.mkdir(exist_ok=True)
status = json.loads((RUN / "run_status.json").read_text())
assert status["complete"] and status["new_attempts"] == 101
manifest = json.loads((RUN / "manifest.json").read_text())
input_changes = [p for p, digest in manifest["inputs"].items()
                 if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
baseline_inputs = json.loads((OUT / "full_v7/manifest.json").read_text())["inputs"]
baseline_input_changes = [p for p, digest in baseline_inputs.items()
                          if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
old_hashes = json.loads((RUN / "historical_result_hashes.json").read_text())
historical_changes = [p for p, digest in old_hashes.items()
                      if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
assert not input_changes and not baseline_input_changes and not historical_changes
assert hashlib.sha256((ROOT / "Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb").read_bytes()).hexdigest() == manifest["notebook_sha256"]


def load(folder):
    rows = [json.loads(p.read_text()) for p in (OUT / folder).glob("automatic/attempts/*/result.json")]
    assert len(rows) == 101 and len({r["problem_id"] for r in rows}) == 101
    return {r["problem_id"]: r for r in rows}


full = load("full_v7")
old = load("few_shot_only_v7")
new = load(RUN.name)
classes = ["AP", "FLP", "NRM", "RA", "TP", "Others", "Mixture"]
comparison, by_class = [], []
for name, lookup in [("Full ReAct v7 (existing)", full), ("Few-shot Only v7 (previous)", old),
                     ("Few-shot Only direct model (new)", new)]:
    rows = list(lookup.values())
    counts = {"classification_correct": sum(r.get("classification_correct") is True for r in rows),
              "solved": sum(r.get("final_ok") is True for r in rows),
              "objective_match": sum(r.get("solution_correct") is True for r in rows)}
    comparison.append({"experiment": name, "cases": 101, **counts,
                       "objective_match_percent": counts["objective_match"] / 101 * 100})
    for label in classes:
        group = [r for r in rows if r["true_label"] == label]
        by_class.append({"experiment": name, "class": label, "cases": len(group),
                         "classification_correct": sum(r.get("classification_correct") is True for r in group),
                         "solved": sum(r.get("final_ok") is True for r in group),
                         "objective_match": sum(r.get("solution_correct") is True for r in group)})

fields = ["problem_id", "true_label", "predicted_label", "assigned_route", "classification_correct",
          "final_ok", "solution_correct", "final_objective", "label_objective", "pipeline_stage",
          "execution_error_type", "execution_error", "external_deadline_exceeded", "seconds",
          "cache_source", "classification_source", "repair_count", "retry_count", "protocol_retry_count",
          "api_retry_count", "api_retry_events", "protocol_retry_events", "fallback_count", "notebook_source_sha256"]
notes = json.loads((RUN / "confirmed_failure_notes.json").read_text()) if (RUN / "confirmed_failure_notes.json").exists() else {}
cases, paired = [], []
for case_id in sorted(new):
    row = new[case_id]
    artifact = RUN / "automatic/attempts" / case_id
    (artifact / "model.md").write_text(row.get("generated_model") or "")
    (artifact / "solve.py").write_text(row.get("solve_code") or "")
    (artifact / "solution.json").write_text(json.dumps({key: row.get(key) for key in [
        "final_ok", "final_objective", "final_solution", "execution_error_type", "execution_error",
        "external_deadline_exceeded"]}, indent=2) + "\n")
    check_attempt = json.loads((RUN / "automatic/attempts" / case_id / "attempt.json").read_text())
    assert check_attempt["state"] == "finished"
    assert row["cache_source"] == "computed" and row["classification_source"] == "final_full_cache"
    assert row["notebook_source_sha256"] == manifest["notebook_sha256"]
    assert row["predicted_label"] == full[case_id]["predicted_label"]
    item = {key: row.get(key) for key in fields}
    if row.get("solution_correct") is True:
        stage, reason = "none", "objective_match"
    elif row.get("final_ok") is True:
        stage, reason = "modeling_or_code_generation_unconfirmed", "objective_mismatch"
    else:
        stage, reason = row.get("pipeline_stage"), row.get("execution_error_type") or "no_valid_optimal_result"
    note = notes.get(case_id, {}) if not row.get("solution_correct") else {}
    item.update(failure_stage=note.get("failure_stage", stage), failure_type=note.get("failure_type", reason),
                cause_confirmed=note.get("confirmed", False), cause_evidence=note.get("evidence", ""))
    cases.append(item)
    paired.append({"problem_id": case_id, "class": row["true_label"],
                   "full_match": full[case_id].get("solution_correct") is True,
                   "previous_few_match": old[case_id].get("solution_correct") is True,
                   "new_few_match": row.get("solution_correct") is True,
                   "change_vs_previous": int(row.get("solution_correct") is True) - int(old[case_id].get("solution_correct") is True),
                   "change_vs_full": int(row.get("solution_correct") is True) - int(full[case_id].get("solution_correct") is True)})
pd.DataFrame(comparison).to_csv(REPORT / "comparison.csv", index=False)
pd.DataFrame(by_class).to_csv(REPORT / "by_class.csv", index=False)
pd.DataFrame(cases).to_csv(REPORT / "all_case_results.csv", index=False)
pd.DataFrame([r for r in cases if not r["solution_correct"]]).to_csv(REPORT / "failures.csv", index=False)
pd.DataFrame(paired).to_csv(REPORT / "paired_changes.csv", index=False)
pd.DataFrame([r for r in cases if not r["solution_correct"]]).groupby(
    ["failure_stage", "failure_type"], dropna=False).size().rename("cases").reset_index().to_csv(
        REPORT / "failure_stage_counts.csv", index=False)
rows = list(new.values())
traces = [json.loads(r.get("csvqa_trace") or "{}") for r in rows]
audit = {"cases": 101, "new_computed_cases": 101, "cached_classification_cases": 101,
         "csvqa_calls": sum(int(t.get("csvqa_call_count", 0)) for t in traces),
         "repair_count": sum(int(r.get("repair_count", 0)) for r in rows),
         "protocol_restart_count": sum(int(r.get("protocol_retry_count", 0)) for r in rows),
         "pipeline_retry_count": sum(int(r.get("retry_count", 0)) - int(r.get("protocol_retry_count", 0)) for r in rows),
         "http_retry_count": sum(int(r.get("api_retry_count", 0)) for r in rows),
         "fallback_count": sum(int(r.get("fallback_count", 0)) for r in rows),
         "missing_observation_trace_cases": sum(not t.get("status") for t in traces),
         "input_files_checked": len(manifest["inputs"]), "input_changes": input_changes,
         "baseline_input_files_checked": len(baseline_inputs), "baseline_input_changes": baseline_input_changes,
         "historical_sources_results_checked": len(old_hashes), "historical_changes": historical_changes,
         "source_sha256": manifest["notebook_sha256"], "complete": True}
(REPORT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
review = json.loads((OUT / "few_shot_v7_transfer_failure_review.json").read_text())
rejected = [{"problem_id": k, "solved": new[k].get("final_ok") is True,
             "objective_match": new[k].get("solution_correct") is True,
             "failure_stage": next(c["failure_stage"] for c in cases if c["problem_id"] == k)}
            for k in review["case_ids"]]
pd.DataFrame(rejected).to_csv(REPORT / "previous_transfer_rejections.csv", index=False)
summary = ["# Revised Few-shot Only: one complete 101-case evaluation", "",
           "All cases were computed once under the revised source. Classification predictions were reused from full ReAct v7. Failures, truncation, context errors and 1,800-second deadlines remain in the denominator. No best-of-round selection is applied.", "",
           "| Experiment | Classification | Optimal solve | Objective Match | OM percent |",
           "|---|---:|---:|---:|---:|"]
for r in comparison:
    summary.append(f"| {r['experiment']} | {r['classification_correct']}/101 | {r['solved']}/101 | {r['objective_match']}/101 | {r['objective_match_percent']:.2f}% |")
summary.extend(["", "| Class | Full v7 OM | Previous Few-shot OM | Revised classification | Revised solve | Revised OM |",
                "|---|---:|---:|---:|---:|---:|---:|"])
for label in classes:
    group = [r for r in by_class if r["class"] == label]
    a, b, c = group
    n = c["cases"]
    summary.append(f"| {label} | {a['objective_match']}/{n} | {b['objective_match']}/{n} | {c['classification_correct']}/{n} | {c['solved']}/{n} | {c['objective_match']}/{n} |")
summary.extend(["", "Execution audit:", "", "```json", json.dumps(audit, indent=2), "```", "",
                "The revised notebook forwards the formulation unchanged, replaces current-case CSVQA with complete Python Observation for modeling, and does not separately forward Observation to code generation. Shared ReAct v7 prompts/examples, model configuration, solver settings, matching tolerance and no-CSV route remain unchanged apart from necessary data-source instructions.", "",
                "The source/interface changes were evaluated together. Differences from the earlier pass include model-output variation; they cannot all be causally assigned to one change. A single complete pass does not establish cross-run stability.", "",
                "Failure-stage causes are marked confirmed only when supported by saved query/model/code/input evidence. Otherwise they remain unconfirmed. Each raw model, code, log and result is retained in automatic/attempts/<problem_id>.", "",
                "Failures:", ""])
for r in cases:
    if not r["solution_correct"]:
        summary.append(f"- {r['problem_id']} ({r['true_label']}): {r['failure_stage']}; {r['failure_type']}.")
(REPORT / "SUMMARY.md").write_text("\n".join(summary) + "\n")
print(json.dumps({"comparison": comparison, "audit": audit, "previous_rejections": rejected}, indent=2))
