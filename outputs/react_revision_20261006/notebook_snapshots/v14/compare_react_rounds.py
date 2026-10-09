"""Compare recorded complete passes without inference, rescoring or result selection."""
import argparse
from collections import Counter
import json

from evaluate_react_revision_20261006 import OUT
from report_react_revision_20261006 import write_csv


def recorded_cases(version):
    root = OUT / f"full_{version}"
    state = json.loads((root / "run_status.json").read_text())
    # A final service error may mark the coordinator incomplete even though every
    # prescribed case has a recorded outcome. Retain those failures in comparison.
    assert state.get("complete") is True or (state.get("new_attempts") == 452
                                               and state.get("fatal_service_error") is True)
    cases = {}
    for path in root.glob("**/attempts/*/result.json"):
        group = str(path.relative_to(root)).split("/attempts/")[0]
        result = json.loads(path.read_text())
        key = (group, result["problem_id"])
        assert key not in cases, f"Duplicate recorded case: {key}"
        cases[key] = result
    assert len(cases) == 452
    return cases


def compare(version, baseline="v7"):
    current, previous = recorded_cases(version), recorded_cases(baseline)
    assert current.keys() == previous.keys()
    manifests = [json.loads((OUT / f"full_{v}" / "manifest.json").read_text())
                 for v in (version, baseline)]
    assert manifests[0]["inputs"] == manifests[1]["inputs"]
    import pandas as pd
    report = OUT / f"report_{version}"
    details = pd.read_csv(report / "all_case_results.csv").fillna("")
    details = {(r["dataset"], r["problem_id"]): r for r in details.to_dict("records")
               if r["method"] == "full"}
    rows = []
    for key in sorted(current):
        before, after, detail = previous[key], current[key], details[key]
        assert before["true_label"] == after["true_label"]
        old_match, new_match = before["solution_correct"] is True, after["solution_correct"] is True
        rows.append({"dataset": key[0], "problem_id": key[1],
                     "true_label": after["true_label"], "baseline_version": baseline,
                     "current_version": version,
                     "baseline_classification_correct": before["classification_correct"],
                     "current_classification_correct": after["classification_correct"],
                     "baseline_predicted_label": before.get("predicted_label"),
                     "current_predicted_label": after.get("predicted_label"),
                     "baseline_solved": before["final_ok"], "current_solved": after["final_ok"],
                     "baseline_objective_match": old_match, "current_objective_match": new_match,
                     "change": "regression" if old_match and not new_match else
                               "improvement" if new_match and not old_match else "unchanged",
                     **{k: detail[k] for k in ("failure_stage", "failure_type", "cause_certainty",
                                               "confirmed_cause", "artifact_folder")}})
    counts = Counter(r["change"] for r in rows)
    delta = sum(r["solution_correct"] is True for r in current.values()) - sum(
        r["solution_correct"] is True for r in previous.values())
    assert counts["improvement"] - counts["regression"] == delta
    summary = {"current_version": version, "baseline_version": baseline, "cases": len(rows),
               "all_prescribed_outcomes_recorded": True,
               "coordinator_complete": json.loads((OUT / f"full_{version}" / "run_status.json").read_text())["complete"],
               "inputs_unchanged": True, "changes": dict(counts), "pooled_delta_matches": delta,
               "regression_stages": dict(Counter(r["failure_stage"] for r in rows
                                                  if r["change"] == "regression")),
               "regression_types": dict(Counter(r["failure_type"] for r in rows
                                                 if r["change"] == "regression")),
               "policy": "First recorded completion of each frozen version; no inference, rescoring or best-result selection."}
    write_csv(report / f"full_vs_{baseline}_cases.csv", rows)
    (report / f"full_vs_{baseline}_transitions.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--baseline", default="v7")
    args = parser.parse_args()
    compare(args.version, args.baseline)
