"""Check staged derivations offline, then optionally publish the exact checked bytes."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import nbformat
import pandas as pd

from check_0927_review import gate
from check_0927_final_methods import definitions, check_complete_payload, payload_in_prompt, messages
from evaluate_0927_optimization import namespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
RUNS = ROOT / "outputs/optimization_0927_20261006"
NAMES = {"full": "LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb",
         "rag_only": "Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb",
         "few_shot_only": "Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb",
         "examples_and_route": "LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(publish=False, runtime=False):
    control = json.loads((OUT / "staged_delivery_control.json").read_text())
    version = control["version"]
    assert gate(version)["passed"]
    staged = Path(control["staged_directory"])
    spaces, defs = {}, {}
    frozen = RUNS / version / "frozen_notebook.ipynb"
    expected = definitions(frozen, 36)
    audit = {"version": version, "network_calls": 0, "hashes": {}, "common_engine": {}, "boundaries": {}, "runtime": {}}
    for method, name in NAMES.items():
        path = staged / name
        book = nbformat.read(path, as_version=4); nbformat.validate(book)
        assert all(not c.outputs and c.execution_count is None for c in book.cells if c.cell_type == "code")
        assert sha(path) == control["delivered_sha256"][name]
        spaces[method], defs[method] = namespace(path), definitions(path, 36)
        audit["hashes"][name] = sha(path)
    full = spaces["full"]
    assert defs["full"] == expected
    evaluated = namespace(frozen)
    for name in ["MODEL_SNAPSHOT", "EMBEDDING_MODEL", "CSV_SOLVER_INSTRUCTIONS", "CSVQA_MODE_BY_ROUTE", "CSV_ROUTE_HINT", "PLANNED_CODE_INSTRUCTIONS", "LEGACY_CODE_INSTRUCTIONS", "prefix", "few_shot_example", "NRM_RETRY_ON_TRUNCATION"]:
        assert full[name] == evaluated[name]
    for name in ['QUERY_SEMANTICS_GUIDANCE', 'SOURCE_TABLE_GUIDANCE']:
        if name in evaluated:
            assert full[name] == evaluated[name]
    common = ["execute_code", "_source_candidate", "objective_is_correct", "make_llm", "make_embeddings", "read_csv_compat", "_read_csv", "case_fields", "error_record", "save_record", "materialize_record", "reusable_record"]
    for method in NAMES:
        if method == "full":
            continue
        for name in common:
            assert defs[method][name] == expected[name], (method, name)
        for name in ["MODEL_SNAPSHOT", "EMBEDDING_MODEL", "CSV_SOLVER_INSTRUCTIONS", "CSVQA_MODE_BY_ROUTE", "NRM_RETRY_ON_TRUNCATION"]:
            assert spaces[method][name] == full[name]
        for name in ['QUERY_SEMANTICS_GUIDANCE', 'SOURCE_TABLE_GUIDANCE']:
            if name in full:
                assert spaces[method][name] == full[name]
        assert spaces[method]["make_llm"]().max_retries == 1
        audit["common_engine"][method] = common
    rag, few, loto = [spaces[x] for x in ["rag_only", "few_shot_only", "examples_and_route"]]
    retained = ["build_csvqa_components", "_load_tables", "load_csv_documents", "_ask_extraction_plan", "_execute_plan", "_build_profile", "csv_schema_preview", "formulate_with_csvqa"]
    if 'validate_legacy_source_rows' in full:
        retained.append('validate_legacy_source_rows')
    for name in retained:
        assert defs["rag_only"][name] == expected[name]
    for route in full["WORKFLOW_ROUTES"]:
        assert rag["retrieve_rag_examples"](route, "audit", 3) == []
    references = pd.read_csv(ROOT / "Large_Scale_Or_Files/RAG_Examples_All.csv", dtype=str, keep_default_na=False)
    audit["boundaries"]["rag_only"] = {"removed_example_rows": references.index.tolist(),
        "retained": ["CSVQA ReAct tool", "source CSV full row context and legacy extraction", "NRM plan and Python executor", "Others schema profiling and runtime complete CSV reads"],
        "implementation_note": "CSVQA source data uses a stuff chain; FAISS retrieves reference examples, not source CSV rows."}
    for name in ["_ask_extraction_plan", "_execute_plan", "build_csvqa_components", "load_csv_documents", "validate_legacy_source_rows"]:
        assert name not in few
    for kwargs in [{"legacy_observation": "raw"}, {"data_payload": {"tables": [{"records": []}]}}]:
        try:
            few["_generate_code"]("model", "NRM", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Complete Observation reached code generation")
    from langchain_core.language_models.fake import FakeListLLM
    original = few["make_llm"]
    try:
        few["make_llm"] = lambda *a, **kw: FakeListLLM(responses=["Thought: use exact source values.\nFinal Answer: x >= 0"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = few["formulate_from_python_observation"]("test", '{"exact":"001"}', "Modeler", "Question: {input}\n{agent_scratchpad}")
        assert result["trace"]["react_data_tool_calls"] == 0
        assert result["observation"] == '{"exact":"001"}'
    finally:
        few["make_llm"] = original
    def augments(code):
        node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "formulate_with_csvqa")
        return [ast.dump(n, include_attributes=False) for n in node.body if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == "prefix"]
    assert augments("".join(nbformat.read(frozen, as_version=4).cells[15].source)) == augments(nbformat.read(staged / NAMES["few_shot_only"], as_version=4).cells[15].source)
    inventory = full["load_benchmark"]()
    source_cells = sum(check_complete_payload(few["direct_source_payload"](case["dataset_address"], "Others")) for case in inventory.to_dict("records"))
    audit["boundaries"]["few_shot_only"] = {"tool_free_react": True, "raw_records_rejected_at_codegen": True, "common_math_guidance_retained": True, "complete_csv_cells_verified": source_cells}
    cache = pd.read_csv(control["classification_main_cache"], dtype=str, keep_default_na=False)
    assert len(cache) == 101 and not set(cache.columns) & {"true_label", "label_objective", "generated_model", "solve_code", "solution_correct"}
    assert sha(control["classification_main_cache"]) == control["classification_main_sha256"]
    for method in ["rag_only", "few_shot_only"]:
        attached = spaces[method]["attach_cached_classifications"](inventory)
        for case in attached.to_dict("records"):
            assert spaces[method]["classification_for_case"](case)["normalized_label"] == case["cached_predicted_label"]
    masks = []
    for label in loto["CLASS_LABELS"]:
        manifest = loto["set_loto_fold"](label)
        disabled = manifest["disabled_workflow_route"]
        assert disabled not in manifest["allowed_routes"]
        assert all(loto["normalize_problem_class"](references.loc[index, "Type"]) != label for index in manifest["remaining_example_indices"])
        for fn in [lambda: loto["invoke_formulation"](disabled, "test", "unopened.csv"),
                   lambda: loto["_generate_code"]("model", disabled),
                   lambda: loto["retrieve_rag_examples"](disabled, "test", 3)]:
            try:
                fn()
            except loto["DisabledRouteError"]:
                pass
            else:
                raise AssertionError("Disabled route passed a guard")
        masks.append({"held_out": label, "disabled": disabled, "removed_rows": [r["row_index"] for r in manifest["removed_examples"]]})
    audit["boundaries"]["loto"] = masks
    if runtime:
        lookup = cache.set_index("problem_id").to_dict("index")
        full_inputs = json.loads((RUNS / version / "run_manifest.json").read_text())["input_sha256"]
        for method in ["rag_only", "few_shot_only", "examples_and_route"]:
            folder = RUNS / control.get('method_runs', {}).get(method, f"{method}_{version}")
            manifest = json.loads((folder / "run_manifest.json").read_text())
            assert manifest["notebook_sha256"] == control["delivered_sha256"][NAMES[method]]
            for path, digest in manifest["input_sha256"].items():
                assert sha(path) == digest
                if path in full_inputs:
                    assert full_inputs[path] == digest
            records = [json.loads(p.read_text()) for p in (folder / "attempts").rglob("result.json")]
            assert {r["problem_id"] for r in records} == set(inventory["problem_id"]) and len(records) == 101
            result = {"evaluated": len(records), "forbidden_routes_used": 0, "full_observations": 0, "verified_csv_cells": 0}
            for record in records:
                assert record["notebook_source_sha256"] == manifest["notebook_sha256"]
                if method != "examples_and_route":
                    expected_row = lookup[record["problem_id"]]
                    assert all(record[k] == expected_row[k] for k in ["query", "dataset_address", "predicted_label", "assigned_route"])
                else:
                    fold = json.loads((folder / "attempts" / record["problem_id"] / "fold_manifest.json").read_text())
                    assert record["held_out_type"] == fold["held_out_type"] == record["true_label"]
                    assert record.get("assigned_route") != fold["disabled_workflow_route"]
                    if record.get("predicted_label"):
                        assert record["predicted_label"] in fold["allowed_labels"]
                if method == "few_shot_only":
                    observed = False
                    for text in messages(folder / "attempts" / record["problem_id"]):
                        is_code = "Mathematical Optimization Model:" in text or "Your task is to strictly follow the User Query" in text
                        if is_code:
                            assert "Observation (complete source CSV data):" not in text
                            marker = "CSVQA_DATA structural schema (no cell values):\n"
                            if marker in text:
                                schema, _ = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())
                                assert all("records" not in t for t in schema["tables"])
                        elif not observed:
                            payload = payload_in_prompt(text)
                            if payload is not None:
                                result["verified_csv_cells"] += check_complete_payload(payload)
                                result["full_observations"] += 1
                                observed = True
                    assert observed or record.get("pipeline_stage") == "classification"
            audit["runtime"][method] = result
    (OUT / ("final_boundary_audit.json" if runtime else "staged_boundary_audit.json")).write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    if publish:
        for name, digest in control["delivered_sha256"].items():
            current = ROOT / name
            baseline = OUT / "baseline" / name
            assert sha(current) in {sha(baseline), digest}, f"Concurrent edits detected: {name}"
        for name, digest in control["delivered_sha256"].items():
            (ROOT / name).write_bytes((staged / name).read_bytes())
            assert sha(ROOT / name) == digest
        (OUT / "delivery_control.json").write_text(json.dumps(control, ensure_ascii=False, indent=2))
    print(json.dumps({"static_boundaries_passed": True, "published": publish, "runtime": audit["runtime"], "verified_csv_cells": source_cells}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    main(args.publish, args.runtime)
