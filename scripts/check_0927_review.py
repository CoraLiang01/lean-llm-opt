"""Offline regression checks and a locked, dataset-wise comparison gate."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import nbformat
import pandas as pd

from evaluate_0927_optimization import namespace, frames

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review_0927_20261006"
RUNS = ROOT / "outputs/optimization_0927_20261006"
BASE = OUT / "baseline/LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb"
CANDIDATE = OUT / "candidates/full_review_v2.ipynb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def must_raise(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def checks():
    nbformat.validate(nbformat.read(CANDIDATE, as_version=4))
    old, ns = namespace(BASE), namespace(CANDIDATE)
    has_v3_fixes = 'compatibility_fixes' in nbformat.read(CANDIDATE, as_version=4).metadata['review_candidate']
    checked = []

    def passed(name):
        checked.append(name)

    with tempfile.TemporaryDirectory(prefix="review0927-") as temp:
        root = Path(temp)
        data = root / "matrix.csv"
        data.write_text('Unnamed: 0,A-B,AB,Blank\n0,3,4,\n1,5,6,\n')
        original = pd.read_csv(data, dtype=str, keep_default_na=False)
        assert "Unnamed: 0" not in old["_read_csv"](data)
        pd.testing.assert_frame_equal(ns["_read_csv"](data), original)
        pd.testing.assert_frame_equal(ns["_read_csv"](data), original)
        passed("Sequential index-like IDs and blank columns preserved; baseline bug reproduced")
        docs = ns["load_csv_documents"](str(data))
        assert [json.loads(d.page_content)["values"] for d in docs] == original.to_dict("records")
        passed("Legacy source documents preserve every parsed cell and source order")
        if 'validate_legacy_source_rows' in ns:
            raw = [json.loads(d.page_content) for d in docs]
            assert ns['validate_legacy_source_rows'](json.dumps(raw[:1]), docs) == json.dumps(raw[:1])
            altered = json.loads(json.dumps(raw)); altered[0]['values']['A-B'] = '999'
            must_raise(lambda: ns['validate_legacy_source_rows'](json.dumps(altered),docs))
            derived = json.loads(json.dumps(raw)); derived[0]['values']['invented_join'] = 'qualified'
            must_raise(lambda: ns['validate_legacy_source_rows'](json.dumps(derived),docs))
            must_raise(lambda: ns['validate_legacy_source_rows'](json.dumps(raw + raw[:1]),docs))
            passed('Legacy source validator rejects altered cells, derived fields and extra duplicates; valid source subsets accepted')
            from langchain_core.language_models.fake import FakeListLLM
            make = ns['make_llm']
            try:
                ns['make_llm'] = lambda *a,**kw: FakeListLLM(responses=[json.dumps(derived)])
                _, tool, sink = ns['build_csvqa_components'](str(data),'Original data: {context}','retrieve',route='AP',user_query='use all rows')
                with contextlib.redirect_stdout(io.StringIO()):
                    returned = tool.invoke('read all original source rows')
                assert json.loads(returned) == raw
                assert sink['trace']['status'] == 'LEGACY_SOURCE_VALIDATION_FALLBACK'
                assert sink['trace']['legacy_extraction_output'] == json.dumps(derived)
                assert sink['trace']['fallback_reason'] and sink['trace']['planner_attempt_count'] == 0
            finally:
                ns['make_llm'] = make
            passed('Invalid legacy extraction recovers exact complete source evidence with rejection reason and raw response retained; zero API calls')
        if has_v3_fixes:
            current = Path.cwd()
            try:
                os.chdir(root)
                ns["_load_rag_table"].cache_clear()
                assert len(ns["_load_rag_table"]()) > 0
                ns["PROJECT_ROOT"] = root
                text = ns["rag_example_observation"]({"Required Data": "", "Data_address": "matrix.csv"})
                assert '"A-B":"3"' in text
            finally:
                ns["PROJECT_ROOT"] = ROOT
                os.chdir(current)
            passed("Example paths resolve from PROJECT_ROOT even when the notebook working directory differs")

        series = pd.Series(["1", "2"], name="value")
        must_raise(lambda: old["_apply_condition"](series, {"operator": "prefix", "dtype": "number", "value": 1}), AttributeError)
        must_raise(lambda: ns["_apply_condition"](series, {"operator": "prefix", "dtype": "number", "value": 1}))
        assert old["_apply_condition"](series, {"operator": "eq", "value": ["1", "2"]}).all()
        must_raise(lambda: ns["_apply_condition"](series, {"operator": "eq", "value": ["1", "2"]}))
        assert ns["_apply_condition"](series, {"operator": "in", "value": ["1"]}).tolist() == [True, False]
        passed("Malformed scalar/string predicates rejected, supported set predicates unchanged")

        tables = [{"file_index": 0, "path": data, "frame": original}]
        malformed = {"route": "NRM", "tables": ["bad"]}
        must_raise(lambda: old["_execute_plan"](malformed, tables, "use all", "NRM"), AttributeError)
        must_raise(lambda: ns["_execute_plan"](malformed, tables, "use all", "NRM"))
        passed("Non-object extraction table rejected as a validation error")
        plan = {"route": "NRM", "tables": [{"file_index": 0, "columns": "*", "filters": {"conditions": [
            {"column": "Unnamed: 0", "operator": "eq", "value": "-1", "evidence": "-1"}]}}]}
        negative = [{"file_index": 0, "path": data, "frame": pd.DataFrame({"Unnamed: 0": ["-1", "1"]})}]
        assert old["_execute_plan"](plan, negative, "select 1", "NRM")["tables"][0]["records"][0]["values"]["Unnamed: 0"] == "-1"
        must_raise(lambda: ns["_execute_plan"](plan, negative, "select 1", "NRM"))
        passed("Query evidence preserves numerical signs; fabricated negative filter rejected")
        if has_v3_fixes:
            quote_plan = {"route": "NRM", "tables": [{"file_index": 0, "filters": {"conditions": [
                {"column": "ID", "operator": "prefix", "value": ["Alpha_"], "evidence": '\"Alpha\" products'}]}}]}
            quote_tables = [{"file_index": 0, "path": data, "frame": pd.DataFrame({"ID": ["Alpha_one", "Beta"]})}]
            quote_result = ns["_execute_plan"](quote_plan, quote_tables, 'sell “Alpha” products', "NRM")
            assert quote_result["tables"][0]["returned_rows"] == 1
            passed("Typographic query quotes and unambiguous singleton scalar arrays accepted without weakening sign checks")

        entities = pd.DataFrame({"id": ["AB", "A-B"]})
        matrix = pd.DataFrame({"id": ["A-B", "AB"], "A-B": ["1", "2"], "AB": ["3", "4"]})
        aligned = [{"file_index": i, "path": data, "frame": df} for i, df in enumerate([entities, matrix])]
        matrix_plan = {"route": "NRM", "tables": [{"file_index": 0}, {"file_index": 1}], "relationships": [{"type": "matrix", "matrix_table_id": "file_1_view_0", "row_id_column": "id", "row_axis": {"table_id": "file_0_view_0", "id_column": "id"}, "column_axis": {"table_id": "file_0_view_0", "id_column": "id"}}]}
        must_raise(lambda: old["_execute_plan"](matrix_plan, aligned, "all entities", "NRM"))
        payload = ns["_execute_plan"](matrix_plan, aligned, "all entities", "NRM")
        assert payload["validation"]["matrix_checks"][0]["row_ids_aligned"]
        assert not payload["validation"]["matrix_checks"][0]["row_order_matches"]
        assert payload["tables"][1]["records"][0]["values"]["id"] == "A-B"
        passed("Matrix IDs retain punctuation; differing source axis order is validated without reordering")

        family = pd.DataFrame({"Category": ["Alpha fruit", "Alpha grain", "Beta"], "value": ["3", "5", "7"]})
        profile = ns["_build_profile"]([{"file_index": 0, "path": data, "frame": family}], 'choose the "Alpha" family')
        evidence = profile[0]["query_terms"]["Alpha"]["value_matches"]["Category"]
        assert evidence["exact"] == 0 and evidence["prefix"] == 2
        assert evidence["matching_values"] == ["Alpha fruit", "Alpha grain"]
        passed("Planned profile provides actual category labels alongside matching counts")
        observation = json.dumps([{"source": "file.csv", "values": {"A": "1", "A ": "2", "001": "03"}}])
        assert len(old["_legacy_records_from_observation"](observation)[0]["values"]) == 2
        assert ns["_legacy_records_from_observation"](observation)[0]["values"] == {"A": "1", "A ": "2", "001": "03"}
        must_raise(lambda: ns["_legacy_records_from_observation"]("A,A\n1,2\n"))
        must_raise(lambda: ns["_legacy_records_from_observation"]("| A | A |\n| - | - |\n| 1 | 2 |"))
        passed("Legacy parser preserves literal column keys and rejects duplicate table headers")

        cases = pd.DataFrame({"Query": ["test", "test"], "problem_id": ["x", "x"]})
        must_raise(lambda: ns["prepare_cases"](cases))
        must_raise(lambda: ns["prepare_cases"](pd.DataFrame({"Query": []})))
        passed("Duplicate/empty case inventories rejected before requests")
        csv = root / "results.csv"
        ns["save_record"](csv, {"problem_id": "x", "query": "test", "record_status": "error", "final_ok": False, "final_solution": None, "external_deadline_exceeded": False, "route_allowed": False, "cache_fingerprint": "same"})
        row = ns["load_records"](csv)[0]
        assert row["external_deadline_exceeded"] is False and row["route_allowed"] is False
        assert ns["reusable_record"](csv, row, "same")["record_status"] == "error"
        assert ns["reusable_record"](csv, row, "changed") is None
        passed("Failed partial attempts resume without rerunning; Boolean false flags parse correctly")

        # Exercise the runner's immutable-round guards without any model/API calls.
        case = ns["prepare_cases"](pd.DataFrame({"Query": ["test"], "problem_id": ["x"]})).iloc[0].to_dict()
        calls = []
        def fake(case, **kwargs):
            calls.append(1)
            raise RuntimeError("synthetic failure")
        ns["execute_pipeline_case"] = fake
        ns["require_api_key"] = lambda: "synthetic"
        ns["_source_fingerprint"] = lambda: "source"
        output = root / "runner.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            ns["run_test"](pd.DataFrame([case]), output_csv=output)
            ns["run_test"](pd.DataFrame([case]), output_csv=output)
        assert len(calls) == 1
        must_raise(lambda: ns["run_test"](pd.DataFrame([case]), output_csv=output, reuse_completed=False))
        ns["_source_fingerprint"] = lambda: "changed"
        must_raise(lambda: ns["run_test"](pd.DataFrame([case]), output_csv=output))
        passed("Runner retains failure and blocks overwrite/stale-source reuse; zero network calls")
        if has_v3_fixes:
            query_ns = namespace(CANDIDATE)
            query_ns["invoke_classifier"] = lambda query: {"normalized_label": "TP"}
            query_ns["invoke_formulation"] = lambda *a: {"formulation": "synthetic query-only model", "trace": {"status": "NOT_APPLICABLE"}}
            chosen = []
            def code(model, route, **kwargs):
                chosen.append(route)
                return "m = None"
            query_ns["get_code"] = code
            query_ns["execute_code"] = lambda code: (1.0, [])
            result = query_ns["execute_pipeline_case"]({"problem_id": "query-only", "Query": "synthetic", "dataset_address": ""})
            assert result["assigned_route"] == chosen[0] == "Others"
            query_ns["invoke_classifier"] = lambda query: (_ for _ in ()).throw(KeyboardInterrupt())
            try:
                query_ns["execute_pipeline_case"]({"problem_id": "interrupt", "Query": "synthetic", "dataset_address": ""})
            except KeyboardInterrupt as exc:
                assert exc.pipeline_context["pipeline_stage"] == "classification"
            else:
                raise AssertionError("Interrupt was swallowed")
            passed("Query-only formulation/code/record routes agree; interrupted stages retain their diagnostic context")

    if 'QUERY_SEMANTICS_GUIDANCE' in ns:
        text = ns['QUERY_SEMANTICS_GUIDANCE']
        assert 'L*selected <= quantity' in text and 'member_flag <= group_flag' in text
        assert 'original query is authoritative' in text and 'lazy constraints' in text
        assert text in ns['CSV_SOLVER_INSTRUCTIONS']
        assert 'EVERY applicable table' in ns['SOURCE_TABLE_GUIDANCE']
        # Synthetic truth tables exercise the stated links without model or API calls.
        for a in [0, 1]:
            for b in [0, 1]:
                possible = [g for g in [0, 1] if a <= g and b <= g and g <= a+b]
                assert possible == [int(bool(a or b))]
        for x in range(8):
            possible = [y for y in [0, 1] if 3*y <= x <= 6*y]
            assert bool(possible) == (x == 0 or 3 <= x <= 6)
        passed('Generic optional-lot and activation contracts; all-table preprocessing order; no model/API calls')

    # Recreate the namespace after mocks, validate ALL 452 inputs and lock baseline.
    ns = namespace(CANDIDATE)
    inventory = frames(ns, ["main", "variants", "columns"])
    for frame in inventory.values():
        assert frame["label_objective"].notna().all()
        assert frame["label_objective"].map(lambda v: ns["math"].isfinite(v)).all()
    passed("All 452 unique benchmark cases have present files and finite reference objectives")
    baseline_manifest = json.loads((RUNS / "full_v1/run_manifest.json").read_text())
    assert all(sha(path) == digest for path, digest in baseline_manifest["input_sha256"].items())
    baseline_progress = json.loads((RUNS / "full_v1/progress.json").read_text())
    manifest = {"current_baseline": "full_v1", "baseline_progress": baseline_progress,
                "baseline_run_manifest_sha256": sha(RUNS / "full_v1/run_manifest.json"),
                "baseline_notebooks": {str(p): sha(p) for p in sorted((OUT / "baseline").glob("*.ipynb"))
                                       if "LOTO_Examples_Only" not in p.name},
                "input_sha256": baseline_manifest["input_sha256"],
                "gate": "All 11 cohorts complete; Objective match and Solved >= current full_v1 in EVERY cohort; Variants >=32 and each column sheet >=31; no partial/subset selection."}
    path = OUT / "baseline_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    result = {"candidate": str(CANDIDATE), "sha256": sha(CANDIDATE), "passed": checked,
              "count": len(checked), "network_calls": 0}
    (OUT / f"offline_checks_{CANDIDATE.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def gate(version):
    lock = json.loads((OUT / "baseline_manifest.json").read_text())
    assert all(sha(path) == digest for path, digest in lock["input_sha256"].items())
    run = RUNS / version
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert sha(run / "frozen_notebook.ipynb") == manifest["notebook_sha256"]
    rows = json.loads((run / "progress.json").read_text())
    assert manifest["method"] == "full"
    assert manifest["input_sha256"] == lock["input_sha256"]
    ns = namespace(run / "frozen_notebook.ipynb")
    inventory = frames(ns, ["main", "variants", "columns"])
    record_paths = list((run / "attempts").rglob("result.json"))
    records = [json.loads(p.read_text()) for p in record_paths]
    provenance = manifest.get('cohort_provenance')
    if provenance:
        source_by_cohort = {row['benchmark']: Path(row['source_run']) for row in provenance}
        assert len(source_by_cohort) == len(provenance) == len(inventory)
        assert set(source_by_cohort) == set(inventory)
        for source_run in set(source_by_cohort.values()):
            source_manifest = json.loads((source_run/'run_manifest.json').read_text())
            for field in ['notebook_sha256', 'model', 'embedding_model', 'csvqa_modes',
                          'rel_tol', 'abs_tol', 'external_deadline_seconds', 'sdk_max_retries']:
                assert source_manifest[field] == manifest[field], f'Changed batch setting: {field}'
            assert sha(source_run/'frozen_notebook.ipynb') == manifest['notebook_sha256']
            assert all(manifest['input_sha256'][p] == digest for p,digest in source_manifest['input_sha256'].items())
        for path,record in zip(record_paths,records):
            origin = json.loads((path.parent/'record_origin.json').read_text())
            expected_path = source_by_cohort[record['benchmark']]/'attempts'/record['problem_id']/'result.json'
            assert Path(origin['original_result']).resolve() == expected_path.resolve()
            assert sha(path) == origin['original_sha256'] == sha(expected_path)
            assert 'credit_balance_exhausted' not in str(record.get('execution_error',''))
    assert len({r["problem_id"] for r in records}) == len(records)
    raw = {r["problem_id"]: r for r in records}
    for name, frame in inventory.items():
        expected = set(frame["problem_id"])
        actual = {r["problem_id"] for r in records if r["benchmark"] == name}
        assert actual == expected, f"Incomplete or changed case selection: {name}"
        for case in frame.to_dict("records"):
            record = raw[case["problem_id"]]
            assert record["notebook_source_sha256"] == manifest["notebook_sha256"]
            assert record["query"] == case["Query"]
            assert record["dataset_address"] == case["dataset_address"]
            assert record["label_objective"] == case["label_objective"]
            expected_score = ns["finalize_record"](record)["solution_correct"]
            assert expected_score == record["solution_correct"]
        saved = next(r for r in rows if r["benchmark"] == name)
        assert saved["evaluated"] == len(actual)
        assert saved["objective_match"] == sum(raw[id]["solution_correct"] is True for id in actual)
        assert saved["solved"] == sum(raw[id]["final_ok"] is True for id in actual)
    baseline = {r["benchmark"]: r for r in lock["baseline_progress"]}
    assert set(baseline) == {r["benchmark"] for r in rows}
    comparison = []
    for row in rows:
        old = baseline[row["benchmark"]]
        complete = row["evaluated"] == row["expected"] == old["expected"]
        threshold = 32 if row["benchmark"] == "variants" else 31 if row["benchmark"] != "main" else 0
        obj_delta = row["objective_match"] - old["objective_match"]
        solved_delta = row["solved"] - old["solved"]
        comparison.append({**row, "baseline_objective_match": old["objective_match"], "baseline_solved": old["solved"],
                           "objective_delta": obj_delta, "solved_delta": solved_delta,
                           "nondecreasing_objective": obj_delta >= 0,
                           "passed": complete and row["objective_match"] >= threshold})
    baseline_objective = sum(r["objective_match"] for r in baseline.values())
    current_objective = sum(r["objective_match"] for r in rows)
    baseline_solved = sum(r["solved"] for r in baseline.values())
    current_solved = sum(r["solved"] for r in rows)
    # User steering: compare the complete aggregate; individual cohorts may decline.
    passed = all(r["passed"] for r in comparison) and current_objective > baseline_objective
    result = {"version": version, "full_notebook_sha256": manifest["notebook_sha256"],
              "policy": "User updated gate: overall Objective match improvement; no per-cohort nondecrease requirement; original target thresholds retained; Solved reported separately.",
              "baseline_objective": baseline_objective, "current_objective": current_objective,
              "baseline_solved": baseline_solved, "current_solved": current_solved,
              "passed": passed, "comparison": comparison,
              "cohort_provenance_verified": bool(provenance),
              "ablation_and_loto_allowed": passed}
    (OUT / f"gate_{version}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate")
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.candidate:
        CANDIDATE = args.candidate.resolve()
    if args.gate:
        raise SystemExit(0 if gate(args.gate)["passed"] else 2)
    checks()
