"""Audit current LOTO source using fixtures and mocks; no model or solver calls.

Uses the repository's existing mock namespace, but avoids its stale fixed-cell-count
assertions. Reference Type/prompt fields are reconstructed from saved manifests;
the remaining reference fields are placeholders. This verifies control logic, not
real retrieval rankings, original CSV contents, or saved objective correctness.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import test_loto_notebooks as mocks

NAMES = mocks.NAMES


def main():
    report = {"scope": __doc__, "notebooks": [], "model_handle_checks": {}}
    books = {name: json.loads((ROOT / name).read_text()) for name in NAMES}
    for index in (37, 41):
        assert len({"".join(book["cells"][index]["source"]) for book in books.values()}) == 1
    report["shared_fold_control_and_runner_cells_identical"] = True
    fixtures = {}
    manifest_root = ROOT / "outputs/leave_one_type_out/gpt41/examples_only_V1"
    for path in manifest_root.glob("*/fold_manifest.json"):
        for item in json.loads(path.read_text())["removed_examples"]:
            fixtures[item["row_index"]] = {
                "Type": item["type"], "prompt": item["prompt"],
                "Data_address": "fixture.csv", "Related": "fixture",
                "Required Data": "fixture", "Label": "fixture", "Code": "fixture",
            }
    assert len(fixtures) == 15
    with tempfile.TemporaryDirectory() as temporary:
        fixture_path = Path(temporary) / "reference_fixture.csv"
        mocks.pd.DataFrame([fixtures[i] for i in sorted(fixtures)]).to_csv(fixture_path, index=False)
        for name, book in books.items():
            book["_filename"] = name
            for i, cell in enumerate(book["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{name}:cell{i}", "exec")
            ns = mocks.namespace(book, Path(temporary) / name)
            ns["RAG_EXAMPLES_ALL_PATH"] = str(fixture_path)
            fold_checks = []
            fingerprints = []
            for heldout in ns["CLASS_LABELS"]:
                manifest = ns["set_loto_fold"](heldout)
                labels = ns["loto_reference_frame"]().Type.map(ns["normalize_problem_class"])
                assert not labels.eq(heldout).any()
                assert manifest["reference_count_after"] == 15 - {"Mixture": 8, "Others": 2}.get(heldout, 1)
                if heldout == "Mixture":
                    assert labels.eq("Others").sum() == 2
                if heldout == "Others":
                    assert labels.eq("Mixture").sum() == 8
                docs = ns["_get_classifier_retriever"]().docs
                assert len(docs) == len(labels)
                assert len(ns["_get_classifier_retriever"]().invoke("fixture query")) == 5
                removed_prompts = {item["prompt"] for item in manifest["removed_examples"]}
                for doc in docs:
                    assert not any(doc.page_content.startswith("prompt: " + prompt.strip() + "\n")
                                   for prompt in removed_prompts)
                if heldout in {"TP", "NRM", "RA", "FLP", "AP"}:
                    if "GPT4.1" in name:
                        assert ns["retrieve_rag_examples"](heldout, "fixture query", 3) == []
                        assert ns["retrieve_csv_code_example"](heldout, "fixture query") == ""
                    else:
                        assert ns["get_route_retriever"](heldout, 3).invoke("fixture query") == []
                if heldout in {"Mixture", "Others"}:
                    count = 2 if heldout == "Mixture" else 8
                    if "GPT4.1" in name:
                        assert len(ns["load_rag_examples"]("Others")) == count
                    else:
                        assert len(ns["get_others_store"]().docs) == count
                ns["_test_label"] = ns["loto_allowed_labels"]()[0]
                with contextlib.redirect_stdout(io.StringIO()):
                    ns["invoke_classifier"]("fixture query only, without a gold label")
                    mocks.check_classifier_prompts(ns, "gpt_oss_20b" in name)
                visits = []
                ns["_LOTO_BASE_INVOKE_FORMULATION"] = lambda *args: visits.append(args)
                disabled = manifest["disabled_workflow_route"]
                if disabled:
                    try:
                        ns["invoke_formulation"](disabled, "q", "fixture.csv")
                    except ns["DisabledRouteError"]:
                        pass
                    else:
                        raise AssertionError("Disabled route was executed")
                    assert not visits
                    if disabled == "Others":
                        assert not {"Mixture", "Others"}.intersection(ns["loto_allowed_labels"]())
                for route in manifest["allowed_routes"]:
                    ns["invoke_formulation"](route, "q", "fixture.csv")
                assert len(visits) == len(manifest["allowed_routes"])
                fingerprints.append(ns["case_fingerprint"]({"Query": "same q", "dataset_address": ""},
                                                          source_fingerprint="same fixture source"))
                fold_checks.append({"held_out": heldout, "remaining": len(labels),
                                    "disabled_route": disabled, "allowed_labels": ns["loto_allowed_labels"]()})
            assert len(set(fingerprints)) == 7
            report["notebooks"].append({"name": name, "syntax": "PASS", "fold_controls": "PASS",
                                        "classifier_repairs": "PASS", "cache_fold_isolation": "PASS",
                                        "folds": fold_checks})
            model_result = {}

            class FakeModel:
                Status = 2
                ObjVal = 1.0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def getVars(self):
                    return []

            ns["gp"] = SimpleNamespace(Model=FakeModel, GRB=SimpleNamespace(OPTIMAL=2), setParam=lambda *a: None)
            for scenario, data in {
                "m_is_model": {"m": FakeModel()},
                "m_none_model_is_model": {"m": None, "model": FakeModel()},
                "alternate_global_model_name": {"solver_model": FakeModel()},
                "m_none_no_exposed_model": {"m": None},
            }.items():
                # Intercept execution: only simulate its namespace, no generated code executes.
                ns["exec"] = lambda code, namespace, data=data: namespace.update(data)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ns["execute_code"]("pass")
                    model_result[scenario] = "accepted"
                except Exception as error:
                    model_result[scenario] = f"{type(error).__name__}: {error}"
            report["model_handle_checks"][name] = model_result
            assert model_result["m_is_model"] == "accepted"
            assert model_result["m_none_no_exposed_model"] != "accepted"
            if "GPT4.1" in name:
                assert model_result["m_none_model_is_model"] != "accepted"
            else:
                assert model_result["m_none_model_is_model"] == "accepted"
                assert model_result["alternate_global_model_name"] == "accepted"
    report["base_notebook_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("LEAN_LLM_OPT_4.1_Large-scale.ipynb", "LEAN_LLM_OPT_gpt_oss_20b_Large-scale.ipynb")
    }
    path = Path(__file__).with_name("offline_controls_result.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PASS: 4 notebooks, 28 fold configurations, filtered reference access, allowed-label corrections,")
    print("route enforcement, seven-fold cache identity, and mocked model-handle regression checks.")
    print(path)


if __name__ == "__main__":
    main()
