"""CSVQA plan-repair regressions. Fake only the external LLM; execute real CSV plans."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BOOK = Path(os.environ.get("TEST_NOTEBOOK", ROOT / "LEAN_LLM_OPT_4.1_Large-scale.ipynb"))


def load_namespace():
    ns = {"__name__": "plan_repair_test"}
    book = json.loads(BOOK.read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        for i, cell in enumerate(book["cells"]):
            if cell["cell_type"] == "code" and i < 37:
                exec(compile("".join(cell["source"]), f"{BOOK}:cell{i}", "exec"), ns)
    ns["PROJECT_ROOT"] = ROOT
    ns["NOTEBOOK_PATH"] = BOOK
    return ns


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, messages):
        self.requests.append(messages[0].content)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return type("Response", (), {"content": response})()


class PlanRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_namespace()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "products.csv"
        self.path.write_text("Product ID,Profit,Noise\n001,8,99\n002,3,77\n")
        self.query = 'Maximize Profit from products. Only product "002" is eligible.'
        self.plan = {
            "route": "RA", "tables": [{"file_index": 0, "role": "products",
                "columns": ["Product ID", "Profit"], "filters": {"logic": "and", "conditions": [
                    {"column": "Product ID", "operator": "exact", "dtype": "string",
                     "value": "002", "evidence": '"002"'}]}}],
            "ignored_file_indices": [], "relationships": [],
        }

    def invoke(self, responses, enabled=None, route="RA"):
        client = ScriptedLLM(responses)
        replacements = {"make_llm": lambda: client}
        if enabled is not None:
            replacements["CSVQA_REPAIR_PLAN_ON_FAILURE"] = enabled
        with patch.dict(self.ns, replacements):
            _, tool, state = self.ns["build_csvqa_components"](
                str(self.path), "Select necessary columns and explicit subsets.", "CSV evidence",
                route=route, user_query=self.query)
            payload = json.loads(tool.invoke("Get all necessary data"))
        return payload, state["trace"], client.requests

    def test_enabled_repairs_wrong_column_and_preserves_real_subset(self):
        wrong = json.loads(json.dumps(self.plan))
        wrong["tables"][0]["columns"] = ["Product ID", "Missing Profit"]
        bad_output = json.dumps(wrong)
        payload, trace, requests = self.invoke([bad_output, json.dumps(self.plan)], enabled=True)
        self.assertEqual(payload["tables"][0]["records"],
                         [{"source_row": 1, "values": {"Product ID": "002", "Profit": "3"}}])
        self.assertEqual(trace["status"], "PLANNED_REPAIRED")
        self.assertTrue(trace["planner_repair_succeeded"])
        self.assertEqual(trace["planner_attempt_count"], 2)
        self.assertEqual(len(requests), 2)
        self.assertIn(bad_output, requests[1])
        self.assertIn("Unknown columns", requests[1])
        self.assertIn(self.query, requests[1])
        self.assertEqual(len(trace["planner_outputs"]), 2)

    def test_enabled_repairs_invalid_json_for_nrm_too(self):
        self.plan["route"] = "NRM"
        payload, trace, _ = self.invoke(["not JSON", json.dumps(self.plan)], enabled=True, route="NRM")
        self.assertEqual(payload["tables"][0]["records"][0]["values"]["Product ID"], "002")
        self.assertEqual(trace["status"], "PLANNED_REPAIRED")

    def test_default_does_not_repair_and_falls_back_to_complete_source(self):
        payload, trace, requests = self.invoke(["not JSON"])
        self.assertEqual(trace["status"], "FALLBACK_FULL_DATA")
        self.assertEqual(len(requests), 1)
        self.assertEqual(payload["tables"][0]["returned_rows"], 2)
        self.assertEqual(payload["tables"][0]["records"][0]["values"]["Noise"], "99")
        self.assertEqual(payload["tables"][0]["records"][1]["values"]["Product ID"], "002")

    def test_explicit_off_never_sends_a_repair_request(self):
        payload, trace, requests = self.invoke(["not JSON"], enabled=False)
        self.assertEqual(trace["status"], "FALLBACK_FULL_DATA")
        self.assertEqual(len(requests), 1)
        self.assertEqual(payload["tables"][0]["returned_rows"], 2)

    def test_valid_first_plan_never_uses_repair_even_when_enabled(self):
        payload, trace, requests = self.invoke([json.dumps(self.plan)], enabled=True)
        self.assertEqual(trace["status"], "PLANNED")
        self.assertEqual(len(requests), 1)
        self.assertEqual(payload["tables"][0]["returned_rows"], 1)

    def test_two_invalid_plans_fall_back_without_third_attempt(self):
        wrong = json.loads(json.dumps(self.plan))
        wrong["tables"][0]["columns"] = ["Missing ID"]
        payload, trace, requests = self.invoke(["not JSON", json.dumps(wrong)], enabled=True)
        self.assertEqual(len(requests), 2)
        self.assertEqual(trace["status"], "FALLBACK_FULL_DATA")
        self.assertEqual(trace["planner_attempt_count"], 2)
        self.assertFalse(trace["planner_repair_succeeded"])
        self.assertEqual(len(trace["planner_errors"]), 2)
        self.assertEqual(payload["tables"][0]["returned_rows"], 2)
        self.assertEqual(payload["tables"][0]["records"][0]["values"]["Noise"], "99")

    def test_failed_repair_request_also_falls_back_after_client_exhaustion(self):
        payload, trace, requests = self.invoke(["not JSON", RuntimeError("service unavailable")], enabled=True)
        self.assertEqual(len(requests), 2)
        self.assertEqual(trace["status"], "FALLBACK_FULL_DATA")
        self.assertIn("service unavailable", trace["planner_errors"][-1])
        self.assertEqual(payload["tables"][0]["returned_rows"], 2)

    def test_initial_service_failure_is_not_misreported_as_invalid_plan(self):
        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            self.invoke([RuntimeError("service unavailable")], enabled=True)

    def test_runtime_switch_changes_cache_fingerprint(self):
        case = {"Query": self.query, "dataset_address": str(self.path)}
        with patch.dict(self.ns, CSVQA_REPAIR_PLAN_ON_FAILURE=False):
            off = self.ns["case_fingerprint"](case, source_fingerprint="same source")
        with patch.dict(self.ns, CSVQA_REPAIR_PLAN_ON_FAILURE=True):
            on = self.ns["case_fingerprint"](case, source_fingerprint="same source")
        self.assertNotEqual(off, on, "Repair-on and repair-off must not share cached results")

    def test_pipeline_saves_repair_trace_as_a_readable_artifact(self):
        payload, trace, _ = self.invoke(["not JSON", json.dumps(self.plan)], enabled=True)
        formulation = {"formulation": "minimize x subject to x >= 2", "observation": json.dumps(payload), "trace": trace}
        code = "import gurobipy as gp\nm=gp.Model()\nm.Params.OutputFlag=0\nx=m.addVar(lb=2)\nm.setObjective(x)\nm.optimize()"
        case = {"problem_id": "fixture", "Query": self.query, "dataset_address": str(self.path)}
        with patch.dict(self.ns, invoke_formulation=lambda *args: formulation,
                        get_csv_code=lambda *args, **kwargs: code), contextlib.redirect_stdout(io.StringIO()):
            record = self.ns["execute_pipeline_case"](case, forced_route="RA")
        self.assertEqual(record.get("csvqa_planner_attempt_count"), 2)
        csv_path = Path(self.tmp.name) / "results.csv"
        rows = self.ns["save_record"](csv_path, record)
        self.assertIn("csvqa_trace_path", rows[0])
        saved = json.loads((csv_path.parent / rows[0]["csvqa_trace_path"]).read_text())
        self.assertEqual(saved["status"], "PLANNED_REPAIRED")
        self.assertEqual(len(saved["planner_outputs"]), 2)

    def test_trace_count_round_trips_when_other_cases_fail_before_formulation(self):
        csv_path = Path(self.tmp.name) / "results.csv"
        self.ns["save_record"](csv_path, {"problem_id": "completed", "query": "q",
                                           "dataset_address": "", "csvqa_planner_attempt_count": 2})
        self.ns["save_record"](csv_path, {"problem_id": "early_failure", "query": "q",
                                           "dataset_address": "", "record_status": "error"})
        try:
            rows = self.ns["load_records"](csv_path)
        except ValueError as exc:
            self.fail(f"Mixed success/error CSV cannot be resumed: {exc}")
        self.assertEqual(rows[0]["csvqa_planner_attempt_count"], 2)


class BulkNamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_namespace()

    def execute_fixture(self, normalize=None):
        source = '''import gurobipy as gp
m = gp.Model()
m.Params.OutputFlag = 0
x = m.addVars(["001", "002"], name="production")
m.addConstrs((x[i] >= 1 for i in ["001", "002"]), name="minimum")
m.update()
lookup = m.getVarByName("production[001]")
names = [v.VarName for v in m.getVars()]
constraint_names = [c.ConstrName for c in m.getConstrs()]
business_keys = list(x.keys())
'''
        replacements = {} if normalize is None else {"NORMALIZE_GUROBI_BULK_NAMES": normalize}
        with patch.dict(self.ns, replacements):
            rewritten = self.ns["_source_candidate"](source)
        env = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(rewritten, env)
        self.addCleanup(env["m"].dispose)
        return env

    def test_default_preserves_names_and_name_based_lookup(self):
        env = self.execute_fixture()
        self.assertIsNotNone(env["lookup"], "Default postprocessing must not break getVarByName")
        self.assertEqual(env["names"], ["production[001]", "production[002]"])
        self.assertEqual(env["constraint_names"], ["minimum[001]", "minimum[002]"])
        self.assertEqual(env["business_keys"], ["001", "002"])

    def test_enabled_retains_previous_normalization_and_business_keys(self):
        env = self.execute_fixture(normalize=True)
        self.assertNotIn("production[001]", env["names"])
        self.assertNotIn("minimum[001]", env["constraint_names"])
        self.assertEqual(env["business_keys"], ["001", "002"])

    def test_runtime_name_switch_changes_cache_fingerprint(self):
        case = {"Query": "minimize cost", "dataset_address": ""}
        with patch.dict(self.ns, NORMALIZE_GUROBI_BULK_NAMES=False):
            off = self.ns["case_fingerprint"](case, source_fingerprint="same source")
        with patch.dict(self.ns, NORMALIZE_GUROBI_BULK_NAMES=True):
            on = self.ns["case_fingerprint"](case, source_fingerprint="same source")
        self.assertNotEqual(off, on)

    def test_public_codegen_changes_naming_instruction_with_switch(self):
        for enabled, expected in [(False, False), (True, True)]:
            client = ScriptedLLM(["import gurobipy as gp\nm=gp.Model()"])
            with patch.dict(self.ns, NORMALIZE_GUROBI_BULK_NAMES=enabled, make_llm=lambda: client,
                            retrieve_csv_code_example=lambda *args: ""), contextlib.redirect_stdout(io.StringIO()):
                self.ns["_generate_code"]("min x", "RA")
            self.assertEqual("Use name='' for addVars/addConstrs" in client.requests[0], expected)

    def test_others_codegen_changes_naming_instruction_with_switch(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        class RecordingLLM(FakeListChatModel):
            requests: list[str] = []

            def _call(self, messages, stop=None, run_manager=None, **kwargs):
                self.requests.append("\n".join(message.content for message in messages))
                return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("Product,Profit\na,1\n")
            for enabled, expected in [(False, False), (True, True)]:
                client = RecordingLLM(responses=["Abstract plan", "import gurobipy as gp\nm=gp.Model()"])
                with patch.dict(self.ns, NORMALIZE_GUROBI_BULK_NAMES=enabled, make_llm=lambda: client,
                                retrieve_rag_examples=lambda *args, **kwargs: []), contextlib.redirect_stdout(io.StringIO()):
                    self.ns["get_Others_response"]("maximize Profit", str(path))
                self.assertEqual("Use name='' for addVars/addConstrs" in client.requests[-1], expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
