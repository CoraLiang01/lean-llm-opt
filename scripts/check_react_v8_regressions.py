"""Reproduce shared interface failures without API calls or benchmark reruns."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import evaluate_react_revision_20261006 as runner


def load(method):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return runner.namespace(runner.ROOT / runner.NAMES[method], method)


class SharedRegressions(unittest.TestCase):
    def test_non_ascii_business_keys_do_not_break_solver_name_extraction(self):
        ns = load("full")
        source = """import gurobipy as gp
m = gp.Model()
m.Params.OutputFlag = 0
x_vars = m.addVars(['café'], lb=1, name='x')
m.setObjective(x_vars['café'])
m.optimize()
for variable in m.getVars():
    print(variable.VarName, variable.X)
"""
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                objective, solution = ns["execute_code"](source)
        except Exception as exc:
            error = exc
        self.assertIsNone(error, f"Non-ASCII business key escaped into an invalid solver display name: {error}")
        self.assertEqual(objective, 1.0)
        self.assertEqual(len(solution), 1)

    def test_named_variable_lookup_survives_source_processing(self):
        ns = load("full")
        source = """import gurobipy as gp
m = gp.Model()
m.Params.OutputFlag = 0
x_vars = m.addVars(['alpha'], lb=1, name='x')
m.setObjective(x_vars['alpha'])
m.optimize()
assert m.getVarByName('x[alpha]') is not None
"""
        error = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                objective, solution = ns["execute_code"](source)
        except Exception as exc:
            error = exc
        self.assertIsNone(error, f"Source processing broke a valid named-variable lookup: {error}")
        self.assertEqual(objective, 1.0)
        self.assertEqual(solution, [('x[alpha]', 1.0)])

    def test_others_csv_uses_current_csvqa_observation(self):
        ns = load("full")
        query = "Allocate units to maximize value under the supplied capacities."
        plan = {"route": "Others", "tables": [{"file_index": 0, "columns": "*"}],
                "ignored_file_indices": [], "relationships": []}
        llm = FakeListChatModel(responses=[
            "Thought: Read current data.\nAction: CSVQA\nAction Input: " + query,
            json.dumps(plan),
            "Thought: Use current records.\nFinal Answer: ## Variables\nx_i >= 0\n"
            "## Objective\nMaximize sum(v_i*x_i)\n## Constraints\nx_i <= c_i",
        ])
        ns["make_llm"] = lambda **kwargs: llm
        # Replace only external embedding retrieval; CSVQA and its planner execute normally.
        ns["retrieve_rag_examples"] = lambda *args, **kwargs: []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("ID,Value,Capacity\n001,17,10\n")
            with contextlib.redirect_stdout(io.StringIO()):
                result = ns["get_Others_response"](query, str(path))
        self.assertEqual(result["trace"].get("csvqa_call_count", 0), 1)
        observation = json.loads(result["observation"])
        self.assertEqual(observation["tables"][0]["records"][0]["values"],
                         {"ID": "001", "Value": "17", "Capacity": "10"})
        self.assertIn("Maximize", result["formulation"])

    def test_invalid_csvqa_file_indices_restart_protocol(self):
        ns = load("full")
        query = "Choose quantities from current values."
        plan = {"route": "RA", "tables": [{"file_index": 0, "columns": "*"}],
                "ignored_file_indices": [], "relationships": []}
        llm = FakeListChatModel(responses=[
            'Thought: Read.\nAction: CSVQA\nAction Input: {"file_indices":"0"}',
            "Thought: Read.\nAction: CSVQA\nAction Input: " + query,
            json.dumps(plan),
            "Thought: Model.\nFinal Answer: ## Variables\nx_i >= 0\n"
            "## Objective\nMaximize sum(v_i*x_i)\n## Constraints\nx_i <= c_i",
        ])
        ns["make_llm"] = lambda **kwargs: llm
        error = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text("ID,Value,Capacity\n001,17,10\n")
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = ns["formulate_with_csvqa"](query, str(path), "RA", "", "", "", "")
            except Exception as exc:
                error = exc
        self.assertIsNone(error, f"Malformed Action Input incorrectly became a case failure: {error}")
        self.assertEqual(result["trace"]["protocol_retry_count"], 1)
        self.assertEqual(json.loads(result["observation"])["tables"][0]["records"][0]["values"]["ID"], "001")

    def test_few_shot_query_only_keeps_query_data_codegen_contract(self):
        ns = load("few_shot_only")
        prompts = []

        class Client:
            def invoke(self, messages):
                prompts.extend(message.content for message in messages)
                return type("Response", (), {"content": "import gurobipy as gp"})()

        ns["make_llm"] = lambda **kwargs: Client()
        ns["retrieve_rag_examples"] = lambda *args, **kwargs: []
        with contextlib.redirect_stdout(io.StringIO()):
            ns["get_code"]("Minimize x with x >= 2", "Others", original_query="Choose x >= 2.")
        self.assertIn("Define all required data in the code", prompts[0])
        self.assertNotIn("Read the exact source CSV paths in Source Schema", prompts[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="v8")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SharedRegressions)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    target = runner.OUT / f"shared_regression_checks_{args.label}.txt"
    target.write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
