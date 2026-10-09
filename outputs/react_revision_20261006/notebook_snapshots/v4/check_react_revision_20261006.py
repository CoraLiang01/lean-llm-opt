"""Offline ReAct protocol and Few-shot Only transfer checks; zero API calls."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile

import evaluate_react_revision_20261006 as runner
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.callbacks import BaseCallbackHandler


class Calls(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
        self.prompts = []

    def on_llm_start(self, *args, **kwargs):
        self.count += 1
        self.prompts.extend(args[1])


def function_source(book, name):
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return ast.get_source_segment(source, node)
    raise ValueError(name)


def main():
    outcomes = []
    query = "Choose quantities using the current costs and capacities."
    action = "Thought: I need current data.\nAction: CSVQA\nAction Input: " + query
    final = "Thought: I can now model the problem.\nFinal Answer: ## Variables\nx_i >= 0\n## Objective\nMaximize sum(p_i*x_i)\n## Constraints\nsum(a_i*x_i) <= B"
    full = json.loads((runner.ROOT / runner.NAMES["full"]).read_text())
    for method in runner.NAMES:
        path = runner.ROOT / runner.NAMES[method]
        book = json.loads(path.read_text())
        if method != "few_shot_only":
            assert function_source(full, "formulate_with_csvqa") == function_source(book, "formulate_with_csvqa")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ns = runner.namespace(path, method)
        if method == "few_shot_only":
            tracker = Calls()
            llm = FakeListLLM(responses=[final], callbacks=[tracker])
            ns["make_llm"] = lambda: llm
            with tempfile.TemporaryDirectory() as directory:
                csv = Path(directory) / "source.csv"
                csv.write_text("ID,Cost,Capacity\n001,17,\n002,23,10\n")
                original = ns["direct_source_observation"](str(csv))
                assert json.loads(original)["tables"][0]["records"] == [
                    {"ID": "001", "Cost": "17", "Capacity": ""},
                    {"ID": "002", "Cost": "23", "Capacity": "10"}]
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ns["formulate_with_csvqa"](query, str(csv), "RA", "", "", "Historical example.", "")
                assert result["observation"] == original
                assert tracker.count == 1 and result["trace"]["csvqa_call_count"] == 0
                ns["_CURRENT_DATASET_ADDRESS"] = str(csv)
                forwarded = []
                ns["_generate_code"] = lambda *args: forwarded.append(args) or "code"
                model = "## Parameters\nCost=[17,23]\n|ID|Cost|\n|001|17|\n## Variables\nx_i >= 0\n## Objective\nMaximize sum(p_i*x_i)\n## Constraints\nsum(a_i*x_i)<=B"
                ns["get_csv_code"](model, "RA", query)
                assert len(forwarded) == 1
                assert "Cost=[17,23]" not in forwarded[0][0] and "|001|17|" not in forwarded[0][0]
                assert "records" not in forwarded[0][2] and "001" not in forwarded[0][2]
                assert "source.csv" in forwarded[0][2]
            outcomes.append({"method": method, "protocol": "ReAct, no tools", "model_calls": 1,
                             "complete_source_values_preserved": True, "full_data_to_codegen": False})
            continue
        for scenario, responses, expected_calls in [("one_tool_then_model", [action, final], 2),
                ("skipped_tool_rejected", [final], 1),
                ("second_tool_request_blocked", [action, action, final], 2)]:
            tracker = Calls()
            llm = FakeListLLM(responses=responses, callbacks=[tracker])
            state, tools = {}, []
            def extract(tool_query):
                tools.append(tool_query)
                state.update(observation='{"tables":[{"records":[{"values":{"ID":"001","Cost":"17"}}]}]}',
                             trace={"status": "PLANNED", "planner_attempt_count": 1, "fallback_count": 0})
                return state["observation"]
            def components(*args, **kwargs):
                return llm, ns["Tool"](name="CSVQA", func=extract, description="Return current source data."), state
            ns["build_csvqa_components"] = components
            caught = None
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    result = ns["formulate_with_csvqa"](query, "fixture.csv", "RA", "", "", "Historical example.", "")
                except RuntimeError as exc:
                    caught = exc
            if scenario == "one_tool_then_model":
                assert caught is None and result["trace"]["csvqa_call_count"] == 1
                assert len(tools) == 1 and "Variables" in result["formulation"]
                assert "CURRENT CSVQA DATA STATE: NOT_LOADED" in tracker.prompts[0]
                assert "CURRENT CSVQA DATA STATE: READY" in tracker.prompts[1]
            else:
                assert caught is not None
                assert len(tools) == (0 if scenario == "skipped_tool_rejected" else 1)
            assert tracker.count == expected_calls, (method, scenario, tracker.count)
            outcomes.append({"method": method, "scenario": scenario, "llm_calls": tracker.count,
                             "actual_data_tool_calls": len(tools), "repair_or_rerun": False})
    (runner.OUT / "offline_react_protocol_checks.json").write_text(json.dumps({"api_calls": 0, "checks": outcomes}, indent=2))
    print(json.dumps({"status": "PASS", "api_calls": 0, "checks": len(outcomes)}, indent=2))


if __name__ == "__main__":
    main()
