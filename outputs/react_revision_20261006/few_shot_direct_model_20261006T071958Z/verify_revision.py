"""Offline checks of the revised Few-shot Only handoff; no API calls."""
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path

import nbformat
import pandas as pd
from langchain_core.language_models.fake import FakeListLLM
from langchain_classic.chains import LLMChain

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = ROOT / "Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb"
FULL = ROOT / "LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb"
nb = json.loads(TARGET.read_text())
full = json.loads(FULL.read_text())
before = json.loads((HERE / TARGET.name).read_text())
manifest = json.loads((HERE / "manifest.json").read_text())
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


def functions(notebook):
    result = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.FunctionDef):
                    result[node.name] = node
    return result


funcs = functions(nb)
base_funcs = functions(full)
nbformat.validate(nb)
checks.append("Notebook schema and every code cell parse successfully")
check("No symbolic-model filter remains", "symbolic_model_for_codegen" not in funcs
      and not any("symbolic_model_for_codegen" in "".join(c["source"]) for c in nb["cells"]))
allowed = {"get_NRM_response", "get_RA_response", "get_TP_response", "get_AP_response",
           "get_FLP_response", "formulate_with_csvqa", "get_Others_response", "get_csv_code",
           "execute_pipeline_case", "classification_for_case", "csv_schema_preview"}
differences = {name for name in funcs.keys() & base_funcs.keys()
               if ast.dump(funcs[name], include_attributes=False)
               != ast.dump(base_funcs[name], include_attributes=False)}
check("Shared function changes are limited to data-source/handoff and cached classification", differences == allowed)
for name in ["build_formulation_examples", "retrieve_rag_examples", "retrieve_csv_code_example",
             "_generate_code", "get_others_without_CSV_response"]:
    check(name + " is unchanged from full v7", ast.dump(funcs[name]) == ast.dump(base_funcs[name]))
for path, digest in manifest["before_sha256"].items():
    if Path(path) != TARGET:
        check(Path(path).name + " byte hash unchanged", hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest)
check("Model and runtime configuration unchanged except results directory",
      "".join(nb["cells"][3]["source"]) == "".join(before["cells"][3]["source"]).replace(
          '"outputs/react_revision_20261006/few_shot_only_v7"',
          '"' + manifest["results_directory"] + '"'))
check("Notebook text remains English", not any(re.search(r"[\u4e00-\u9fff]", "".join(c["source"])) for c in nb["cells"]))

captured = []
env = {"json": json, "read_csv_compat": pd.read_csv,
       "normalize_data_address": lambda address: str(address),
       "normalize_route": lambda route: route,
       "REACT_PROTOCOL_EVENTS": [], "LLMChain": LLMChain,
       "make_llm": lambda: FakeListLLM(responses=["unused"]),
       "react_protocol_record_fields": lambda: {},
       "_generate_code": lambda *args, **kwargs: captured.append((args, kwargs)) or "generated",
       "escape_braces": lambda text: text.replace("{", "{{").replace("}", "}}")}
for name in ["direct_source_observation", "source_schema_only", "get_csv_code", "formulate_with_csvqa"]:
    exec(compile(ast.Module(body=[funcs[name]], type_ignores=[]), str(TARGET), "exec"), env)

prompts = []
model = "Decision Variables:\nx >= 0\nParameters:\np comes from Value\nObjective:\nmax p*x\nSubject to:\nx <= 3\nData Mapping:\np: source.csv, Value\n"


def protocol(factory, query, route):
    executor = factory()
    check("ReAct agent has no current-case tools: " + route, executor.tools == [] and executor.agent.allowed_tools == [])
    prompt = executor.agent.llm_chain.prompt.format(input=query, agent_scratchpad="")
    prompts.append(prompt)
    check("Complete Python Observation reaches modeler: " + route, observation in prompt)
    check("Shared modeling guidance retained: " + route,
          "Preserve all variable domains, constraints, objective sense and additive constants." in prompt)
    check("No active CSVQA call requirement: " + route,
          "call CSVQA at least once before the Final Answer" not in prompt)
    return {"output": model, "intermediate_steps": []}


env["invoke_react_protocol"] = protocol
with tempfile.TemporaryDirectory() as temporary:
    paths = []
    frames = [pd.DataFrame({"ID": ["001", "003"], "Value": ["987654321.123456789", ""],
                            "Notes": ["{literal}", "UNIQUE_SOURCE_SENTINEL"]}),
              pd.DataFrame({"Capacity": ["0005"], "Unused": ["keep_this_field"]})]
    for index, frame in enumerate(frames):
        path = Path(temporary) / f"data_{index}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))
    address = "\n".join(paths)
    env["_CURRENT_DATASET_ADDRESS"] = address
    observation = env["direct_source_observation"](address)
    blocks = json.loads(observation)["tables"]
    check("All files, rows, columns, empty values and text identifiers preserved",
          len(blocks) == 2 and all(block["records"] == frame.to_dict("records")
                                  and block["columns"] == list(frame.columns)
                                  for block, frame in zip(blocks, frames)))
    check("Observation and source schema use matching source identifiers",
          [b["table_id"] for b in blocks] == [b["table_id"] for b in json.loads(env["source_schema_only"](address))])
    for route in ["NRM", "RA", "TP", "AP", "FLP"]:
        result = env["formulate_with_csvqa"]("current query", address, route, "unused", "unused", "Shared route examples. ", "unused")
        check("Unchanged returned formulation: " + route, result["formulation"] == model)
        check("No CSVQA/planner/repair: " + route, all(result["trace"][key] == 0
              for key in ["csvqa_call_count", "planner_attempt_count", "repair_count"]))
        env["get_csv_code"](model, route, "current query")
        args, kwargs = captured[-1]
        check("Complete model passes unchanged to codegen: " + route, args[0] == model)
        check("No separate Observation sent to codegen: " + route,
              "UNIQUE_SOURCE_SENTINEL" not in str((args, kwargs)) and "987654321.123456789" not in str((args, kwargs)))
    example = "HISTORICAL_EXAMPLE_BEGIN\nAction: CSVQA\nObservation: {historical data}\nHISTORICAL_EXAMPLE_END"
    env.update(build_formulation_examples=lambda *args, **kwargs: example,
               CSVQA_PLANNED_PROMPTS={r: "unused" for r in ["NRM", "RA"]},
               CSVQA_TOOL_DESCRIPTIONS={r: "unused" for r in ["NRM", "RA"]},
               CSVQA_LEGACY_SYSTEM_PROMPT="unused", CSVQA_LEGACY_TOOL_DESCRIPTION="unused")
    for route in ["NRM", "RA", "TP", "AP", "FLP"]:
        name = "get_" + route + "_response"
        exec(compile(ast.Module(body=[funcs[name]], type_ignores=[]), str(TARGET), "exec"), env)
        env[name]("current query", address)
        prompt = prompts[-1]
        check("Historical few-shot example unchanged: " + route, example in prompt)
        active = prompt.replace(example, "")
        check("No contradictory current-case CSVQA instruction: " + route,
              not re.search(r"(?:call CSVQA (?:exactly|at least)|MUST call CSVQA|use the provided tool)", active, re.I))
    review = json.loads((ROOT / "outputs/react_revision_20261006/few_shot_v7_transfer_failure_review.json").read_text())
    for case_id in review["case_ids"]:
        result = json.loads((ROOT / "outputs/react_revision_20261006/few_shot_only_v7/automatic/attempts" / case_id / "result.json").read_text())
        env["get_csv_code"](result["generated_model"], "RA", "current query")
        check("Saved rejected model now passes unchanged: " + case_id, captured[-1][0][0] == result["generated_model"])
    try:
        env["get_csv_code"](model, "NRM", "current query", data_payload=observation)
    except ValueError:
        checks.append("Explicit Observation handoff remains blocked")
    else:
        raise AssertionError("Observation handoff guard missing")

# The Others CSV route forwards its full abstract plan and a schema without rows.
other = ast.unparse(funcs["get_Others_response"])
check("Others forwards full model without filtering", "abstract_plan=abstract_model_plan" in other)
check("Others codegen receives only source schema", "schema=source_schema_only(dataset_address)" in other)
pipeline = ast.unparse(funcs["execute_pipeline_case"])
check("Pipeline does not pass RA Observation", "legacy_observation=''" in pipeline)
report = {"checks_passed": len(checks), "checks": checks, "api_calls": 0,
          "evaluation_status": "not_run", "notebook_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
          "baseline_v7_results_preserved": True,
          "note": "The 12 models now pass the handoff unchanged. This does not establish mathematical correctness or Objective Match."}
(HERE / "offline_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ["checks_passed", "api_calls", "evaluation_status"]}))
