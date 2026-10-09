"""Apply the authorized data-source-only Few-shot Only revision to v7."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TARGET = ROOT / "Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb"
FULL = ROOT / "LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb"
initial_manifest = json.loads((HERE / "manifest.json").read_text())
assert hashlib.sha256(FULL.read_bytes()).hexdigest() == initial_manifest["before_sha256"][str(FULL)]
nb = json.loads((HERE / TARGET.name).read_text())
full = json.loads(FULL.read_text())
changed = []


def source(index):
    return "".join(nb["cells"][index]["source"])


def put(index, text):
    if text == source(index):
        return
    nb["cells"][index]["source"] = text.splitlines(keepends=True)
    if nb["cells"][index]["cell_type"] == "code":
        nb["cells"][index]["outputs"] = []
        nb["cells"][index]["execution_count"] = None
    changed.append(index)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


put(0, """# Large-scale OR: Few-shot Only

Use the full ReAct v7 route examples, modeling guidance, model interface, solver settings and matching tolerances. Replace current-case CSVQA retrieval and extraction with complete CSV data read directly by Python and supplied as Observation to modeling. Reuse the full-model classifications.

Forward the returned formulation unchanged to code generation. Do not separately forward the current Observation to the code-generation LLM. Supply source paths and column names so generated programs can read the original files at runtime. No symbolic-section filter, extra formulation-content restriction, data selection, model repair or numeric rewrite is introduced. The no-CSV route is unchanged.
""")
text = source(3)
text = replace_once(text, '"outputs/react_revision_20261006/few_shot_only_v7"',
                    '"outputs/react_revision_20261006/few_shot_only_v7_direct_model_20261006T071958Z"')
put(3, text)

text = source(11)
node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
            and n.name == "symbolic_model_for_codegen")
lines = text.splitlines(keepends=True)
del lines[node.lineno - 1:node.end_lineno]
text = "".join(lines).rstrip() + "\n"
# Preserve all source values; add only stable source identifiers used by v7 prompts.
text = replace_once(text, 'for raw in normalize_data_address(dataset_address).splitlines():',
                    'for index, raw in enumerate(normalize_data_address(dataset_address).splitlines()):')
text = replace_once(text, 'blocks.append({"source": raw, "columns": list(frame.columns),',
                    'blocks.append({"table_id": f"file_{index}_view_0", "source": raw, "columns": list(frame.columns),')
text = replace_once(text, '[{"source": raw, "columns": list(read_csv_compat(raw, nrows=0).columns)}\n'
                    '                       for raw in normalize_data_address(dataset_address).splitlines()]',
                    '[{"table_id": f"file_{index}_view_0", "source": raw,\n'
                    '                        "columns": list(read_csv_compat(raw, nrows=0).columns)}\n'
                    '                       for index, raw in enumerate(normalize_data_address(dataset_address).splitlines())]')
put(11, text)

# Inherit the shared v7 modeling guidance, changing only its data-source instructions.
full_formulation = "".join(full["cells"][15]["source"])
guidance_start = full_formulation.index('               "Bind every parameter')
guidance_end = full_formulation.index('    suffix =', guidance_start)
guidance = full_formulation[guidance_start:guidance_end]
formulation = '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """The v7 ReAct modeler with a complete Python Observation and no CSVQA tool."""
    route = normalize_route(route)
    if route in {"NRM", "RA", "TP", "AP", "FLP"}:
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    observation = direct_source_observation(dataset_address)
    prefix = escape_braces(prefix.replace("{{", "{").replace("}}", "}"))
    sources = dict(enumerate(raw.strip() for raw in str(dataset_address).splitlines() if raw.strip()))
    prefix += ("\\nFor the CURRENT problem, use the complete Python Observation supplied below. "
               "No CSVQA tool or extraction planner is available in this experiment. "
               "CSVQA actions in demonstrations describe historical examples only. "
               "Demonstration Observations cannot supply current data. "
               "Use a concise symbolic model and exact Data Mapping with table_id and column names. "
''' + guidance + '''    prefix += "\\nCurrent Python Observation:\\n" + escape_braces(observation)
    suffix = """Begin!

CURRENT USER DESCRIPTION START
{input}
CURRENT USER DESCRIPTION END

CURRENT PYTHON DATA STATE: READY

The complete current Observation is supplied above. No tools are available.
Respond with this envelope:
Thought: I have the required current data and can formulate the model.
Final Answer:
<the complete symbolic mathematical model and Data Mapping>
The literal label "Final Answer:" is mandatory before every mathematical-model heading.
Never output a bare model or markdown heading before these labels.
Keep numerical data in the Observation; do not repeat its tables or enumerate parameters.
Do not treat any demonstration's Observation as current problem data.
{agent_scratchpad}"""
    # initialize_agent rejects an empty tool list; use the same ReAct agent directly.
    from langchain_classic.agents import AgentExecutor, ZeroShotAgent
    def create_agent():
        retry_note = ("\\nA previous protocol attempt was discarded. Follow the exact "
                      "Thought/Final Answer envelope using the supplied current Python Observation."
                      if REACT_PROTOCOL_EVENTS else "")
        model_prompt = ZeroShotAgent.create_prompt(tools=[], prefix=prefix + retry_note, suffix=suffix)
        reasoning_agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=make_llm(), prompt=model_prompt), allowed_tools=[],
        )
        return AgentExecutor(
            agent=reasoning_agent, tools=[], verbose=True, handle_parsing_errors=False,
            return_intermediate_steps=True, early_stopping_method="force",
        )
    result = invoke_react_protocol(create_agent, query, route)
    if result.get("intermediate_steps"):
        raise RuntimeError("Few-shot Only requested an unavailable tool; no model repair")
    return {"formulation": result["output"], "observation": observation,
            "trace": {"status": "DIRECT_FULL_SOURCE", "formulation_protocol": "ReAct",
                      "csvqa_call_count": 0, "planner_attempt_count": 0,
                      "repair_count": 0, "fallback_count": 0, **react_protocol_record_fields()}}
'''
text = source(15)
old = ast.get_source_segment(text, next(n for n in ast.parse(text).body
                            if isinstance(n, ast.FunctionDef) and n.name == "formulate_with_csvqa"))
text = replace_once(text, old, formulation.rstrip())
text = replace_once(text, 'Call CSVQA exactly once and return an ABSTRACT model.',
                    'Use the supplied Python Observation and return an ABSTRACT model.')
put(15, text)

text = source(17)
text = replace_once(text, 'You MUST call CSVQA exactly once before the Final Answer and use all returned rows.',
                    'Use the supplied Python Observation before the Final Answer and use all supplied rows.')
text = replace_once(text, 'column names from CSVQA_DATA.', 'column names from the current Python Observation.')
put(17, text)
text = source(19)
text = replace_once(text, 'Call CSVQA at least once and return a complete numerical formulation.',
                    'Use the supplied Python Observation and return a complete numerical formulation.')
put(19, text)
for index in (21, 23):
    text = source(index)
    text = replace_once(text, 'Please retrieve all neccessary information from the CSV file to generate the answer.',
                        'Please use the supplied complete Python CSV Observation to generate the answer.')
    text = replace_once(text, 'When you need to retrieve information from the CSV file, use the provided tool.',
                        'The complete current CSV data is already supplied as Python Observation; no tool call is needed.')
    put(index, text)

put(25, replace_once(source(25), 'abstract_plan=symbolic_model_for_codegen(abstract_model_plan),',
                     'abstract_plan=abstract_model_plan,'))
put(27, replace_once(source(27), '_generate_code(symbolic_model_for_codegen(output), route, original_query',
                     '_generate_code(output, route, original_query'))
text = source(29)
text = replace_once(text, 'legacy_observation=formulation.get("observation", "")\n'
                    '                    if route == "RA" and mode == "legacy" else "",',
                    'legacy_observation="",')
put(29, text)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
manifest = json.loads((HERE / "manifest.json").read_text())
manifest.update(changed_cell_indices=changed, after_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                results_directory="outputs/react_revision_20261006/few_shot_only_v7_direct_model_20261006T071958Z")
(HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"changed_cell_indices": changed, "evaluation_status": "not_run"}))
