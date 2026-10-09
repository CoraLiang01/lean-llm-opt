"""Restore original ReAct modeling while preserving component boundaries and history."""
import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/react_revision_20261006"
SOURCE_FILES = {
    "full": "LEAN_LLM_OPT_4.1_Large-scale_1006.ipynb",
    "rag_only": "Ablation_Study_Large_Scale_Or_RAG_Only.ipynb",
    "few_shot_only": "Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb",
    "examples_only": "LOTO_Examples_Only_GPT4.1_Large-scale.ipynb",
    "examples_and_route": "LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb",
}
FILES = {method: name.removesuffix(".ipynb") + "_ReAct.ipynb"
         for method, name in SOURCE_FILES.items()}

CSV_REACT = '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Original ReAct agent chooses CSVQA; validate one call without repair or rerun."""
    route = normalize_route(route)
    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    csvqa_calls = 0
    original_tool_func = qa_tool.func
    def invoke_csvqa_once(tool_query):
        nonlocal csvqa_calls
        if csvqa_calls:
            raise RuntimeError("Repeated CSVQA request blocked; no extraction rerun is allowed")
        csvqa_calls += 1
        return original_tool_func(tool_query)
    qa_tool.func = invoke_csvqa_once
    prefix += ("\\nFor the CURRENT problem, retrieve data through CSVQA exactly once before the Final Answer. "
               "Demonstration Observations are historical and cannot supply current data. "
               "Use a concise symbolic model and exact Data Mapping with table_id and column names. "
               "Bind every parameter to source data or a query-defined expression; do not invent a missing limit. "
               "If total capacity denotes listed per-entity capacities, retain their bounds and their sum. "
               "Preserve all variable domains, constraints, objective sense and additive constants. "
               "Define each index set from ALL current returned entities unless the original query restricts it. "
               "Example counts and profile samples must not determine the current entity set. "
               "Use optional matrix alias mappings only when supplied; otherwise preserve exact raw IDs.")
    suffix = """Begin!

CURRENT USER DESCRIPTION START
{input}
CURRENT USER DESCRIPTION END

CURRENT CSVQA DATA STATE: {current_data_state}

If the state is NOT_LOADED, a Final Answer is forbidden. Respond with:
Thought: I need the current source data.
Action: CSVQA
Action Input: <copy only the user description between the START and END markers>
Do not copy markers, protocol instructions, examples or status into Action Input.
When the state is READY, your response MUST have exactly this envelope:
Thought: I have the current data and can formulate the model.
Final Answer:
<the complete symbolic mathematical model and Data Mapping>
The literal label "Final Answer:" is mandatory before every mathematical-model heading.
Never output a bare model, a markdown heading or explanatory prose before these labels.
Keep numerical data in the Observation; do not repeat its tables or enumerate parameters.
Do not request CSVQA again.
Do not treat any demonstration's Observation as current problem data.
{agent_scratchpad}"""
    agent = initialize_agent(
        tools=[qa_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": prefix, "suffix": suffix}, verbose=True,
        handle_parsing_errors=False, return_intermediate_steps=True,
        early_stopping_method="force",
    )
    # Prompt metadata reports actual tool state; the ReAct agent still chooses its Action.
    agent.agent.llm_chain.prompt = agent.agent.llm_chain.prompt.partial(
        current_data_state=lambda: "READY" if csvqa_result.get("observation") else "NOT_LOADED",
    )
    try:
        result = invoke_react_with_required_csvqa(agent, query, route)
    except Exception as exc:
        evidence = dict(csvqa_result)
        evidence["trace"] = {"status": "NOT_CALLED" if not csvqa_calls else "DATA_TOOL_FAILED",
                             **evidence.get("trace", {}), "formulation_protocol": "ReAct",
                             "csvqa_call_count": csvqa_calls}
        exc.csvqa_result = evidence
        if not evidence.get("observation"):
            exc.pipeline_stage = "data_extraction"
        raise
    csvqa_result.setdefault("trace", {}).update(formulation_protocol="ReAct", csvqa_call_count=1)
    return {"formulation": result["output"], **csvqa_result}
'''

FEW_REACT = '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """ReAct with a complete Python Observation; no CSVQA tool or extraction planner."""
    observation = direct_source_observation(dataset_address)
    prefix = prefix.replace("{{", "{").replace("}}", "}")
    prefix = escape_braces(prefix)
    prefix += ("\\nThe CSVQA actions in demonstrations describe historical examples. "
               "No tools are available in this experiment. Complete current source data is already "
               "provided by Python in the Observation. Use it directly without screening, rewriting "
               "or inventing values. Return a concise symbolic model and source-column Data Mapping; "
               "do not copy numerical tables. Preserve the original objective and all constraints.")
    prefix += "\\nCurrent Python Observation:\\n" + escape_braces(observation)
    suffix = """Begin!

Original user description: {input}
The complete CURRENT Observation has already been provided. No tools are available.
Your response MUST have exactly this envelope:
Thought: I can formulate from the supplied current data.
Final Answer:
<the complete symbolic model and Data Mapping>
The literal label "Final Answer:" is mandatory before any model heading.
Never output a bare model or markdown heading before these labels.
Do not emit Action or Action Input for a historical demonstration.
{agent_scratchpad}"""
    # initialize_agent rejects an empty tool list; construct the same ReAct agent directly.
    from langchain_classic.agents import AgentExecutor, ZeroShotAgent
    model_prompt = ZeroShotAgent.create_prompt(tools=[], prefix=prefix, suffix=suffix)
    reasoning_agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=make_llm(), prompt=model_prompt), allowed_tools=[],
    )
    agent = AgentExecutor(
        agent=reasoning_agent, tools=[], verbose=True, handle_parsing_errors=False,
        return_intermediate_steps=True, early_stopping_method="force",
    )
    result = agent.invoke(query)
    if result.get("intermediate_steps"):
        raise RuntimeError("Few-shot Only requested an unavailable tool; no repair or rerun")
    return {"formulation": result["output"], "observation": observation,
            "trace": {"status": "DIRECT_FULL_SOURCE", "formulation_protocol": "ReAct",
                      "csvqa_call_count": 0, "planner_attempt_count": 0,
                      "repair_count": 0, "retry_count": 0, "fallback_count": 0}}
'''


def replace_function(book, name, replacement):
    matches = 0
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                lines = source.splitlines(keepends=True)
                cell["source"] = ("".join(lines[:node.lineno-1]) + replacement +
                                  "".join(lines[node.end_lineno:])).splitlines(keepends=True)
                matches += 1
    assert matches == 1, (name, matches)


def save(path, book):
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            if path.parent == ROOT and path.name in FILES.values():
                source = "".join(cell["source"])
                source = re.sub(r'(?m)^NOTEBOOK_FILENAME = "[^"\n]+"$',
                                f'NOTEBOOK_FILENAME = "{path.name}"', source)
                source = re.sub(r'(?m)^LOTO_BASE_NOTEBOOK = "[^"\n]+"$',
                                f'LOTO_BASE_NOTEBOOK = "{FILES["full"]}"', source)
                cell["source"] = source.splitlines(keepends=True)
            ast.parse("".join(cell["source"]))
            cell["outputs"] = []
            cell["execution_count"] = None
    assert not re.search(r"[\u4e00-\u9fff]", "\n".join("".join(c["source"]) for c in book["cells"]))
    path.write_text(json.dumps(book, indent=1, ensure_ascii=False) + "\n")


def main(version="v4"):
    OUT.mkdir(exist_ok=True)
    backups = OUT / "backups_direct_call_version"
    backups.mkdir(exist_ok=True)
    initial = {}
    for method, name in SOURCE_FILES.items():
        path = ROOT / name
        if method == "full" and not path.exists():
            path = ROOT / "LEAN_LLM_OPT_4.1_Large-scale.ipynb"
        destination = backups / name
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
        initial[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = OUT / "pre_change_backups.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                                        "sha256": initial}, indent=2))
    books = {}
    for method, name in FILES.items():
        book = json.loads((backups / SOURCE_FILES[method]).read_text())
        replace_function(book, "formulate_with_csvqa", FEW_REACT if method == "few_shot_only" else CSV_REACT)
        for i, cell in enumerate(book["cells"]):
            source = "".join(cell["source"])
            source = source.replace("outputs/minimal_revision_20261005", "outputs/react_revision_20261006")
            source = re.sub(rf'{method}_v[45]', method + '_' + version, source)
            source = source.replace("full_v4", "full_" + version)
            source = source.replace('VERSION = "single-pass-source-data-v4"', f'VERSION = "react-source-data-{version}"')
            if method == "full":
                source = source.replace('NOTEBOOK_FILENAME = "LEAN_LLM_OPT_4.1_Large-scale.ipynb"',
                                        'NOTEBOOK_FILENAME = "LEAN_LLM_OPT_4.1_Large-scale_1006.ipynb"')
                if i == 38:
                    source = ("## 19. Test all routes (606 cases)\n\n"
                              "The ReAct revision requires a new complete full-model evaluation and nine-sheet review. "
                              "RUN_FORCED_ROUTES remains disabled until the new gate passes. "
                              "The earlier direct-call gate and results do not validate this architecture.\n")
                if i == 39:
                    source = source.replace("RUN_FORCED_ROUTES = True", "RUN_FORCED_ROUTES = False")
            if i == 14:
                source = ("## 7. ReAct formulation\n\n" +
                          ("The original ReAct agent receives a complete Python Observation and has no data tools. "
                           "CSVQA/planning remain removed; full source data is never sent to code generation.\n"
                           if method == "few_shot_only" else
                           "The original ReAct agent chooses CSVQA before its Final Answer. Python validates "
                           "one tool call after the single agent invocation; no protocol correction or rerun exists.\n"))
            if i == 27:
                source = source.replace('Set MIPGap=1e-4 before optimize().',
                    'Use numeric variable bounds or omit them; never pass None as lb or ub.\n'
                    'Keep decision-variable containers distinct from loop, record and parameter names; never overwrite them.\n'
                    'Add comparisons as constraints; never sum TempConstr objects or use them as linear-expression terms.\n\n'
                    'Set MIPGap=1e-4 before optimize().')
            if 'fallback_count=evidence.get("trace", {}).get("fallback_count", 0)' not in source:
                source = source.replace('csvqa_status=evidence.get("trace", {}).get("status"))',
                    'csvqa_status=evidence.get("trace", {}).get("status"),\n'
                    '                          fallback_count=evidence.get("trace", {}).get("fallback_count", 0))')
            cell["source"] = source.splitlines(keepends=True)
        books[method] = book
    save(ROOT / FILES["full"], books["full"])
    full_sha = hashlib.sha256((ROOT / FILES["full"]).read_bytes()).hexdigest()
    for method in FILES:
        if method == "full":
            continue
        if method in {"rag_only", "few_shot_only"}:
            source = "".join(books[method]["cells"][3]["source"])
            source = re.sub(r'BASE_NOTEBOOK_SHA256 = "[a-f0-9]+"',
                            f'BASE_NOTEBOOK_SHA256 = "{full_sha}"', source)
            books[method]["cells"][3]["source"] = source.splitlines(keepends=True)
        save(ROOT / FILES[method], books[method])
    (OUT / f"revision_scope_{version}.json").write_text(json.dumps({
        "version": version,
        "architecture": "Original ZERO_SHOT_REACT_DESCRIPTION modeling agent restored",
        "full_and_rag_and_loto": "Agent chooses CSVQA, then uses its Observation; one agent invocation",
        "few_shot_only": "ReAct agent with complete Python Observation and no tools",
        "Others_CSV_and_query_only": "Existing original workflows retained",
        "parser_correction": False, "protocol_repair": False, "model_or_code_rerun": False,
        "606_enabled": False, "old_results": "outputs/minimal_revision_20261005 retained unchanged",
        "files": FILES,
    }, indent=2))
    print("Restored ReAct in five notebooks; all sources are English and parse successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v4")
    main(parser.parse_args().version)
