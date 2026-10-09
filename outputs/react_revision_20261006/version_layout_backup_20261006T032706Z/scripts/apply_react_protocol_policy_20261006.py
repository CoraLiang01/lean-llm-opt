"""Apply the user's revised ReAct retry policy to the preserved v4 notebooks."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from restore_react_revision_20261006 import FILES, ROOT, OUT, replace_function, save

PROTOCOL = '''REACT_PROTOCOL_EVENTS = []
REACT_PROTOCOL_DEADLINE = None


class ReActProtocolError(RuntimeError):
    """A discarded protocol attempt, not a completed benchmark outcome."""


def react_protocol_record_fields():
    return {"retry_count": len(REACT_PROTOCOL_EVENTS),
            "protocol_retry_count": len(REACT_PROTOCOL_EVENTS),
            "protocol_retry_events": json.dumps(REACT_PROTOCOL_EVENTS, ensure_ascii=False)}


def invoke_react_protocol(agent_factory, query, route, *, required_tool=None):
    """Restart only malformed ReAct output or a missing required CSVQA call."""
    import time
    from langchain_core.exceptions import OutputParserException
    deadline = REACT_PROTOCOL_DEADLINE or (time.monotonic() + REACT_PROTOCOL_TIMEOUT_SECONDS)
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("ReAct protocol remained incomplete at the common case deadline")
        try:
            agent = agent_factory()
            if REACT_PROTOCOL_EVENTS:
                prompt = agent.agent.llm_chain.prompt
                marker = "PROTOCOL RESTART REMINDER:"
                if marker not in prompt.template:
                    prompt.template += ("\\n" + marker + " Use exact Thought/Action/Action Input or "
                                        "Thought/Final Answer labels. Do not output a bare model." +
                                        (" Call CSVQA before the Final Answer." if required_tool else ""))
            result = agent.invoke(query)
            output = str(result.get("output", "") or "").strip()
            calls = sum(getattr(step[0], "tool", None) == required_tool
                        for step in result.get("intermediate_steps", [])
                        if isinstance(step, tuple) and step) if required_tool else 0
            if required_tool and calls < 1:
                raise ReActProtocolError("missing_required_csvqa")
            if not output or "agent stopped due to" in output.lower():
                raise ReActProtocolError("incomplete_final_answer")
            return result
        except (OutputParserException, ValueError, ReActProtocolError) as exc:
            cause, parser_error = exc, False
            while cause is not None:
                parser_error = parser_error or isinstance(cause, OutputParserException)
                cause = cause.__cause__
            if not parser_error and not isinstance(exc, ReActProtocolError):
                raise
            reason = "react_output_format" if parser_error else str(exc)
            REACT_PROTOCOL_EVENTS.append({"route": str(route), "reason": reason,
                                          "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            print(f"[ReAct protocol restart] {route}: {reason}; discard this attempt and restart the agent.")


def invoke_react_with_required_csvqa(agent_factory, query, route):
    """Return the first valid ReAct result with at least one actual CSVQA call."""
    return invoke_react_protocol(agent_factory, query, route, required_tool="CSVQA")
'''

LOAD_TABLES = '''def _load_tables(dataset_address, file_indices=None):
    """Read requested CSV files, retaining their original zero-based source indices."""
    paths = [raw.strip() for raw in str(dataset_address or "").splitlines() if raw.strip()]
    if file_indices is not None:
        if (not isinstance(file_indices, list) or not file_indices
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(paths)
                       for i in file_indices)):
            raise ValueError("CSVQA file_indices must be nonempty valid zero-based source indices")
    tables = []
    for index, raw in enumerate(paths):
        if file_indices is not None and index not in file_indices:
            continue
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSVQA cannot find dataset file: {path}")
        tables.append({"file_index": index, "path": path, "frame": _read_csv(path)})
    if not tables:
        raise ValueError("CSVQA requires at least one CSV path")
    return tables
'''

CSV_REACT = '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Original ReAct with mandatory current CSVQA and user-authorized protocol restarts."""
    route = normalize_route(route)
    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    tool_history, current_observations = [], []
    original_tool_func = qa_tool.func
    def invoke_csvqa(tool_query):
        entry = {"request": str(tool_query)}
        tool_history.append(entry)
        observation = original_tool_func(tool_query)
        entry.update(trace=dict(csvqa_result.get("trace", {})), observation=observation)
        if CSVQA_MODE_BY_ROUTE[route] == "planned":
            current_observations.append(json.loads(observation))
        return observation
    qa_tool.func = invoke_csvqa
    sources = dict(enumerate(raw.strip() for raw in str(dataset_address).splitlines() if raw.strip()))
    prefix += ("\\nFor the CURRENT problem, call CSVQA at least once before the Final Answer. "
               "Demonstration Observations are historical and cannot supply current data. "
               "You may call CSVQA multiple times for different current CSV files. "
               "Use the original user description as Action Input to read all CSV files, or use "
               'JSON Action Input with keys "query" and "file_indices" to read selected zero-based indices. '
               "Use a concise symbolic model and exact Data Mapping with table_id and column names. "
               "Bind every parameter to source data or a query-defined expression; do not invent a missing limit. "
               "If total capacity denotes listed per-entity capacities, retain their bounds and their sum. "
               "Preserve all variable domains, constraints, objective sense and additive constants. "
               "Define each index set from ALL current returned entities unless the original query restricts it. "
               "Example counts and profile samples must not determine the current entity set. "
               "Use optional matrix alias mappings only when supplied; otherwise preserve exact raw IDs.\\n"
               "Current source index-to-path mapping: " + escape_braces(json.dumps(sources)))
    suffix = """Begin!

CURRENT USER DESCRIPTION START
{input}
CURRENT USER DESCRIPTION END

CURRENT CSVQA DATA STATE: {current_data_state}

If the state is NOT_LOADED, a Final Answer is forbidden. Respond with:
Thought: I need the current source data.
Action: CSVQA
Action Input: <copy only the user description between START and END, or provide the specified JSON>
Do not copy markers, protocol instructions, examples or status into Action Input.
When the state is READY, request another CSV if needed, or respond with this envelope:
Thought: I have the required current data and can formulate the model.
Final Answer:
<the complete symbolic mathematical model and Data Mapping>
The literal label "Final Answer:" is mandatory before every mathematical-model heading.
Never output a bare model or markdown heading before these labels.
Keep numerical data in the Observation; do not repeat its tables or enumerate parameters.
Do not treat any demonstration's Observation as current problem data.
{agent_scratchpad}"""
    def create_agent():
        csvqa_result.clear()
        current_observations.clear()
        retry_note = ("\\nA previous protocol attempt was discarded. Follow the exact Thought/Action/Action Input "
                      "or Thought/Final Answer envelope and call CSVQA before your final model."
                      if REACT_PROTOCOL_EVENTS else "")
        agent = initialize_agent(
            tools=[qa_tool], llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            agent_kwargs={"prefix": prefix + retry_note, "suffix": suffix}, verbose=True,
            handle_parsing_errors=False, return_intermediate_steps=True, early_stopping_method="force",
        )
        agent.agent.llm_chain.prompt = agent.agent.llm_chain.prompt.partial(
            current_data_state=lambda: "READY" if csvqa_result.get("observation") else "NOT_LOADED",
        )
        return agent
    try:
        result = invoke_react_with_required_csvqa(create_agent, query, route)
        if len(current_observations) > 1:
            tables = {t["table_id"]: t for payload in current_observations for t in payload["tables"]}
            relationships = {r["matrix_table_id"]: r for payload in current_observations
                             for r in payload.get("relationships", [])}
            payload = {**current_observations[-1], "tables": list(tables.values()),
                       "relationships": list(relationships.values()), "ignored_file_indices": []}
            csvqa_result["observation"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        evidence = dict(csvqa_result)
        evidence["trace"] = {**evidence.get("trace", {}), "formulation_protocol": "ReAct",
                             "csvqa_call_count": len(tool_history), "tool_calls": tool_history,
                             **react_protocol_record_fields()}
        exc.csvqa_result = evidence
        if not evidence.get("observation"):
            exc.pipeline_stage = "data_extraction"
        raise
    trace = csvqa_result.setdefault("trace", {})
    trace.update(formulation_protocol="ReAct", csvqa_call_count=len(tool_history), tool_calls=tool_history,
                 fallback_count=sum(c.get("trace", {}).get("fallback_count", 0) for c in tool_history),
                 **react_protocol_record_fields())
    trace["payload_hash"] = hashlib.sha256(csvqa_result["observation"].encode("utf-8")).hexdigest()
    return {"formulation": result["output"], **csvqa_result}
'''


def function_source(book, name):
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return ast.get_source_segment(source, node)
    raise ValueError(name)


def main():
    books = {}
    for method, name in FILES.items():
        book = json.loads((OUT / "notebook_snapshots/v4" / name).read_text())
        replace_function(book, "invoke_react_with_required_csvqa", PROTOCOL)
        replace_function(book, "_load_tables", LOAD_TABLES)
        if method != "few_shot_only":
            replace_function(book, "formulate_with_csvqa", CSV_REACT)
        for cell in book["cells"]:
            source = "".join(cell["source"])
            source = source.replace("_v4", "_v5").replace("react-source-data-v4", "react-source-data-v5")
            if 'API_MAX_RETRIES = 2' in source:
                source = source.replace('API_MAX_RETRIES = 2  # Original SDK transport setting; no pipeline/model retry.',
                    'API_MAX_RETRIES = 2  # Unchanged SDK transport retry setting.\n'
                    'REACT_PROTOCOL_TIMEOUT_SECONDS = 1800  # User-authorized protocol restarts share the case deadline.')
            if 'def build_csvqa_components(' in source:
                source = source.replace('    def qa_wrapper(tool_query):\n',
                    '    def qa_wrapper(tool_query):\n'
                    '        file_indices = None\n'
                    '        if str(tool_query).lstrip().startswith("{"):\n'
                    '            request = json.loads(tool_query)\n'
                    '            file_indices = request.get("file_indices")\n')
                source = source.replace('tables = _load_tables(dataset_address)',
                                        'tables = _load_tables(dataset_address, file_indices=file_indices)')
                source = source.replace('"status": status, "plan": plan,',
                                        '"status": status, "requested_file_indices": file_indices, "plan": plan,')
            if 'def invoke_classifier(' in source:
                source = source.replace('_get_classification_agent().invoke(str(query))',
                                        'invoke_react_protocol(_get_classification_agent, str(query), "classification")')
            if method == "few_shot_only" and 'def formulate_with_csvqa(' in source:
                source = source.replace('result = agent.invoke(query)',
                                        'result = invoke_react_protocol(lambda: agent, query, route)')
                source = source.replace('"repair_count": 0, "retry_count": 0, "fallback_count": 0}',
                                        '"repair_count": 0, "fallback_count": 0, **react_protocol_record_fields()}')
            if 'def get_others_without_CSV_response(' in source:
                source = source.replace('output = agent.run({"input": query})',
                                        'output = invoke_react_protocol(lambda: agent, {"input": query}, "Others_without_CSV")["output"]')
            if 'def execute_pipeline_case(' in source:
                source = source.replace('    HTTP_RETRY_EVENTS.clear()',
                    '    global REACT_PROTOCOL_DEADLINE\n'
                    '    import time\n'
                    '    REACT_PROTOCOL_EVENTS.clear()\n'
                    '    REACT_PROTOCOL_DEADLINE = time.monotonic() + REACT_PROTOCOL_TIMEOUT_SECONDS\n'
                    '    HTTP_RETRY_EVENTS.clear()')
                source = source.replace('record.update(api_retry_count=len(HTTP_RETRY_EVENTS)',
                                        'record.update(**react_protocol_record_fields())\n        record.update(api_retry_count=len(HTTP_RETRY_EVENTS)')
            if 'def error_record(' in source:
                source = source.replace('"repair_count": 0, "retry_count": 0,',
                                        '"repair_count": 0, **react_protocol_record_fields(),')
            if cell['cell_type'] == 'markdown' and '## 7. ReAct formulation' in source:
                source = ('## 7. ReAct formulation\n\n' +
                    ('Complete Python Observation, no CSVQA or planner. ReAct format errors restart the agent.\n'
                     if method == 'few_shot_only' else
                     'The ReAct agent must call CSVQA at least once. Different CSV files may be read in multiple '
                     'Actions. Missing CSVQA calls and malformed ReAct output restart the agent within the '
                     'common case deadline. Intermediate protocol attempts are logged, not scored as failed cases.\n'))
            if cell['cell_type'] == 'markdown':
                source = source.replace('No failed formulation, generated program, solver result or truncated model response is regenerated.',
                    'Missing CSVQA calls and malformed ReAct output restart the agent within the common case deadline. These intermediate attempts are logged and are not separate scored failures. Generated programs, solver results and truncated model responses are not regenerated.')
                source = source.replace('No repair or model rerun is available.',
                    'Only user-authorized ReAct protocol restarts are available; code and solver repair remain disabled.')
                source = source.replace('The pipeline classifies, formulates, generates code, and solves once.',
                    'The pipeline classifies, obtains a protocol-valid formulation, generates code, and solves once. ReAct protocol restarts are recorded separately.')
                source = source.replace('Run each code version and case once. Resume recorded successes and failures without regenerating them.',
                    'Use one scored pipeline attempt per code version and case. Authorized ReAct protocol restarts occur inside that attempt. Resume recorded final successes and failures without regenerating them.')
            cell['source'] = source.splitlines(keepends=True)
        books[method] = book
    save(ROOT / FILES['full'], books['full'])
    full_sha = hashlib.sha256((ROOT / FILES['full']).read_bytes()).hexdigest()
    frozen_full = OUT / 'full_v5/frozen_notebook.ipynb'
    if frozen_full.exists():
        frozen = json.loads(frozen_full.read_text())
        assert [c['source'] for c in frozen['cells'] if c['cell_type'] == 'code'] == [
            c['source'] for c in books['full']['cells'] if c['cell_type'] == 'code'], 'Do not modify an active code version'
        full_sha = hashlib.sha256(frozen_full.read_bytes()).hexdigest()
    for method, book in books.items():
        if method == 'full':
            continue
        if method in {'rag_only', 'few_shot_only'}:
            source = ''.join(book['cells'][3]['source'])
            source = re.sub(r'BASE_NOTEBOOK_SHA256 = "[a-f0-9]+"', f'BASE_NOTEBOOK_SHA256 = "{full_sha}"', source)
            book['cells'][3]['source'] = source.splitlines(keepends=True)
        elif method.startswith('examples'):
            for cell in book['cells']:
                source = ''.join(cell['source'])
                source = source.replace('LOTO_BASE_NOTEBOOK = "LEAN_LLM_OPT_4.1_Large-scale.ipynb"',
                                        f'LOTO_BASE_NOTEBOOK = "{FILES["full"]}"')
                source = re.sub(r'LOTO_BASE_SHA256 = "[a-f0-9]+"', f'LOTO_BASE_SHA256 = "{full_sha}"', source)
                cell['source'] = source.splitlines(keepends=True)
        save(ROOT / FILES[method], book)
    (OUT / 'revision_scope_v5.json').write_text(json.dumps({
        'version': 'v5', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User requires CSVQA and format restarts; different-CSV calls may repeat.',
        'architecture': 'Original ReAct', 'csvqa_minimum_calls': 1,
        'multiple_csvqa_calls_allowed': True, 'selected_csv_indices': 'Zero-based original file order',
        'protocol_retry_reasons': ['missing_required_csvqa', 'react_output_format', 'incomplete_final_answer'],
        'protocol_retry_deadline_seconds': 1800, 'successful_outcome': 'First protocol-valid completion',
        'code_repair': False, 'objective_based_retry': False,
        '606_enabled': False, 'files': FILES,
    }, indent=2))
    print('Applied user-authorized ReAct protocol policy to five English notebooks (v5).')


if __name__ == '__main__':
    main()
