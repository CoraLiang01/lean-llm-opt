"""Apply the evidence-based notebook revision; original files are already backed up."""
import ast
import copy
import difflib
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/minimal_revision_20261005"
FULL = "LEAN_LLM_OPT_4.1_Large-scale.ipynb"
REVISION = "v4"


def source(nb, index):
    return "".join(nb["cells"][index]["source"])


def put(nb, index, text):
    nb["cells"][index]["source"] = text.splitlines(keepends=True)


def replace(nb, index, old, new):
    text = source(nb, index)
    assert old in text, (index, old)
    put(nb, index, text.replace(old, new))


def replace_function(nb, index, name, text):
    src = source(nb, index)
    node = next(n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    lines = src.splitlines(keepends=True)
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    put(nb, index, "".join(lines[:first]) + text.rstrip() + "\n" + "".join(lines[node.end_lineno:]))


def append_cell(nb, text, kind="code"):
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    nb["cells"].append(cell)


def write(name, nb):
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{name}:cell{i}")
            cell.update(outputs=[], execution_count=None)
    target = ROOT / name
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    before = json.loads((OUT / "backups" / name).read_text())
    old = "\n\n".join("".join(c["source"]) for c in before["cells"])
    new = "\n\n".join("".join(c["source"]) for c in nb["cells"])
    (OUT / (name + ".diff")).write_text("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=name+":before", tofile=name+":after")))


nb = json.loads((OUT / "backups" / FULL).read_text())
replace(nb, 3, 'RESULTS_DIR = PROJECT_ROOT / "outputs/interface_v4_full"', 'RESULTS_DIR = PROJECT_ROOT / "outputs/minimal_revision_20261005/full_v1"')
replace(nb, 3, '\nVERSION = "rag-101-ra-planned-interface-v4"', '\nVERSION = "single-pass-source-data-v1"')
replace(nb, 3, 'CACHE_SCHEMA_VERSION = "rag-101-ra-planned-interface-v4"', 'CACHE_SCHEMA_VERSION = VERSION')
replace(nb, 3, 'NRM_RETRY_ON_TRUNCATION = True  # False: propagate truncation immediately; True: retry once.', 'NRM_RETRY_ON_TRUNCATION = False  # Single attempt; truncation remains a failed case.')
replace(nb, 3, '"TP": "legacy",\n    "AP": "legacy", "FLP": "legacy"', '"TP": "planned",\n    "AP": "planned", "FLP": "planned"')
replace(nb, 3, '# NRM and RA use source-derived records; other routes retain their existing flow.', '# Canonical CSV routes use source-derived records; Others retains its original flow.')
replace(nb, 5, 'max_retries=2', 'max_retries=0')
replace_function(nb, 7, 'invoke_react_with_required_csvqa', '''def invoke_react_with_required_csvqa(agent, query, route):
    """Validate tool use after one agent invocation; never correct or rerun it."""
    result = agent.invoke(query)
    calls = sum(isinstance(step, tuple) and len(step) >= 2
                and getattr(step[0], "tool", None) == "CSVQA"
                for step in result.get("intermediate_steps", []))
    if calls != 1:
        raise RuntimeError(f"{route} ReAct must call CSVQA exactly once; observed {calls}")
    return result''')
put(nb, 7, source(nb, 7) + '''

for _route in ("TP", "AP", "FLP"):
    CSVQA_PLANNED_PROMPTS[_route] = (
        "Select every source row and column required by the original query, including entity IDs, "
        "matrix coefficients and axes, costs, capacities, demands and additional restrictions. "
        "Use exact source column names and retain business keys. Exclude a column only when it "
        "has no role in the query. Illustrative examples do not restrict the entity set."
    )
    CSVQA_TOOL_DESCRIPTIONS[_route] = "Return source-derived records, exact columns, filters and matrix axes."
''')
replace(nb, 9, 'Every explicit subset must have a filter and every filter must quote supporting query evidence.', 'Filter only an explicit restriction; names introduced as examples (e.g., such as, for example)\ndo not restrict the entity set. Every filter must quote the query restriction that requires it.')
replace(nb, 9, 'normalized = [key(value) for value in values]', 'normalized = [str(value) for value in values]')
# A quoted illustrative name is evidence for the planner, not a mandatory Python filter.
start = source(nb, 9).index('    required_values = {')
end = source(nb, 9).index('    required_columns = {', start)
put(nb, 9, source(nb, 9)[:start] + source(nb, 9)[end:])
replace(nb, 11, 'if route not in {"NRM", "RA"}:', 'if CSVQA_MODE_BY_ROUTE[route] != "planned":')
replace(nb, 11, '"payload_hash": hashlib.sha256(observation.encode("utf-8")).hexdigest(),', '"repair_count": 0, "retry_count": 0,\n            "fallback_count": int(status == "FALLBACK_FULL_DATA"),\n            "payload_hash": hashlib.sha256(observation.encode("utf-8")).hexdigest(),')
replace(nb, 15, '    prefix += ("\\nPreserve', '''    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
        prefix += ("\\nReturn a concise symbolic model and Data Mapping with exact table_id and column names. "
                   "Use every returned row needed by the original query; do not transcribe numerical tables. "
                   "Bind every parameter to a source or a query-defined expression. Never invent a missing limit. "
                   "If total capacity refers to all listed per-entity capacities, retain each bound and their sum.")
    prefix += ("\\nPreserve''')
replace(nb, 15, '    result = invoke_react_with_required_csvqa(agent, query, route)\n    return', '''    try:
        result = invoke_react_with_required_csvqa(agent, query, route)
    except Exception as exc:
        exc.csvqa_result = dict(csvqa_result)
        raise
    return''')
src = source(nb, 15)
start = src.index('    for attempt in range(2 if NRM_RETRY_ON_TRUNCATION else 1):')
put(nb, 15, src[:start] + '''    return formulate_with_csvqa(
        query, dataset_address, "NRM", csvqa_system_prompt, csvqa_tool_description, prefix, suffix,
    )
''')
# General API/indexing errors recorded in historical Variant runs.
api_hints = '''
Use a list of unique flat tuple keys with addVars(keys); do not pass a tuplelist and
another Cartesian index unless that extra dimension is intended. Preserve every component
of composite business keys in dictionaries and joins. Sum repeated component rows by the
complete key instead of overwriting them. Ineligible decision pairs must be constrained to
zero or excluded; their presence in source data is not a reason to reject the entire input.
Never use chained comparisons, Python and/or, or bitwise |/& on Gurobi constraints.
Represent logical conditions with explicit binary variables and indicator/linear constraints.
Use pandas Series.str.casefold() for vectorized text and string.casefold() for one string.
'''
replace(nb, 27, 'Set MIPGap=1e-4 before optimize().', api_hints + '\nSet MIPGap=1e-4 before optimize().')
replace(nb, 25, '        code_gen_template += "\\n" + MODEL_INTERFACE_INSTRUCTIONS', '        code_gen_template += "\\n" + CSV_SOLVER_INSTRUCTIONS + MODEL_INTERFACE_INSTRUCTIONS')
# The query-only formulation function and its retrieved examples remain intact.
for index in (13, 15, 25):
    replace(nb, index, 'handle_parsing_errors=True', 'handle_parsing_errors=False')
replace(nb, 29, 'def execute_pipeline_case(case, *, forced_route=None):', '''def classification_for_case(case):
    return invoke_classifier(case["Query"])


def execute_pipeline_case(case, *, forced_route=None):''')
replace(nb, 29, 'classification = invoke_classifier(query) if forced_route is None else {}', 'classification = classification_for_case(case) if forced_route is None else {}')
replace(nb, 29, '"experiment_mode": "forced" if forced_route else "automatic"}', '"experiment_mode": "forced" if forced_route else "automatic",\n              "repair_count": 0, "retry_count": 0, "fallback_count": 0}')
replace(nb, 29, 'csvqa_status=formulation.get("trace", {}).get("status"))', 'csvqa_status=formulation.get("trace", {}).get("status"),\n                      csvqa_trace=json.dumps(formulation.get("trace", {}), ensure_ascii=False),\n                      fallback_count=formulation.get("trace", {}).get("fallback_count", 0))')
replace(nb, 29, '    except Exception as exc:\n        exc.pipeline_context = record', '''    except Exception as exc:
        evidence = getattr(exc, "csvqa_result", {})
        if evidence:
            record.update(csvqa_observation=evidence.get("observation", ""),
                          csvqa_trace=json.dumps(evidence.get("trace", {}), ensure_ascii=False),
                          csvqa_status=evidence.get("trace", {}).get("status"))
        if isinstance(exc, ModelInterfaceError):
            record["pipeline_stage"] = "result_extraction"
        exc.pipeline_context = record''')
replace(nb, 33, '"record_status": "error", "cache_source": "computed",', '''"record_status": "error", "cache_source": "computed",
        "generated_model": getattr(exc, "pipeline_context", {}).get("generated_model", ""),
        "solve_code": getattr(exc, "pipeline_context", {}).get("solve_code", ""),
        "csvqa_observation": getattr(exc, "pipeline_context", {}).get("csvqa_observation", ""),
        "csvqa_trace": getattr(exc, "pipeline_context", {}).get("csvqa_trace", "{}"),
        "repair_count": 0, "retry_count": 0,
        "fallback_count": getattr(exc, "pipeline_context", {}).get("fallback_count", 0),''')
replace(nb, 33, '"csvqa_observation": "data_overview.md"}', '"csvqa_observation": "data_overview.md", "csvqa_trace": "csvqa_trace.json"}')
# One recorded attempt includes failures; resuming must not rerun failed cases.
replace(nb, 33, 'or not record.get("final_ok") or record.get("cache_fingerprint") != fingerprint', 'or record.get("cache_fingerprint") != fingerprint')
replace(nb, 33, 'record.get("record_status") != "completed"', 'record.get("record_status") not in {"completed", "error"}')
replace(nb, 35, '    cached = {record_key(row): row for row in load_records(output_csv)} if reuse_completed else {}', '''    existing = load_records(output_csv)
    if existing and not reuse_completed:
        raise ValueError("Use a new round directory for a fresh run; historical attempts cannot be overwritten")
    cached = {record_key(row): row for row in existing}''')
replace(nb, 35, '            fingerprint = case_fingerprint(case, route, source_fingerprint=source)', '''            fingerprint = case_fingerprint(case, route, source_fingerprint=source)
            if key in cached and cached[key].get("cache_fingerprint") != fingerprint:
                raise ValueError("Existing attempts use different code/data; choose a new version directory")''')
# materialize_record requires optional evidence on old rows; fresh failures have all artifacts.
replace(nb, 35, '            if record is not None:\n                record["cache_source"] = "csv"', '''            if key in cached and record is None:
                raise ValueError("Recorded attempt has missing/corrupt artifacts; do not rerun it")
            if record is not None:
                record["cache_source"] = "csv"''')
replace(nb, 33, '    """Reuse only successful, intact results from the same code and data."""', '    """Reuse every intact recorded attempt, including failures; never select successes."""')
replace(nb, 35, '"""Reuse matching successful results; retry failures. False disables all result reuse."""', '"""Evaluate once per case/version; resume recorded successes and failures."""')
replace(nb, 35, '            record["cache_fingerprint"] = fingerprint\n            save_record', '''            record["cache_fingerprint"] = fingerprint
            record["base_notebook_sha256"] = globals().get("BASE_NOTEBOOK_SHA256", hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest())
            save_record''')
for index in (37, 45, 47):
    put(nb, index, source(nb, index).replace('= True', '= False'))
replace(nb, 47, 'COLUMN_SHEETS = ["200pct-S1"]', 'COLUMN_SHEETS = [f"{pct}-S{seed}" for pct in ["50pct", "100pct", "200pct"] for seed in [1, 2, 3]]')
put(nb, 6, '## 3. CSV loading and route modes\n\nNRM, RA, TP, AP and FLP use the existing declarative planner and deterministic source-data extraction. Others keeps its CSV-schema workflow and its query-only formulation route. No repair or model rerun is available.\n')
put(nb, 34, '## 17. Shared test runner\n\nRun each code version and case once. Resume recorded successes and failures without regenerating them. Use a new directory for another round; historical results cannot be overwritten. Objective Match includes every requested case in the denominator. CSV-plan failures retain the existing deterministic full-source-data fallback, which is logged separately from repair and retries.\n')
put(nb, 46, '## 23. Redundant-column benchmark\n\nAll nine sheets are selected by default. Enable RUN_REDUNDANT_COLUMNS for one pass per sheet. Each sheet has its own results directory. Review every sheet before enabling the 606 experiment.\n')
write(FULL, nb)

# V1 diagnosed an agent skipping the mandatory data tool. Call the same tool once
# before the modeling request, instead of asking the model to repair its protocol.
replace_function(nb, 15, 'formulate_with_csvqa', '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Call CSVQA once, then model its Observation once; no agent protocol repair."""
    route = normalize_route(route)
    if CSVQA_MODE_BY_ROUTE[route] == "planned":
        system_prompt = CSVQA_PLANNED_PROMPTS[route]
        tool_description = CSVQA_TOOL_DESCRIPTIONS[route]
        prefix = prefix.replace("please output required parameters in a whole text, including all vectors and matrices.",
                                "return symbolic parameters and an exact source Data Mapping; do not enumerate values.")
    llm, qa_tool, csvqa_result = build_csvqa_components(
        dataset_address, system_prompt, tool_description, route=route, user_query=query,
    )
    try:
        qa_tool.invoke(query)
    except Exception as exc:
        exc.csvqa_result = dict(csvqa_result)
        exc.pipeline_stage = "data_extraction"
        raise
    prompt = (prefix.replace("{{", "{").replace("}}", "}") +
              "\\nCSVQA has already been called exactly once by the program. Its complete current "
              "Observation is supplied below. Return the final mathematical model directly; do not "
              "request tools or copy example data. Use a concise symbolic model and an exact Data "
              "Mapping with table_id and column names. Do not enumerate numerical tables. "
              "Bind every parameter to source data or a query-defined expression; never invent a "
              "missing limit. If total capacity refers to listed per-entity capacities, retain "
              "their bounds and their sum. Preserve all variable domains, constraints, objective "
              "sense and additive constants.\\n" +
              f"Original Query:\\n{query}\\nObservation:\\n{csvqa_result['observation']}")
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"formulation": result.content, **csvqa_result}
''')
replace(nb, 27, 'Use a list of unique flat tuple keys with addVars(keys);',
        'Gurobi addVars supports lb, ub, obj, vtype and name only; do not invent keyword arguments.\nUse a list of unique flat tuple keys with addVars(keys);')
replace(nb, 29, '        if isinstance(exc, ModelInterfaceError):',
        '        record["pipeline_stage"] = getattr(exc, "pipeline_stage", record["pipeline_stage"])\n        if isinstance(exc, ModelInterfaceError):')
# Retain raw labels and expose a bijection for a declared axis whose source tables
# use different prefixes. Complete suffix strings (including leading zeros) must
# be unique and identical on both sides. No row-order guesses or numeric rewriting.
replace(nb, 9, '        checks = {\n            "matrix_table_id": matrix_id,', '''        def axis_mapping(actual, expected):
            axis_keys(actual, "source")
            axis_keys(expected, "reference")
            if set(actual) == set(expected):
                return dict(zip(actual, actual)), "exact"
            def suffix_map(values):
                matches = [re.fullmatch(r"\\D*(\\d+)", value) for value in values]
                if not all(matches):
                    return None
                keys = [match.group(1) for match in matches]
                return dict(zip(keys, values)) if len(set(keys)) == len(keys) else None
            left, right = suffix_map(actual), suffix_map(expected)
            if left is not None and right is not None and set(left) == set(right):
                return {value: right[key] for key, value in left.items()}, "unique_complete_suffix"
            return None, "unresolved"
        row_map, row_basis = axis_mapping(matrix_row_ids, row_ids)
        column_map, column_basis = axis_mapping(matrix_columns, column_ids)
        relationship = {**relationship, "row_id_mapping": row_map,
                        "column_id_mapping": column_map}
        checks = {
            "matrix_table_id": matrix_id,''')
replace(nb, 9, '"row_ids_aligned": axis_keys(matrix_row_ids, "matrix row") == axis_keys(row_ids, "row axis"),\n            "column_ids_aligned": axis_keys(matrix_columns, "matrix column") == axis_keys(column_ids, "column axis"),',
        '"row_ids_aligned": row_map is not None, "column_ids_aligned": column_map is not None,\n            "row_mapping_basis": row_basis, "column_mapping_basis": column_basis,')
replace(nb, 27, 'Records are already structured dictionaries;', '''Declared matrix relationships supply row_id_mapping and column_id_mapping from raw
matrix labels to the corresponding entity IDs. Apply those bijections; raw prefixes may
differ. Preserve both originals and every coefficient. Never substitute a positional join.
Records are already structured dictionaries;''')
replace(nb, 15, '"sense and additive constants.\\n" +', '''"sense and additive constants. For declared matrices use the supplied row_id_mapping "
              "and column_id_mapping rather than assuming identical raw axis labels.\\n" +''')
replace(nb, 9, '            mask = _apply_condition(frame[column], condition)', '''            illustrative = re.search(r"\\bsuch as\\b|\\bfor example\\b|\\be\\.g\\.", evidence, re.I)
            restrictive = re.search(r"\\bonly\\b|\\bexclud\\w*\\b|\\bexcept\\b|\\brestrict\\w*\\b|\\blimited to\\b", evidence, re.I)
            if illustrative and not restrictive:
                raise ValueError(f"Illustrative query examples cannot justify a source-row filter: {evidence}")
            mask = _apply_condition(frame[column], condition)''')
# Supply alias mappings only for different raw IDs. Identity axes need no new
# dictionary interface and continue to use the existing record['values'] fields.
replace(nb, 9, '        relationship = {**relationship, "row_id_mapping": row_map,\n                        "column_id_mapping": column_map}', '''        relationship = dict(relationship)
        if row_basis == "unique_complete_suffix":
            relationship["row_id_mapping"] = row_map
        if column_basis == "unique_complete_suffix":
            relationship["column_id_mapping"] = column_map''')
replace(nb, 27, 'Declared matrix relationships supply row_id_mapping and column_id_mapping from raw\nmatrix labels to the corresponding entity IDs. Apply those bijections; raw prefixes may\ndiffer. Preserve both originals and every coefficient. Never substitute a positional join.', '''Optional row_id_mapping/column_id_mapping in a declared matrix relationship are
dictionaries from raw matrix label strings to entity ID strings. Apply them only when
present; otherwise use exact raw IDs. Obtain a row label from record['values'][row_id_column],
never from the record dictionary itself. Preserve every coefficient and both original IDs.
Derive complete index sets from current records; an example or a copied formulation's
literal entity list must not narrow them without an explicit original-query restriction.''')
replace(nb, 15, '"sense and additive constants. For declared matrices use the supplied row_id_mapping "\n              "and column_id_mapping rather than assuming identical raw axis labels.\\n" +', '''"sense and additive constants. Define each index set from ALL current returned entities "
              "unless the original query explicitly restricts it. Demonstration names, counts and "
              "profile samples must never determine the current entity set. Optional matrix alias "
              "mappings apply only when present; otherwise preserve exact raw ID matching.\\n" +''')
replace(nb, 9, 'Declare a matrix relationship only when the profile exposes its two axes unambiguously.', '''Choose the filter operator from the requested matching semantics. A categorical code
classifying identifier families refers to a prefix when no separate category column exists;
arbitrary substring matching also admits unrelated identifiers and requires explicit query support.
Declare a matrix relationship only when the profile exposes its two axes unambiguously.''')
# Preserve the baseline SDK's two transport retries. They do not regenerate a
# failed model/program, and every actual HTTP retry is recorded separately.
put(nb, 2, source(nb, 2) + '\nimport httpx\n')
put(nb, 3, source(nb, 3) + '\nAPI_MAX_RETRIES = 2  # Original SDK transport setting; no pipeline/model retry.\n')
put(nb, 5, '''HTTP_RETRY_EVENTS = []


def _record_http_request(request):
    retry_index = int(request.headers.get("x-stainless-retry-count", "0"))
    if retry_index:
        HTTP_RETRY_EVENTS.append({"endpoint": request.url.path, "retry_index": retry_index})


@lru_cache(maxsize=1)
def _get_http_client():
    return httpx.Client(timeout=180, event_hooks={"request": [_record_http_request]})


''' + source(nb, 5))
replace(nb, 5, 'max_retries=0', 'max_retries=API_MAX_RETRIES, http_client=_get_http_client()')
replace(nb, 29, '    query, address = record["query"], record["dataset_address"]',
        '    HTTP_RETRY_EVENTS.clear()\n    query, address = record["query"], record["dataset_address"]')
replace(nb, 29, '        record.update(final_objective=objective',
        '        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))\n        record.update(final_objective=objective')
replace(nb, 29, '        evidence = getattr(exc, "csvqa_result", {})',
        '        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))\n        evidence = getattr(exc, "csvqa_result", {})')
put(nb, 4, '## 2. Shared model clients\n\nUse the fixed GPT-4.1 snapshot and shared HTTP connection pool. Preserve the original SDK setting of at most two transport retries for connection/service errors, recording every actual retry. No failed formulation, generated program, solver result or truncated model response is regenerated.\n')
for index in (3,):
    put(nb, index, source(nb, index).replace('full_v1', f'full_{REVISION}').replace('single-pass-source-data-v1', f'single-pass-source-data-{REVISION}'))
put(nb, 14, '## 7. Data retrieval and formulation\n\nPython invokes the existing CSVQA Tool once before a single modeling request. Planned extraction and all route demonstrations remain. The modeling request receives the resulting Observation without an agent deciding whether to call the tool.\n')
write(FULL, nb)

# All experimental notebooks start with the same final definition cells.
base = copy.deepcopy(nb)
base["cells"] = base["cells"][:36]
base_sha = hashlib.sha256((ROOT / FULL).read_bytes()).hexdigest()

for name, method in [('Ablation_Study_Large_Scale_Or_RAG_Only.ipynb', 'rag_only'),
                     ('Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb', 'few_shot_only')]:
    exp = copy.deepcopy(base)
    old = json.loads((OUT / 'backups' / name).read_text())
    replace(exp, 3, f'NOTEBOOK_FILENAME = "{FULL}"', f'NOTEBOOK_FILENAME = "{name}"')
    replace(exp, 3, f'outputs/minimal_revision_20261005/full_{REVISION}', f'outputs/minimal_revision_20261005/{method}_{REVISION}')
    put(exp, 3, source(exp, 3) + f'\nMETHOD = "{method}"\nCLASSIFICATION_RESULTS_PATH = PROJECT_ROOT / "outputs/minimal_revision_20261005/full_{REVISION}/automatic/results.csv"\nBASE_NOTEBOOK_SHA256 = "{base_sha}"\n')
    cached_source = source(old, 13)
    start = cached_source.index('def attach_cached_classifications(')
    append_cell(exp, '## Experimental component removal\n\nReuse the final full-model classifications. No model output or solver result is reused. All common settings and execution definitions above match the full notebook.\n', 'markdown')
    cached_source = cached_source[start:]
    # Require provenance for exactly the final full code and current data.
    cached_source = cached_source.replace('labels, routes = [], []', 'labels, routes = [], []\n    provenance = load_records(path)\n    if any(row.get("base_notebook_sha256") != BASE_NOTEBOOK_SHA256 for row in provenance):\n        raise ValueError("Classification results are not from the final full notebook")')
    cached_source = cached_source.replace('        label = normalize_problem_class(row["predicted_label"])',
        '        if not row["predicted_label"] and not row["assigned_route"]:\n            labels.append(None)\n            routes.append(None)\n            continue\n        label = normalize_problem_class(row["predicted_label"])')
    cached_source = cached_source.replace('    label = normalize_problem_class(case["cached_predicted_label"])',
        '    if not case.get("cached_predicted_label"):\n        raise RuntimeError("Final full-model classification is unavailable; no classification rerun is allowed")\n    label = normalize_problem_class(case["cached_predicted_label"])')
    append_cell(exp, cached_source)
    if method == 'rag_only':
        # These fixed Question/Final Answer triggers are also few-shot insertion.
        query_only = source(exp, 25)
        start = query_only.index('    """ \n    Use the following triggers')
        end = query_only.index('    "USER QUESTION:', start)
        put(exp, 25, query_only[:start] + query_only[end:])
        append_cell(exp, '''def retrieve_rag_examples(route, query, k=1):
    """Remove only formulation/code demonstrations; FileQA and CSVQA remain."""
    return []


def retrieve_similar_texts(query, retriever):
    """Remove query-only few-shot insertion; retain its ORLM_QA retrieval tool."""
    return []
''')
        put(exp, 0, '# Large-scale OR: RAG Only\n\nRemoved: retrieved model/Observation demonstrations from NRM, RA, TP, AP and FLP; Others Abstract Model Plan and code demonstrations; all retrieved code examples; query-only inserted demonstrations and its three fixed Question/Final Answer triggers. Retained: saved full-model classification, classification FileQA retrieval, CSVQA Tool, CSV loading/profiling, declarative extraction plans, Python extraction and validation, deterministic full-data fallback, Others CSV schema/statistics with runtime reading, and query-only ORLM_QA retrieval.\n')
    else:
        # Remove planner/tool definitions; preserve CSV readers and route few-shot retrieval.
        put(exp, 9, '# Few-shot Only: no extraction planner or extraction-plan execution.\n')
        put(exp, 11, '''def direct_source_observation(dataset_address):
    """Serialize every original CSV field as text, in source order, without selection."""
    blocks = []
    for raw in normalize_data_address(dataset_address).splitlines():
        frame = read_csv_compat(raw, dtype=str, keep_default_na=False)
        blocks.append({"source": raw, "columns": list(frame.columns),
                       "records": frame.to_dict("records")})
    return json.dumps({"tables": blocks}, ensure_ascii=False)


def source_schema_only(dataset_address):
    """Paths and exact column names only: no source rows reach code generation."""
    return json.dumps([{"source": raw, "columns": list(read_csv_compat(raw, nrows=0).columns)}
                       for raw in normalize_data_address(dataset_address).splitlines()], ensure_ascii=False)
''')
        replace_function(exp, 15, 'formulate_with_csvqa', '''def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """Direct Python Observation; one modeling call without tools or an extraction plan."""
    observation = direct_source_observation(dataset_address)
    prompt = (prefix.replace("{{", "{").replace("}}", "}") +
              "\\nThe CSVQA actions in demonstrations describe historical examples. No tools are available. "
              "Use the complete current Python Observation below directly; never screen, rewrite or invent data. "
              "Return a concise symbolic model and source-column Data Mapping; do not copy data rows. "
              "Preserve the original objective sense, all constants, and all constraints.\\n" +
              f"Original Query:\\n{query}\\nObservation:\\n{observation}")
    result = make_llm().invoke([HumanMessage(content=prompt)])
    return {"formulation": result.content, "observation": observation,
            "trace": {"status": "DIRECT_FULL_SOURCE", "planner_attempt_count": 0,
                      "repair_count": 0, "retry_count": 0, "fallback_count": 0}}
''')
        replace_function(exp, 25, 'csv_schema_preview', '''def csv_schema_preview(dataset_address, *, query=""):
    return direct_source_observation(dataset_address)''')
        replace(exp, 25, '            schema=schema,\n            abstract_plan=', '            schema=source_schema_only(dataset_address),\n            abstract_plan=')
        replace(exp, 27, 'LEGACY_CODE_INSTRUCTIONS = """', 'LEGACY_CODE_INSTRUCTIONS = """')
        src = source(exp, 27)
        a = src.index('LEGACY_CODE_INSTRUCTIONS = """')
        b = src.index('"""', a + len('LEGACY_CODE_INSTRUCTIONS = """')) + 3
        put(exp, 27, src[:a] + '''LEGACY_CODE_INSTRUCTIONS = """
Read the exact source CSV paths in Source Schema at runtime with pandas. Preserve text
identifiers and complete source rows, then implement query-required joins and calculations.
No complete current CSV data is provided to this generation stage. Never invent coefficients
or encode assumed data values. Use only source columns and the mathematical model.
Try UTF-8-sig, UTF-8, GBK and Latin-1 only on decoding errors, matching the shared reader.
"""''' + src[b:])
        replace_function(exp, 27, 'get_csv_code', '''def get_csv_code(output, route, original_query, data_payload="", legacy_observation=""):
    if data_payload or legacy_observation:
        raise ValueError("Few-shot Only must not send complete source data to code generation")
    return _generate_code(output, route, original_query + "\\nSource Schema:\\n" +
                          source_schema_only(_CURRENT_DATASET_ADDRESS))''')
        replace(exp, 29, '    query, address = record["query"], record["dataset_address"]', '    global _CURRENT_DATASET_ADDRESS\n    query, address = record["query"], record["dataset_address"]\n    _CURRENT_DATASET_ADDRESS = address')
        replace(exp, 29, 'payload = formulation.get("observation", "") if mode == "planned" else ""', 'payload = ""  # Complete Observation is only used during mathematical modeling.')
        put(exp, 3, source(exp, 3) + '\nCSVQA_MODE_BY_ROUTE = {route: "direct" for route in WORKFLOW_ROUTES}\n')
        put(exp, 0, '# Large-scale OR: Few-shot Only\n\nKeep route model and code demonstrations. Python reads every field of every current-case CSV directly as text, with no selection or numeric rewrite, and supplies the complete data as Observation only to modeling. There is no CSVQA Tool, retrieval call, extraction plan, planner repair, or LLM data summary. Code generation receives the symbolic model, original query, paths and column names, and reads CSV files at runtime.\n')
    if method == "few_shot_only":
        put(exp, 11, source(exp, 11) + "\n" + (ROOT / "scripts/few_shot_symbolic_boundary_20261005.py").read_text())
        replace(exp, 3, "few_shot_only_v4", "few_shot_only_v5")
        replace(exp, 25, "abstract_plan=abstract_model_plan,", "abstract_plan=symbolic_model_for_codegen(abstract_model_plan),")
        replace(exp, 27, "return _generate_code(output, route,", "return _generate_code(symbolic_model_for_codegen(output), route,")
        exp["cells"][0]["source"].append("\nBefore code generation, Python forwards only variable, objective and constraint sections of the generated model; data-parameter/Data Mapping sections and echoed markdown tables are excluded. The exact complete Observation is unchanged and retained for audit. No LLM data summary, model regeneration or numeric rewrite is used.\n")
    append_cell(exp, '''RUN_API = False
if RUN_API:
    cases = attach_cached_classifications(load_benchmark())
    ablation_results = run_test(cases, output_csv=RESULTS_DIR / "automatic/results.csv")
    report_results(ablation_results, RESULTS_DIR / "automatic")
''')
    write(name, exp)

for name, remove_route in [('LOTO_Examples_Only_GPT4.1_Large-scale.ipynb', False),
                            ('LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb', True)]:
    exp = copy.deepcopy(base)
    old = json.loads((OUT / 'backups' / name).read_text())
    replace(exp, 3, f'NOTEBOOK_FILENAME = "{FULL}"', f'NOTEBOOK_FILENAME = "{name}"')
    method = 'examples_and_route' if remove_route else 'examples_only'
    replace(exp, 3, f'outputs/minimal_revision_20261005/full_{REVISION}', f'outputs/minimal_revision_20261005/{method}_{REVISION}')
    put(exp, 3, source(exp, 3) + f'\nLOTO_VARIANT = "{method}"\nLOTO_REMOVE_ROUTE = {remove_route}\nLOTO_BASE_NOTEBOOK = "{FULL}"\nLOTO_BASE_SHA256 = "{base_sha}"\n')
    exp['cells'].extend(copy.deepcopy(old['cells'][36:]))
    # Query-only references use a separate typed file in the baseline. Filter that
    # access too; leave the full notebook's query-only route unchanged.
    replace(exp, 25, '    documents = loader.load()', '    documents = filter_loto_query_only_documents(loader.load())')
    replace(exp, 25, '            "prefix": prefix,', '            "prefix": filter_loto_query_only_triggers(prefix),')
    put(exp, 39, source(exp, 39) + '''


def filter_loto_query_only_documents(documents):
    """Scope the separate query-only library by its original problem-type field."""
    target = require_loto_fold()["held_out_type"]
    def semantic_type(document):
        match = re.search(r"^problem type:\\s*(.*)$", document.page_content, re.MULTILINE | re.I)
        text = match.group(1).casefold() if match else "others"
        if "network revenue" in text:
            return "NRM"
        if "resource allocation" in text or "knapsack" in text:
            return "RA"
        if "assignment" in text:
            return "AP"
        if "facility location" in text:
            return "FLP"
        if "transport" in text or "transshipment" in text or "minimum-cost flow" in text:
            return "TP"
        return "Others"
    return [document for document in documents if semantic_type(document) != target]


def filter_loto_query_only_triggers(text):
    """Remove target-type fixed demonstrations while retaining generic contracts."""
    target = require_loto_fold()["held_out_type"]
    # Explicit structural types of the three fixed baseline demonstrations.
    trigger_types = {1: "RA", 2: "Mixture", 3: "Mixture"}
    for number, label in trigger_types.items():
        if label == target:
            text = re.sub(r"\\(" + str(number) + r"\\).*?(?=\\(\\d\\)|USER QUESTION:)",
                          "", text, flags=re.DOTALL)
    return text
''')
    replace(exp, 37, '    if not state["disabled_route"]:\n        return _LOTO_BASE_CLASSIFIER_PREFIX', '''    blocks = re.split(r"(?=Example \\d+)", few_shot_example)
    kept = [block for block in blocks if not re.search(
        r"Final Answer:\\s*" + re.escape(state["held_out_type"]) + r"\\b", block)]
    fold_prefix = _LOTO_BASE_CLASSIFIER_PREFIX.replace(few_shot_example, "".join(kept))
    if not state["disabled_route"]:
        return fold_prefix''')
    replace(exp, 37, 'return _LOTO_BASE_CLASSIFIER_PREFIX + f"""', 'return fold_prefix + f"""')
    replace(exp, 37, '"fixed_prompt_examples": "Preserved from the model-specific baseline"', '"fixed_prompt_examples": "Target-type classifier demonstrations removed; generic definitions retained"')
    replace(exp, 37, '    if LOTO_REMOVE_ROUTE:\n        # The target label is deliberately unavailable: ordinary class accuracy is not meaningful.\n        result["classification_correct"] = None', '    # Gold-label accuracy remains measurable; the forbidden label is expected to score zero.')
    replace(exp, 41, 'previous.get("record_status") == "completed" and previous.get("final_ok")', 'previous.get("record_status") in {"completed", "error"}')
    src = source(exp, 41)
    a = src.index('    if not LOTO_REMOVE_ROUTE:\n        scored = frame["classification_correct"].dropna()')
    b = src.index('    summary_rows.append({"metric": "Macro objective match', a)
    put(exp, 41, src[:a] + '''    summary_rows.append({"metric": "Classification", "correct": int(frame["classification_correct"].eq(True).sum()),
                         "total": len(frame), "accuracy": float(frame["classification_correct"].eq(True).mean())})
''' + src[b:])
    replace(exp, 41, '        cached = {record_key(record): record for record in load_records(output_csv)} if reuse_completed else {}', '''        existing = load_records(output_csv)
        if existing and not reuse_completed:
            raise ValueError("Use a fresh round directory; recorded LOTO attempts cannot be overwritten")
        cached = {record_key(record): record for record in existing}''')
    replace(exp, 41, '            previous = cached.get(key)', '''            previous = cached.get(key)
            if previous and previous.get("cache_fingerprint") != fingerprint:
                raise ValueError("LOTO code/data changed; use a fresh version directory")''')
    # Direct guard before any modeling/code generation uses a disallowed route.
    replace(exp, 37, '        record = _LOTO_BASE_EXECUTE_PIPELINE(case, forced_route=None)', '        record = _LOTO_BASE_EXECUTE_PIPELINE(case, forced_route=None)')
    put(exp, 0, f'# Large-scale OR: LOTO {"Examples And Route" if remove_route else "Examples Only"}\n\nBased on the same final full-model definitions. Remove the target semantic type from the effective classification reference (RAG_Examples_All), route modeling/code examples, and fixed classifier demonstrations. Generic class definitions and execution interfaces remain. {"Disable the target route and validate it before formulation; reclassify using only remaining allowed labels." if remove_route else "Retain every route; a route with an empty reference pool uses generic instructions and current-case data."} No gold label or reference objective is passed to generation.\n')
    put(exp, 36, '## Fold controls and route enforcement\n\nFilter target-type references before route normalization, remove its fixed classifier demonstrations, and clear reference caches at each fold. Generic definitions remain. Validate route availability before modeling.\n')
    put(exp, 40, '## Fold runner and reports\n\nrun_loto executes the seven disjoint semantic folds once per case. Preserve failures during resume. Every fold has its own manifest and results. Gold-label classification accuracy, solve success and Objective Match are separate; gold-label accuracy in route-removal folds is expected to be zero because the target label is forbidden.\n')
    put(exp, 43, '# Run loto_preflight() explicitly before enabling the experiment.\n')
    write(name, exp)

# Preserve the reviewed delivery switch without changing evaluated model definitions.
delivery_path = OUT / "delivery_control.json"
if delivery_path.exists():
    delivery = json.loads(delivery_path.read_text())
    if delivery.get("RUN_FORCED_ROUTES"):
        assert json.loads((OUT / "report/gate.json").read_text())["passed"]
        full_delivery = json.loads((ROOT / FULL).read_text())
        for index, cell_source in delivery["606_control_cells"].items():
            put(full_delivery, int(index), cell_source)
        write(FULL, full_delivery)

print('Updated five notebooks; all definition cells parse successfully.')
