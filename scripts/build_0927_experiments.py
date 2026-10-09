"""Build controlled experiments from the user-specified 0927 full notebook."""
import ast
import copy
import hashlib
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'
OLD_LOTO = ROOT / 'LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb'
BACKUP = ROOT / 'outputs/ablation_loto_0927/source_copies_before_rebuild'
BACKUP.mkdir(parents=True, exist_ok=True)
base = json.loads(BASE.read_text())
# The former copy supplies only the fold/report scaffolding, never model definitions.
if not (BACKUP / OLD_LOTO.name).exists():
    (BACKUP / OLD_LOTO.name).write_bytes(OLD_LOTO.read_bytes())
old = json.loads((BACKUP / OLD_LOTO.name).read_text())
newer = json.loads((ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_1006.ipynb').read_text())

def source(book, index):
    return ''.join(book['cells'][index]['source'])

def put(book, index, text):
    book['cells'][index]['source'] = text.splitlines(keepends=True)

def cell(text, kind='code', experiment=False):
    c = {'cell_type': kind, 'metadata': {'tags': ['experiment']} if experiment else {},
         'source': text.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c

def replace_function(text, name, replacement):
    node = next(n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    lines = text.splitlines(keepends=True)
    start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(lines[:start])+replacement.strip()+'\n'+''.join(lines[node.end_lineno:])

COMMON = '''
BASE_NOTEBOOK_PATH = PROJECT_ROOT / "LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb"
BASE_NOTEBOOK_SHA256 = hashlib.sha256(BASE_NOTEBOOK_PATH.read_bytes()).hexdigest()
CLASSIFICATION_RESULTS_PATH = PROJECT_ROOT / "outputs/final_101_V2/automatic/results.csv"
CLASSIFICATION_RESULTS_SHA256 = hashlib.sha256(CLASSIFICATION_RESULTS_PATH.read_bytes()).hexdigest()
MATCH_REL_TOL = 1e-4
MATCH_ABS_TOL = 1e-4
'''
CACHE = '''
def attach_cached_classifications(frame, classification_csv=CLASSIFICATION_RESULTS_PATH):
    """Read only predictions; verify exact questions and CSV paths before reusing them."""
    path = Path(classification_csv).resolve()
    required = {"problem_id", "query", "dataset_address", "predicted_label", "assigned_route", "experiment_mode", "forced_route"}
    cached = read_csv_compat(path, dtype=str, keep_default_na=False,
                             usecols=lambda name: name in required)
    if required.difference(cached.columns):
        raise ValueError("Classification cache is missing required prediction fields")
    if cached["problem_id"].duplicated().any() or cached["problem_id"].eq("").any():
        raise ValueError("Classification cache needs unique nonempty case IDs")
    if not cached["experiment_mode"].eq("automatic").all() or cached["forced_route"].ne("").any():
        raise ValueError("Use only the final full model's automatic classification")
    lookup = cached.set_index("problem_id").to_dict("index")
    result = frame.copy()
    labels, routes = [], []
    for case in result.to_dict("records"):
        row = lookup.get(str(case["problem_id"]))
        if row is None or row["query"] != str(case["Query"]) or resolve_dataset_address(row["dataset_address"]) != case["dataset_address"]:
            raise ValueError(f"Classification cache question/data mismatch: {case['problem_id']}")
        label = normalize_problem_class(row["predicted_label"])
        route = normalize_route(row["assigned_route"])
        if class_to_workflow_route(label) != route:
            raise ValueError(f"Cached label/route mismatch: {case['problem_id']}")
        labels.append(label); routes.append(route)
    result["cached_predicted_label"] = labels
    result["cached_assigned_route"] = routes
    return result


def classification_for_case(case):
    if "cached_predicted_label" not in case or "cached_assigned_route" not in case:
        raise ValueError("Attach final full-model classifications before running the ablation")
    label = normalize_problem_class(case["cached_predicted_label"])
    if class_to_workflow_route(label) != case["cached_assigned_route"]:
        raise ValueError("Cached classification route is inconsistent")
    return {"normalized_label": label}
'''
DIRECT = '''
def direct_source_payload(dataset_address, route):
    """No tool or LLM: preserve every parsed source row, column and cell as text."""
    tables = []
    for i, raw in enumerate(normalize_data_address(dataset_address).splitlines()):
        path = Path(raw)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        frame = read_csv_compat(path, dtype=str, keep_default_na=False)
        columns = list(frame.columns)
        tables.append({"table_id": f"file_{i}_view_0", "file_index": i,
            "file_name": path.name, "source": str(path.resolve()), "role": f"file_{i}",
            "columns": columns, "original_rows": len(frame), "returned_rows": len(frame),
            "filters": {"logic": "and", "conditions": []},
            "records": [{"source_row": int(index), "values": {column: str(row[column]) for column in columns}}
                        for index, row in frame.iterrows()]})
    if not tables:
        raise ValueError("No CSV data supplied")
    return {"route": normalize_route(route), "tables": tables, "relationships": [],
            "ignored_file_indices": [], "validation": {"status": "PYTHON_FULL_CSV"}}


def direct_source_observation(dataset_address, route="Others"):
    return json.dumps(direct_source_payload(dataset_address, route), ensure_ascii=False)


def data_schema_only(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {"tables": [{key: value for key, value in table.items() if key not in {"records", "filters"}}
                        for table in payload["tables"]]}


def direct_source_schema(dataset_address):
    return json.dumps(data_schema_only(direct_source_payload(dataset_address, "Others")), ensure_ascii=False)
'''
DIRECT_FORMULATE = '''
def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    """CSVQA is removed. Supply Python's complete raw Observation to modeling only."""
    observation = direct_source_observation(dataset_address, route)
    # Reference demonstrations retain their original content; the current task has no tool.
    model_prompt = prefix.replace("{{", "{").replace("}}", "}")
    contract = (
        "For the CURRENT task, CSVQA and all extraction planners are absent. "
        "Python has already read every CSV cell into the Observation below. "
        "Do not request a tool, a data extraction, an LLM data summary, or a rewritten dataset. "
        "Use the source values exactly. The demonstrations illustrate model structure only. "
        "Return the requested mathematical formulation directly; no Thought/Action protocol. "
        "Preserve the original objective sense, full expression, and constant terms. "
        "Do not replace it with an optimizer-equivalent objective with a different value."
    )
    response = make_llm().invoke([HumanMessage(content=model_prompt+"\\n\\n"+contract+
        "\\n\\nUser Description:\\n"+str(query)+"\\n\\nObservation (complete source CSV data):\\n"+observation)])
    return {"formulation": str(response.content), "observation": observation,
            "trace": {"status": "PYTHON_FULL_CSV", "planner_attempt_count": 0,
                      "payload_hash": hashlib.sha256(observation.encode()).hexdigest()}}
'''
DIRECT_CODE = '''
def _generate_code(output, route, original_query="", data_payload="", legacy_observation=""):
    """Do not append complete source Observation to a code-generation prompt."""
    route = normalize_route(route)
    parts = [ORIGINAL_CODE_PROMPT.format(output=output)]
    if original_query:
        parts.append(f"Original Query:\\n{original_query}")
    if data_payload:
        instructions = PLANNED_CODE_INSTRUCTIONS.replace(
            "The complete payload below is provided at execution as CSVQA_DATA.",
            "The full source data is supplied ONLY at execution as CSVQA_DATA; it is not in this prompt.")
        parts.extend([instructions, "CSVQA_DATA structural schema (no cell values):\\n"+
                      json.dumps(data_schema_only(data_payload), ensure_ascii=False)])
    else:
        parts.append(LEGACY_CODE_INSTRUCTIONS)
    parts.append(CSV_ROUTE_HINT[route])
    examples = retrieve_csv_code_example(route, original_query or output)
    if examples:
        parts.extend([
            "The following examples are coding references only. Reuse indexing and "
            "Gurobi modeling patterns, but never copy their coefficients, dimensions, "
            "identifiers, or data. The current formulation and current data are authoritative.", examples])
    parts.append(CSV_SOLVER_INSTRUCTIONS)
    response = make_llm().invoke([HumanMessage(content="\\n\\n".join(parts))])
    code = response.content
    print(code)
    return code
'''

def make(method, name):
    p = ROOT/name
    if p.exists() and not (BACKUP/name).exists():
        (BACKUP/name).write_bytes(p.read_bytes())
    book = copy.deepcopy(base)
    book['cells'] = book['cells'][:36]
    for c in book['cells']:
        if c['cell_type']=='code': c.update(execution_count=None,outputs=[])
    config = source(book,3).replace('"outputs/final_101_V2"',f'"outputs/ablation_loto_0927/{method}_v1"')
    config = config.replace('NOTEBOOK_FILENAME = "LEAN_LLM_OPT_4.1_Large-scale.ipynb"',f'NOTEBOOK_FILENAME = "{name}"')
    config += COMMON+f'\nEXPERIMENT_METHOD = "{method}"\n'
    put(book,3,config)
    book['metadata']['experiment_provenance'] = {'base_notebook':BASE.name,'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'method':method,'baseline_results':'outputs/final_101_V2','selection':'one preregistered implementation; all cases reported'}
    return book

rag_name='Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb'
rag=make('rag_only',rag_name)
put(rag,0,'''# Large-scale OR — RAG Only (0927)

Base: `LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb`; classification cache: `outputs/final_101_V2/automatic/results.csv`.

Removed from modeling/code prompts: NRM/RA/TP/AP/FLP demonstrations from `RAG_Examples_All.csv` (Query, Required Data, Label, Code); Others/Mixture abstract-plan and code demonstrations; query-only retrieved demonstrations and three fixed modeling examples. All 15 reference rows are excluded from modeling and coding, including AP row 0, UFLP/FLP row 1, NRM row 2, Mixture rows 3–10, Others rows 11–12, RA row 13, TP row 14 (zero-based).

Retained: the baseline classification predictions (including its classifier reference retrieval), CSVQA Tool, source-ordered full-file legacy evidence, NRM JSON extraction planner/executor/validation and baseline full-data fallback, Others Python CSV preview/statistics/name evidence and complete-file runtime reads. Removing route examples does not remove current-case data retrieval.

Model snapshot, temperature, SDK retries, original route modes, code execution, solver instructions and objective tolerances are copied from the full baseline. Run all 101 cases once and report every failure; do not optimize this ablation for lower accuracy.
''')
s=source(rag,15);s=replace_function(s,'retrieve_rag_examples','''def retrieve_rag_examples(route, query, k=1):
    """No route model/code demonstrations; CSVQA current-case retrieval is preserved."""
    return []''');put(rag,15,s)
# Current-case CSV modeling remains byte-for-byte identical. Query-only examples are removed too.
s=source(rag,25);s=replace_function(s,'get_others_without_CSV_response','''def get_others_without_CSV_response(query):
    response = make_llm().invoke([HumanMessage(content=
        "Formulate a complete mathematical optimization model using only this question. "
        "Include sets, parameters, variable domains, full objective and every constraint. "
        "Preserve the original objective value including constants. Return the model only.\\n"+str(query))])
    return str(response.content)''');put(rag,25,s)
put(rag,29,source(rag,29).replace('invoke_classifier(query) if forced_route is None else {}','classification_for_case(case) if forced_route is None else {}'))
rag['cells'] += [cell('## Reuse final full-model classification\nOnly prediction columns are loaded; models, answers and solve outputs are not reused.\n','markdown'),cell(CACHE),cell('''RUN_ABLATION = False
if RUN_ABLATION:
    ablation_cases = attach_cached_classifications(load_benchmark())
    ablation_results = run_test(ablation_cases, output_csv=RESULTS_DIR / "automatic/results.csv",
                               reuse_completed=False, rel_tol=MATCH_REL_TOL, abs_tol=MATCH_ABS_TOL)
    report_results(ablation_results, RESULTS_DIR / "automatic")
''',experiment=True)]

few_name='Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb'
few=make('few_shot_only',few_name)
put(few,0,'''# Large-scale OR — Few-shot Only (0927)

Use the same 101 questions, final full-model classification, model snapshot, reference-example selection, solver instructions and scoring tolerances as the 0927 full baseline.

Retain all route-specific modeling and code few-shot examples, including Others/Mixture abstract plans and code references. Remove CSVQA Tool, LLM current-case evidence retrieval, NRM extraction planning/execution and its fallback. Python reads every source CSV row and cell as text without selection, truncation, numerical rewriting, invented values, or an LLM data preprocessing call. This complete Observation goes to modeling only.

Code generation receives the resulting formulation and coding references. It never receives the complete raw Observation/records. NRM additionally receives table IDs, column names and row counts (no cell values); Python binds complete data only when executing the generated code. Others code generation receives schema metadata (no source cell values) and the abstract plan, and reads complete source files at runtime, as in the baseline.

Classification predictions are reused from `outputs/final_101_V2/automatic/results.csv`; all new model/solve outputs are computed. Record every attempt without selecting results to reduce accuracy.
''')
put(few,8,'## 4. Removed extraction planner\nFew-shot Only has no LLM CSV screening, extraction plan, plan executor, or plan fallback.\n')
put(few,9,'# CSVQA extraction planner and plan executor are intentionally absent.\n')
put(few,10,'## 5. Python source Observation\nRead the complete current-case CSV data exactly once per source read, with no LLM preprocessing.\n')
put(few,11,DIRECT)
put(few,15,replace_function(source(few,15),'formulate_with_csvqa',DIRECT_FORMULATE))
s=source(few,25)
s=s.replace('schema = csv_schema_preview(dataset_address, query=user_query)',
            'source_observation = direct_source_observation(dataset_address, "Others")\n        schema = source_observation\n        code_schema = direct_source_schema(dataset_address)')
s=s.replace('schema=schema,\n            abstract_plan=abstract_model_plan,','schema=code_schema,\n            abstract_plan=abstract_model_plan,')
s=s.replace('"observation": schema,','"observation": source_observation,').replace('"status": "LEGACY_SCHEMA"','"status": "PYTHON_FULL_CSV"')
s=s.replace('hashlib.sha256(schema.encode()).hexdigest()', 'hashlib.sha256(source_observation.encode()).hexdigest()')
put(few,25,s)
put(few,27,replace_function(source(few,27),'_generate_code',DIRECT_CODE))
put(few,29,source(few,29).replace('invoke_classifier(query) if forced_route is None else {}','classification_for_case(case) if forced_route is None else {}'))
# Remove dead source-cleaning/data-planning helpers that are no longer needed.
s=source(few,7)
for n in ['invoke_react_with_required_csvqa','_build_profile']:
    s=replace_function(s,n,f'def {n}(*args, **kwargs):\n    raise RuntimeError("CSVQA/extraction is removed in Few-shot Only")')
put(few,7,s)
few['cells'] += copy.deepcopy(rag['cells'][36:])

loto_name='LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb'
loto=make('examples_and_route',loto_name)
put(loto,0,'''# Large-scale OR — LOTO Examples And Route (0927)

Model/data/execution baseline: `LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb`. For each disjoint gold-type evaluation fold, remove that exact semantic type from classifier RefData and model/code reference libraries BEFORE merging Mixture and Others to the shared Others route. Remove fixed classifier/model demonstrations of that type too. Disable its executable route; therefore either a Mixture or an Others fold disables the shared Others route and both output labels.

Gold types choose evaluation folds only. The classification agent sees the current question and permitted labels, retrieves only remaining reference examples, and independently selects an allowed route. No reference model or expected objective enters generation. Validate the route before modeling, code generation and execution. Empty remaining example pools yield no examples, never refill with excluded types.

A route supplies reference style; the current query determines the actual objective, variables, domains and constraints. This generic instruction avoids forcing held-out problems into a mismatching archetype. Source-data access, model settings, solver instructions, execution and matching tolerance otherwise stay at the 0927 baseline. Report the single implementation's complete pass without answer-guided route selection or result selection.

Independent Variants and redundant-column switches below use the SAME LOTO boundaries and fresh classification for every fold/case.\n''')
put(loto,3,source(loto,3)+'''\nLOTO_VARIANT = "examples_and_route"
LOTO_REMOVE_ROUTE = True
LOTO_BASE_NOTEBOOK = BASE_NOTEBOOK_PATH.name
LOTO_BASE_SHA256 = BASE_NOTEBOOK_SHA256
''')
# Query-only library and fixed demonstrations also respect the fold boundary.
s=source(loto,25).replace('documents = loader.load()', 'documents = filter_loto_query_only_documents(loader.load())')
s=s.replace('"prefix": prefix,\n            "suffix": suffix,','"prefix": filter_loto_query_only_triggers(prefix),\n            "suffix": suffix,')
put(loto,25,s)
controls=source(old,37)
controls=controls.replace('"ollama_base_url": globals().get("OLLAMA_BASE_URL"),\n                "num_ctx": os.environ.get("LEAN_NUM_CTX", "131072"),\n                "num_predict": os.environ.get("LEAN_NUM_PREDICT", "8192"),\n                "seed": os.environ.get("LEAN_SEED", "42")','"model": MODEL_SNAPSHOT')
controls += '''
_LOTO_BASE_CLASSIFY = invoke_classifier
_LOTO_BASE_FORMULATE_CSVQA = formulate_with_csvqa
_LOTO_BASE_GENERATE_CODE = _generate_code


def invoke_classifier(query):
    result = _LOTO_BASE_CLASSIFY(query)
    if result["normalized_label"] not in loto_allowed_labels():
        raise DisabledRouteError("Classifier emitted an unavailable label; no alternative route is assigned")
    assert_loto_route_allowed(class_to_workflow_route(result["normalized_label"]))
    return result


def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    assert_loto_route_allowed(route)
    prefix += ("\\nThe selected workflow is a source of reference patterns only. "
        "Infer the current mathematical structure entirely from the user's question and current CSV data. "
        "Do not force the question into the selected route's archetype or copy an example's extra constraints. "
        "Retain every question-specific decision family, coupling, domain and objective term, "
        "even when its structure differs from the remaining reference examples.")
    return _LOTO_BASE_FORMULATE_CSVQA(query, dataset_address, route, system_prompt, tool_description, prefix, suffix)


def _generate_code(output, route, original_query="", data_payload="", legacy_observation=""):
    assert_loto_route_allowed(route)
    return _LOTO_BASE_GENERATE_CODE(output, route, original_query, data_payload, legacy_observation)
'''
filtered=source(old,39)
filtered=filtered.replace('route = normalize_route(route)\n    examples = load_rag_examples(route)',
                        'route = assert_loto_route_allowed(route)\n    examples = load_rag_examples(route)')
filtered=filtered.replace('if "resource allocation" in text or "knapsack" in text:', 'if "resource allocation" in text:')
filtered=filtered.replace('if "transport" in text or "transshipment" in text or "minimum-cost flow" in text:', 'if "transportation" in text:')
# Minimum-cost-flow and transshipment do not match the baseline definition of clean TP.
# They are outside canonical clean classes; no ambiguity allows a target example through.
runner=source(old,41)
start=runner.index('def run_loto(')
runner=runner[:start]+runner[start:].replace('def run_loto(*, folds=None, rows=None, reuse_completed=True, continue_on_error=True,',
    'def run_loto(test=None, *, output_dir=None, folds=None, rows=None, reuse_completed=True, continue_on_error=True,')
runner=runner.replace('frame, _ = loto_preflight()', 'frame = load_benchmark() if test is None else prepare_cases(test)\n    validate_loto_cases(frame)\n    destination = RESULTS_DIR if output_dir is None else Path(output_dir)')
runner=runner.replace('folder = RESULTS_DIR / held_out','folder = destination / held_out').replace('loto_report(results, RESULTS_DIR)','loto_report(results, destination)')
runner=runner.replace('raise ValueError("ROWS must be distinct zero-based benchmark row indices from 0 to 100")',
                      'raise ValueError("ROWS must be distinct valid zero-based row indices")')
runner += '''

def validate_loto_cases(frame):
    if frame.empty or frame["problem_id"].duplicated().any():
        raise ValueError("LOTO requires nonempty unique case IDs")
    if frame["true_label"].isna().any() or not frame["true_label"].isin(CLASS_LABELS).all():
        raise ValueError("Every evaluation case needs a declared semantic type for fold grouping")
    for address in frame["dataset_address"]:
        if not address or any(not Path(raw).is_file() for raw in address.splitlines()):
            raise FileNotFoundError(f"Missing current-case CSV data: {address}")


_BASE_PREPARE_CASES = prepare_cases

def prepare_cases(test, *, dataset_root=None):
    if dataset_root is None:
        return _BASE_PREPARE_CASES(test)
    frame = test.copy()
    address_column = "dataset_address" if "dataset_address" in frame else "Dataset_address"
    def resolve_relocated(value):
        paths = []
        root = Path(dataset_root).resolve()
        for raw in normalize_data_address(value).splitlines():
            path = Path(raw)
            standard = path if path.is_absolute() else PROJECT_ROOT / path
            if standard.is_file():
                paths.append(str(standard.resolve())); continue
            candidates = [root/path, root.joinpath(*path.parts[1:])]
            matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely resolve relocated data: {raw}")
            paths.append(str(matches[0]))
        return "\\n".join(paths)
    frame[address_column] = frame[address_column].map(resolve_relocated)
    return _BASE_PREPARE_CASES(frame)
'''
loto['cells'] += [cell('## Fold controls and route guards\nExclude the held-out semantic type from every effective reference pool before route normalization.\n','markdown'),cell(controls),
    cell('## Filtered reference access\nNo excluded-type rows or disabled-route examples can enter generation.\n','markdown'),cell(filtered),
    cell('## LOTO runner, scoring and dataset loaders\nOnly the scorer reads expected objectives; classifiers and formulators receive question/data only.\n','markdown'),cell(runner),
    cell('## Run the 101-case LOTO experiment\nSwitch on for all seven disjoint folds. Resume preserves successful and failed attempts.\n','markdown'),
    cell('''RUN_LOTO = False
FOLDS = None
ROWS = None
if RUN_LOTO:
    loto_preflight()
    loto_results = run_loto(folds=FOLDS, rows=ROWS, rel_tol=MATCH_REL_TOL, abs_tol=MATCH_ABS_TOL)
''',experiment=True)]
variants=source(newer,45).replace('variants_results = run_test(','variants_results = run_loto(')
variants=variants.replace('variants_cases, output_csv=VARIANTS_OUTPUT_DIR / "results.csv",','variants_cases, output_dir=VARIANTS_OUTPUT_DIR,')
variants=variants.replace('report_results(variants_results, VARIANTS_OUTPUT_DIR)','loto_report(variants_results, VARIANTS_OUTPUT_DIR)')
columns=source(newer,47).replace('sheet_results = run_test(column_cases, output_csv=sheet_output_dir / "results.csv")',
                               'sheet_results = run_loto(column_cases, output_dir=sheet_output_dir)')
columns=columns.replace('report_results(sheet_results, sheet_output_dir)','loto_report(sheet_results, sheet_output_dir)')
loto['cells'] += [cell('## Variants benchmark\nIndependent switch; fresh LOTO classification and route restrictions for all 36 cases.\n','markdown'),cell(variants,experiment=True),
    cell('## Redundant-column benchmark\nIndependent switch; nine sheets, each scored with the same LOTO rules. Paths resolve under the workbook folder.\n','markdown'),cell(columns,experiment=True)]

for name,book in [(rag_name,rag),(few_name,few),(loto_name,loto)]:
    for c in book['cells']:
        c.setdefault('id', uuid.uuid4().hex[:8])
        if c['cell_type']=='code': compile(''.join(c['source']),name,'exec')
    (ROOT/name).write_text(json.dumps(book,ensure_ascii=False,indent=1)+'\n')
    print(name, 'base',BASE.name,'cells',len(book['cells']))
