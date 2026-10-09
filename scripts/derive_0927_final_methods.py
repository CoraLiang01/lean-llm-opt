"""Deliver the evaluated full revision and derive only the specified removals."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import uuid
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/optimization_0927_20261006'
SNAP = OUT/'source_snapshots/before'
FULL_NAME = 'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb'
NAMES = {'rag_only':'Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb',
         'few_shot_only':'Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb',
         'examples_and_route':'LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb'}


def source(book, index): return ''.join(book['cells'][index]['source'])
def put(book, index, text): book['cells'][index]['source'] = text.splitlines(True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_function(text, name, replacement):
    node = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name == name)
    lines = text.splitlines(True)
    start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(lines[:start])+replacement.strip()+'\n'+''.join(lines[node.end_lineno:])


def cell(text, experiment=False):
    return nbformat.v4.new_code_cell(text, metadata={'tags':['experiment']} if experiment else {})


def save(path, book):
    for i,c in enumerate(book['cells']):
        c.setdefault('id',uuid.uuid4().hex[:8])
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), str(path)+f':cell{i}', 'exec')
            c['execution_count'] = None; c['outputs'] = []
    path.write_text(json.dumps(book, ensure_ascii=False, indent=1)+'\n')
    nbformat.validate(nbformat.read(path,as_version=4))


# The original builder has top-level file mutations; read its string constants as data.
strings = {}
for node in ast.parse((ROOT/'scripts/build_0927_experiments.py').read_text()).body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
        for target in node.targets:
            if isinstance(target,ast.Name):strings[target.id]=node.value.value

REACT_HELPER = '''
def formulate_from_python_observation(query, observation, prefix, suffix):
    """Retain the original ReAct parser/executor after removing all data tools."""
    from langchain_classic.agents import AgentExecutor, ZeroShotAgent
    contract = (
        "For the CURRENT task there are no callable tools. Python has already supplied "
        "the full source Observation below. Reference examples illustrate mathematical "
        "structure only; their tool actions are unavailable here. Do not request CSVQA, "
        "a data extraction plan, a data summary or a rewritten dataset. Use original "
        "source values exactly and return the required mathematical model. "
        "Preserve original objective sense, constants, variable domains and boundaries. "
        "Conclude with Final Answer: followed by the model."
    )
    data_prefix = prefix + "\\n" + contract + "\\nObservation (complete source CSV data):\\n" + escape_braces(observation)
    prompt = ZeroShotAgent.create_prompt(
        tools=[], prefix=data_prefix, suffix=suffix,
        format_instructions="No tools are available. Use Thought: followed by Final Answer:; do not output Action.",
        input_variables=["input", "agent_scratchpad"],
    )
    # initialize_agent rejects an empty tool list; constructing its SAME agent class
    # directly keeps the ReAct engine and removes only the ablated tool dependency.
    agent = AgentExecutor(
        agent=ZeroShotAgent(llm_chain=LLMChain(llm=make_llm(), prompt=prompt), allowed_tools=[]),
        tools=[], verbose=True, handle_parsing_errors=True,
        return_intermediate_steps=True, early_stopping_method="generate",
    )
    result = agent.invoke(query)
    return {"formulation": result["output"], "observation": observation,
            "trace": {"status": "PYTHON_FULL_CSV", "planner_attempt_count": 0,
                      "react_data_tool_calls": len(result.get("intermediate_steps", [])),
                      "payload_hash": hashlib.sha256(observation.encode()).hexdigest()}}


def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    observation = direct_source_observation(dataset_address, route)
    return formulate_from_python_observation(query, observation, prefix, suffix)
'''


QUERY_ONLY_RAG = '''def get_others_without_CSV_response(query):
    """No reference/fixed demonstrations; keep the ReAct engine for a query-only task."""
    from langchain_classic.agents import AgentExecutor, ZeroShotAgent
    prompt = ZeroShotAgent.create_prompt(
        tools=[], prefix="Formulate the optimization problem entirely from the current question. "
        "Include index sets, parameters, variable domains, the original objective and all constraints. "
        "Preserve objective constants and units. No reference examples or tools are available.",
        suffix="User Description: {input}\\n{agent_scratchpad}",
        format_instructions="Use Thought: followed by Final Answer: containing the mathematical model.",
        input_variables=["input", "agent_scratchpad"],
    )
    agent = AgentExecutor(agent=ZeroShotAgent(llm_chain=LLMChain(llm=make_llm(),prompt=prompt),allowed_tools=[]),
        tools=[],verbose=True,handle_parsing_errors=True,return_intermediate_steps=True,early_stopping_method="generate")
    return agent.invoke(query)["output"]
'''


def main():
    progress = json.loads((OUT/'full_v1/progress.json').read_text())
    assert sum(r['evaluated'] for r in progress) == 452
    assert all(r['evaluated'] == r['expected'] for r in progress)
    assert all(r['objective_match'] >= (32 if r['benchmark']=='variants' else 31)
               for r in progress if r['benchmark']!='main'), 'Full targets must pass before deriving methods'
    evaluated = json.loads((OUT/'full_v1/frozen_notebook.ipynb').read_text())
    original = json.loads((SNAP/FULL_NAME).read_text())
    full = copy.deepcopy(original)
    full['cells'][:36] = copy.deepcopy(evaluated['cells'][:36])
    config = source(full,3).replace('LEAN_LLM_OPT_4.1_Large-scale_0927_optimized.ipynb', FULL_NAME)
    config = '\n'.join(line for line in config.splitlines() if not line.startswith('LOTO_'))+'\n'
    put(full,3,config)
    put(full,0,'# LEAN-LLM-OPT — 0927 最小修改版\n\n'
        '原有 ReAct、NRM planned 与其余 route legacy 保留。对应完整冻结评测：'
        'outputs/optimization_0927_20261006/full_v1。Variants 34/36；九个冗余列 sheet 全部至少 32/35。'
        '101 题 Objective match 92/101、Solved 96/101；历史基准为 95/101、99/101，退步已完整记录。'
        '所有修改前代码、失败与不匹配均保留。\n')
    for i in [37,39,41,43]:
        s=source(full,i).replace('RUN_AUTOMATIC = True','RUN_AUTOMATIC = False').replace('RUN_FORCED_ROUTES = True','RUN_FORCED_ROUTES = False')
        put(full,i,s);full['cells'][i]['metadata']['tags']=['experiment']
    # Keep the original main/606/other/single-case cell order; add dataset loaders afterwards.
    full['cells'].append(nbformat.v4.new_markdown_cell('## Additional benchmark loaders\nOriginal data and stable Variant/sheet IDs; no gold model enters generation.'))
    full['cells'] += copy.deepcopy(evaluated['cells'][36:])
    full['cells'] += [cell('''RUN_VARIANTS = False
if RUN_VARIANTS:
    rows = load_variants_for_baseline(VARIANTS_FILE)
    results = run_test(rows, output_csv=RESULTS_DIR / "additional_benchmarks/variants/results.csv")
    report_results(results, RESULTS_DIR / "additional_benchmarks/variants")
''',True),cell('''RUN_REDUNDANT_COLUMNS = False
if RUN_REDUNDANT_COLUMNS:
    by_sheet = load_redundant_sheets_for_baseline(COLUMNS_FILE, COLUMN_SHEETS)
    for sheet_name, rows in by_sheet.items():
        folder = RESULTS_DIR / "additional_benchmarks/columns" / sheet_name
        results = run_test(rows, output_csv=folder / "results.csv")
        report_results(results, folder)
''',True)]
    full['metadata']['experiment_provenance']={'revision':'minimal-v1','evaluated_frozen_notebook':str(OUT/'full_v1/frozen_notebook.ipynb'),
        'evaluated_frozen_sha256':sha(OUT/'full_v1/frozen_notebook.ipynb'),'original_full_sha256':sha(SNAP/FULL_NAME),
        'original_archive':str(SNAP/FULL_NAME),'delivery_changes':'Notebook filename, switches, documentation and experiment entry cells only'}
    save(ROOT/FULL_NAME,full)
    # A prediction-only cache has no gold label, reference objective or model output columns.
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from evaluate_0927_optimization import namespace
    ns=namespace(OUT/'full_v1/frozen_notebook.ipynb')
    records=ns['load_records'](OUT/'full_v1/automatic/results.csv')
    records=[r for r in records if r['benchmark']=='main']
    assert len(records)==101 and all(r.get('predicted_label') and r.get('assigned_route') for r in records)
    cache=OUT/'full_v1/classification_main.csv'
    fields=['problem_id','query','dataset_address','predicted_label','assigned_route','experiment_mode','forced_route']
    with cache.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    guidance_node=next(n for n in ast.parse(source(evaluated,27)).body if isinstance(n,ast.FunctionDef) and n.name=='_generate_code')
    guidance=[n.value.args[0].value for n in ast.walk(guidance_node)
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
        and n.value.func.attr=='append' and n.value.args and isinstance(n.value.args[0],ast.Constant)
        and isinstance(n.value.args[0].value,str) and 'Mandatory result interface:' in n.value.args[0].value]
    assert len(guidance)==1
    delivered={FULL_NAME:sha(ROOT/FULL_NAME)}
    for method,name in NAMES.items():
        old=json.loads((SNAP/name).read_text())
        book=copy.deepcopy(old);book['cells'][:36]=copy.deepcopy(full['cells'][:36])
        cfg=source(book,3).replace(f'NOTEBOOK_FILENAME = "{FULL_NAME}"',f'NOTEBOOK_FILENAME = "{name}"')
        cfg=cfg.replace('outputs/optimization_0927_20261006/full_v1',f'outputs/optimization_0927_20261006/{method}_final')
        cfg=cfg.replace('outputs/final_101_V2/automatic/results.csv','outputs/optimization_0927_20261006/full_v1/classification_main.csv')
        if method=='examples_and_route':
            cfg += '\nLOTO_VARIANT = "examples_and_route"\nLOTO_REMOVE_ROUTE = True\nLOTO_BASE_NOTEBOOK = BASE_NOTEBOOK_PATH.name\nLOTO_BASE_SHA256 = BASE_NOTEBOOK_SHA256\n'
        put(book,3,cfg)
        put(book,0,source(old,0)+'\n\n## Final common baseline\n'
            'Shared definitions derive from the evaluated minimal full revision, with the same model, '
            'solver/executor/tolerance and HTTP policy. The full 101 predictions are reused by the two ablations; '
            'LOTO reclassifies under its masks. Original 303 results stay in outputs/ablation_loto_0927. '
            'Current results go to outputs/optimization_0927_20261006/'+method+'_final.\n')
        if method!='examples_and_route':
            put(book,29,source(book,29).replace('invoke_classifier(query) if forced_route is None else {}',
                'classification_for_case(case) if forced_route is None else {}'))
        if method=='rag_only':
            s=replace_function(source(book,15),'retrieve_rag_examples','def retrieve_rag_examples(route, query, k=1):\n    return []')
            put(book,15,s);put(book,25,replace_function(source(book,25),'get_others_without_CSV_response',QUERY_ONLY_RAG))
        elif method=='few_shot_only':
            put(book,8,'## Removed current-case extraction planner\nPython supplies complete source data; no CSVQA/planner or LLM data preprocessing.\n')
            put(book,9,'# CSVQA extraction planner/executor removed for this ablation.\n')
            put(book,10,'## Complete Python Observation\nNo row/column/cell filtering, summarizing, rewriting or invented values.\n')
            put(book,11,strings['DIRECT'])
            s=replace_function(source(book,15),'formulate_with_csvqa',REACT_HELPER)
            put(book,15,s)
            s=source(book,25).replace('schema = csv_schema_preview(dataset_address, query=user_query)',
                'source_observation = direct_source_observation(dataset_address, "Others")\n        schema = source_observation\n        code_schema = direct_source_schema(dataset_address)')
            s=s.replace('schema=schema,\n            abstract_plan=abstract_model_plan,','schema=code_schema,\n            abstract_plan=abstract_model_plan,')
            s=s.replace('"observation": schema,','"observation": source_observation,').replace('"status": "LEGACY_SCHEMA"','"status": "PYTHON_FULL_CSV"')
            s=s.replace('hashlib.sha256(schema.encode()).hexdigest()','hashlib.sha256(source_observation.encode()).hexdigest()')
            put(book,25,s)
            direct_code=strings['DIRECT_CODE'].replace('    response = make_llm().invoke(',
                '    parts.append('+repr(guidance[0])+')\n    response = make_llm().invoke(')
            direct_code=direct_code.replace('    route = normalize_route(route)', '''    route = normalize_route(route)
    if legacy_observation:
        raise ValueError("Complete Observation must not enter the code-generation function")
    if data_payload:
        schema = json.loads(data_payload) if isinstance(data_payload, str) else data_payload
        if any("records" in table for table in schema["tables"]):
            raise ValueError("Code generation accepts structural schema only, never source records")''',1)
            put(book,27,replace_function(source(book,27),'_generate_code',direct_code))
            pipeline=source(book,29)
            pipeline=pipeline.replace('        record["pipeline_stage"] = "code_generation"',
                '        code_schema = json.dumps(data_schema_only(payload), ensure_ascii=False) if payload else ""\n'
                '        record["pipeline_stage"] = "code_generation"')
            pipeline=pipeline.replace('route, query, data_payload=payload,','route, query, data_payload=code_schema,')
            pipeline=pipeline.replace('legacy_observation=formulation.get("observation", "")\n'
                '                    if route == "RA" and mode == "legacy" else "",','legacy_observation="",')
            put(book,29,pipeline)
            s=source(book,7)
            for fn in ['invoke_react_with_required_csvqa','_build_profile']:
                s=replace_function(s,fn,f'def {fn}(*args, **kwargs):\n    raise RuntimeError("CSVQA/extraction is removed in Few-shot Only")')
            put(book,7,s)
        else:
            s=source(book,25).replace('documents = loader.load()','documents = filter_loto_query_only_documents(loader.load())')
            s=s.replace('"prefix": prefix,\n            "suffix": suffix,','"prefix": filter_loto_query_only_triggers(prefix),\n            "suffix": suffix,')
            put(book,25,s)
            # These cells contain loaders as well as false switches; definitions must be available to the evaluator.
            for i in [45,47]:book['cells'][i]['metadata']['tags']=[]
        book['metadata']['experiment_provenance']={'method':method,'final_full_notebook':FULL_NAME,
            'final_full_sha256':sha(ROOT/FULL_NAME),'evaluated_full_sha256':sha(OUT/'full_v1/frozen_notebook.ipynb'),
            'classification_cache_sha256':sha(cache),'revision':'minimal-v1-derived','selection':'one complete pass; all failures retained'}
        save(ROOT/name,book);delivered[name]=sha(ROOT/name)
    (OUT/'delivery_control.json').write_text(json.dumps({'version':'minimal-v1',
        'full_targets':progress,'delivered_sha256':delivered,'evaluated_full_sha256':sha(OUT/'full_v1/frozen_notebook.ipynb'),
        'classification_main_cache':str(cache),'classification_main_sha256':sha(cache),
        'historical_original':str(SNAP/FULL_NAME),'full_algorithm_delivery':'same function bodies; filename, switches, descriptions and run entry cells adjusted'},ensure_ascii=False,indent=2))
    print(json.dumps(delivered,indent=2))


if __name__=='__main__':main()
