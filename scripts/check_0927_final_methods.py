"""Read-only provenance, common-engine, and experiment-boundary checks; no API calls."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import nbformat
import pandas as pd

from evaluate_0927_optimization import namespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/optimization_0927_20261006'
RUNS = {'rag_only': 'rag_only_final', 'few_shot_only': 'few_shot_only_final_v2',
        'examples_and_route': 'examples_and_route_final'}
NAMES = {'full': 'LEAN_LLM_OPT_4.1_Large-scale_0927.ipynb',
    'rag_only': 'Ablation_Study_Large_Scale_Or_RAG_Only_0927.ipynb',
    'few_shot_only': 'Ablation_Study_Large_Scale_Or_Few-shot_Only_0927.ipynb',
    'examples_and_route': 'LOTO_Examples_And_Route_GPT4.1_Large-scale_0927.ipynb'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path, limit=None):
    book = json.loads(Path(path).read_text())
    found = {}
    for cell in book['cells'][:limit]:
        if cell['cell_type'] != 'code': continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                found[node.name] = ast.dump(node, include_attributes=False)
    return found


def messages(folder):
    file = folder / 'prompts.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            for group in json.loads(line)['messages']:
                for message in group:
                    yield message['content']


def check_complete_payload(payload):
    cells = 0
    assert payload['validation']['status'] == 'PYTHON_FULL_CSV'
    assert not payload['ignored_file_indices']
    for table in payload['tables']:
        frame = pd.read_csv(table['source'], dtype=str, keep_default_na=False)
        assert table['columns'] == list(frame.columns)
        assert table['original_rows'] == table['returned_rows'] == len(frame)
        assert len(table['records']) == len(frame)
        for index, actual in enumerate(table['records']):
            assert actual['source_row'] == index
            expected = {column: str(value) for column, value in frame.iloc[index].items()}
            assert actual['values'] == expected
            cells += len(expected)
    return cells


def payload_in_prompt(text):
    for marker in ['Observation (complete source CSV data):\n', 'CSV Schema:\n']:
        index = text.find(marker)
        if index < 0: continue
        value = text[index+len(marker):].lstrip()
        if not value.startswith('{'): continue
        payload, _ = json.JSONDecoder().raw_decode(value)
        if payload.get('validation', {}).get('status') == 'PYTHON_FULL_CSV':
            return payload
    return None


def main():
    control = json.loads((OUT/'delivery_control.json').read_text())
    audit = {'delivery_hashes': {}, 'input_hashes': {}, 'function_comparisons': {}, 'static_boundaries': {}, 'runtime': {}}
    full = namespace(OUT/'full_v1/frozen_notebook.ipynb')
    full_delivered = namespace(ROOT/NAMES['full'])
    evaluated = definitions(OUT/'full_v1/frozen_notebook.ipynb', 36)
    delivered = definitions(ROOT/NAMES['full'], 36)
    assert evaluated == delivered, 'Delivered full function AST differs from evaluated snapshot'
    audit['full_function_ast_including_constants_matches_frozen_run'] = True
    constants = ['MODEL_SNAPSHOT','EMBEDDING_MODEL','CSV_SOLVER_INSTRUCTIONS','CSVQA_MODE_BY_ROUTE',
        'CSV_ROUTE_HINT','PLANNED_CODE_INSTRUCTIONS','LEGACY_CODE_INSTRUCTIONS','ORIGINAL_CODE_PROMPT',
        'prefix','few_shot_example','NRM_RETRY_ON_TRUNCATION']
    for name in constants:
        if name in full: assert full_delivered[name] == full[name], f'Delivered full changed {name}'
    audit['full_prompt_and_mode_constants_match_frozen_run'] = True
    spaces = {'full': full}
    for method, name in NAMES.items():
        path = ROOT/name
        assert sha(path) == control['delivered_sha256'][name]
        book = nbformat.read(path, as_version=4); nbformat.validate(book)
        assert all(not c.outputs and c.execution_count is None for c in book.cells if c.cell_type == 'code')
        audit['delivery_hashes'][name] = sha(path)
        if method != 'full': spaces[method] = namespace(path)
    common = ['execute_code', '_source_candidate', 'objective_is_correct', 'make_llm',
              'read_csv_compat', 'get_embeddings', 'normalize_data_address']
    for method in RUNS:
        current = definitions(ROOT/NAMES[method], 36)
        checked = []
        for name in common:
            if name in evaluated:
                assert current[name] == evaluated[name], f'{method}: changed common function {name}'
                checked.append(name)
        ns = spaces[method]
        for name in ['MODEL_SNAPSHOT', 'EMBEDDING_MODEL', 'CSV_SOLVER_INSTRUCTIONS', 'CSVQA_MODE_BY_ROUTE',
                     'NRM_RETRY_ON_TRUNCATION']:
            assert ns[name] == full[name], f'{method}: changed {name}'
        assert ns['make_llm']().max_retries == full['make_llm']().max_retries == 1
        audit['function_comparisons'][method] = checked

    rag = spaces['rag_only']; few = spaces['few_shot_only']; loto = spaces['examples_and_route']
    ragdefs = definitions(ROOT/NAMES['rag_only'], 36)
    for name in ['build_csvqa_components', '_build_source_documents', 'plan_csv_extraction',
                 'execute_csv_plan', 'formulate_with_csvqa']:
        if name in evaluated:
            assert evaluated[name] == ragdefs[name], f'RAG Only changed retained function {name}'
    for route in full['WORKFLOW_ROUTES']:
        assert rag['retrieve_rag_examples'](route, 'boundary audit query', k=3) == []
    removed = pd.read_csv(ROOT/'Large_Scale_Or_Files/RAG_Examples_All.csv', dtype=str, keep_default_na=False)
    audit['static_boundaries']['rag_only'] = {
        'removed_reference_rows': [{'row': int(i), 'type': row['Type'], 'prompt': row['prompt']}
                                   for i,row in removed.iterrows()],
        'retained': ['source CSV row documents and embeddings', 'CSVQA ReAct tool',
                     'legacy retrieval and extraction', 'NRM planned extraction and Python execution',
                     'Others structural source profiling and complete runtime CSV reads']}
    for name in ['plan_csv_extraction', 'execute_csv_plan', 'build_csvqa_components']:
        assert name not in few, f'Few-shot Only retained {name}'
    for kwargs in [{'legacy_observation': 'raw data'},
                   {'data_payload': {'tables': [{'records': []}]}}]:
        try: few['_generate_code']('model', 'TP', **kwargs)
        except ValueError: pass
        else: raise AssertionError('Raw data accepted by code-generation entry')
    from langchain_core.language_models.fake import FakeListLLM
    original = few['make_llm']
    try:
        few['make_llm'] = lambda *a,**kw: FakeListLLM(responses=['Thought: Formulate from exact observation.\nFinal Answer: x >= 0'])
        result = few['formulate_from_python_observation']('query', '{"exact":"001"}', 'Modeler',
            'Question: {input}\n{agent_scratchpad}')
        assert result['trace']['react_data_tool_calls'] == 0
        assert result['observation'] == '{"exact":"001"}'
    finally: few['make_llm'] = original
    audit['static_boundaries']['few_shot_only'] = {
        'raw_data_rejected_at_codegen_function_entry': True,
        'tool_free_react_agent_verified': True,
        'extraction_functions_absent': True,
        'scope': 'raw complete CSV Observation excluded; normal formulation and retained reference examples still passed'}

    guards = []
    for label in loto['CLASS_LABELS']:
        manifest = loto['set_loto_fold'](label)
        disabled = manifest['disabled_workflow_route']
        assert disabled not in manifest['allowed_routes']
        remaining = manifest['remaining_example_indices']
        assert all(loto['normalize_problem_class'](removed.loc[i,'Type']) != label for i in remaining)
        for call in [lambda: loto['invoke_formulation'](disabled, 'query', 'not opened'),
                     lambda: loto['_generate_code']('model', disabled),
                     lambda: loto['retrieve_rag_examples'](disabled, 'query', k=3)]:
            try: call()
            except loto['DisabledRouteError']: pass
            else: raise AssertionError('Disabled route passed a pre-execution guard')
        guards.append({'held_out': label, 'disabled': disabled, 'removed_rows': [x['row_index'] for x in manifest['removed_examples']]})
    audit['static_boundaries']['loto'] = guards

    cache = pd.read_csv(OUT/'full_v1/classification_main.csv', dtype=str, keep_default_na=False)
    assert len(cache) == 101
    assert not set(cache.columns) & {'true_label','label_objective','generated_model','solve_code','solution_correct'}
    cache = cache.set_index('problem_id')
    baseline = full['load_benchmark']()
    for method in ['rag_only','few_shot_only']:
        attached = spaces[method]['attach_cached_classifications'](baseline)
        assert len(attached) == 101
        for row in attached.to_dict('records'):
            expected = cache.loc[row['problem_id']]
            actual = spaces[method]['classification_for_case'](row)
            assert actual['normalized_label'] == expected['predicted_label']
    decoder = json.JSONDecoder()
    for method, run in RUNS.items():
        root = OUT/run
        manifest = json.loads((root/'run_manifest.json').read_text())
        for path, expected in manifest['input_sha256'].items():
            assert sha(path) == expected, f'Input changed during evaluation: {path}'
        audit['input_hashes'][run] = {'verified_inputs': len(manifest['input_sha256']), 'unchanged': True}
        ns = spaces[method]
        records = ns['load_records'](root/'automatic/results.csv')
        seen = set(); result = {'evaluated': len(records), 'complete': len(records) == 101,
                                'violations': [], 'api_retry_count': 0, 'error_types': {}}
        failures = Counter(); full_observations = 0; observed_cells = 0; code_prompts = 0
        for record in records:
            id = record['problem_id']; assert id not in seen; seen.add(id)
            folder = root/'attempts'/id
            assert (folder/'result.json').exists()
            assert record['notebook_source_sha256'] == sha(root/'frozen_notebook.ipynb')
            result['api_retry_count'] += int(record.get('api_retry_count') or 0)
            if record.get('execution_error_type'): failures[record['execution_error_type']] += 1
            if method != 'examples_and_route':
                expected = cache.loc[id]
                assert record['predicted_label'] == expected['predicted_label']
                assert record['assigned_route'] == expected['assigned_route']
                assert record['query'] == expected['query']
                assert record['dataset_address'] == expected['dataset_address']
            else:
                manifest = json.loads((folder/'fold_manifest.json').read_text())
                assert record['held_out_type'] == manifest['held_out_type'] == record['true_label']
                assert record.get('assigned_route') != manifest['disabled_workflow_route']
                if record.get('predicted_label'):
                    assert record['predicted_label'] in manifest['allowed_labels']
                assert all(loto['normalize_problem_class'](removed.loc[i,'Type']) != record['true_label']
                           for i in manifest['remaining_example_indices'])
            if method == 'few_shot_only':
                observed = False
                for text in messages(folder):
                    is_code = 'Mathematical Optimization Model:' in text or 'Your task is to strictly follow the User Query' in text
                    if is_code:
                        code_prompts += 1
                        assert 'Observation (complete source CSV data):' not in text
                        assert 'Complete CSVQA_DATA JSON:' not in text
                        assert 'Complete LEGACY_RECORDS JSON:' not in text
                        marker = 'CSVQA_DATA structural schema (no cell values):\n'
                        if marker in text:
                            schema, _ = decoder.raw_decode(text.split(marker,1)[1].lstrip())
                            assert all('records' not in t for t in schema['tables'])
                    elif not observed:
                        payload = payload_in_prompt(text)
                        if payload is not None:
                            observed_cells += check_complete_payload(payload)
                            full_observations += 1; observed = True
                if not observed and record.get('pipeline_stage') != 'classification':
                    result['violations'].append({'id': id, 'reason': 'No complete Python Observation logged'})
        result.update(error_types=dict(failures), complete_python_observations=full_observations,
                      verified_original_csv_cells=observed_cells, code_generation_prompts=code_prompts)
        audit['runtime'][method] = result
        assert not result['violations'], result['violations']
    (OUT/'final_boundary_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({m: {k:v for k,v in row.items() if k not in {'error_types'}} for m,row in audit['runtime'].items()}, ensure_ascii=False, indent=2))
    print('All checks passed; completeness is reported separately.')


if __name__ == '__main__': main()
