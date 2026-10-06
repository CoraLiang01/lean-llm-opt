"""Create three standalone notebook copies; never modify the source notebooks."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpt41_model_interface_v2_validation'
SOURCES = [
    'LEAN_LLM_OPT_4.1_Large-scale.ipynb',
    'LOTO_Examples_Only_GPT4.1_Large-scale.ipynb',
    'LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb',
]
RUNTIME = (ROOT / 'scripts/gpt41_model_interface_v2_runtime.py').read_text()


def replace_function(source, name, replacement):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno - 1]) + replacement.rstrip() + '\n' + ''.join(lines[node.end_lineno:])


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    base_copy, base_sha = None, None
    for position, original in enumerate(SOURCES):
        source_path = ROOT / original
        source_bytes = source_path.read_bytes()
        book = json.loads(source_bytes)
        destination = original.replace('.ipynb', '_Model_Interface_V2.ipynb')
        target = ROOT / destination
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite notebook copy: {target}')
        variant = ['full', 'examples_only', 'examples_and_route'][position]
        source_cells = [''.join(c.get('source', [])) for c in book['cells']]
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'], cell['execution_count'] = [], None
        config = source_cells[3]
        config = replace_once(config, f'NOTEBOOK_FILENAME = "{original}"', f'NOTEBOOK_FILENAME = "{destination}"')
        config = '\n'.join(
            f'RESULTS_DIR = PROJECT_ROOT / "outputs/gpt41_model_interface_v2/{variant}"' if line.startswith('RESULTS_DIR = ') else
            'CACHE_SCHEMA_VERSION = "rag-101-model-interface-v2"' if line.startswith('CACHE_SCHEMA_VERSION = ') else line
            for line in config.split('\n'))
        if position:
            config = replace_once(config, 'LOTO_BASE_NOTEBOOK = "LEAN_LLM_OPT_4.1_Large-scale.ipynb"', f'LOTO_BASE_NOTEBOOK = "{base_copy}"')
            config = '\n'.join(f'LOTO_BASE_SHA256 = "{base_sha}"' if line.startswith('LOTO_BASE_SHA256 = ') else line for line in config.split('\n'))
        book['cells'][3]['source'] = config.splitlines(keepends=True)

        other = replace_once(source_cells[25], '        code_gen_prompt = PromptTemplate(',
                             '        code_gen_template += "\\n" + MODEL_RETURN_CONTRACT\n\n        code_gen_prompt = PromptTemplate(')
        book['cells'][25]['source'] = other.splitlines(keepends=True)
        generation = replace_once(source_cells[27], '    parts.append(CSV_SOLVER_INSTRUCTIONS)',
                                  '    parts.append(CSV_SOLVER_INSTRUCTIONS)\n    parts.append(MODEL_RETURN_CONTRACT)')
        book['cells'][27]['source'] = generation.splitlines(keepends=True)

        pipeline = replace_function(source_cells[29], 'execute_code', RUNTIME)
        pipeline = replace_once(pipeline, '    query, address = record["query"], record["dataset_address"]',
            '    record.update(model_interface_version=MODEL_INTERFACE_VERSION, selected_route=None,\n'
            '                  executed_route=None, route_allowed=None, execution_ok=False,\n'
            '                  model_contract_recovered=False, model_return_strategy=None,\n'
            '                  failure_category=None, classification_repair_count=0, code_repair_count=0, service_retry_count=0)\n'
            '    query, address = record["query"], record["dataset_address"]')
        pipeline = replace_once(pipeline, '        record["pipeline_stage"] = "formulation"\n        formulation = invoke_formulation(route, query, address)',
            '        record["selected_route"] = record["assigned_route"]\n'
            '        record["pipeline_stage"] = "route_selection"\n'
            '        guard = globals().get("assert_loto_route_allowed")\n'
            '        if callable(guard):\n'
            '            guard(record["selected_route"])\n'
            '        record.update(route_allowed=True, executed_route=record["selected_route"])\n'
            '        record["pipeline_stage"] = "formulation"\n'
            '        formulation = invoke_formulation(route, query, address)')
        pipeline = replace_once(pipeline, '        source = _source_candidate(code)\n        if payload:',
            '        record["raw_generated_code"] = str(code)\n        source = _source_candidate(code)\n        if payload:')
        pipeline = replace_once(pipeline, '        objective, solution = execute_code(record["solve_code"])',
            '        objective, solution, trace = execute_code(record["solve_code"], return_trace=True)\n'
            '        _record_execution_trace(record, trace)')
        pipeline = replace_once(pipeline, '        exc.pipeline_context = record',
            '        trace = getattr(exc, "execution_trace", None)\n'
            '        if trace is not None:\n'
            '            _record_execution_trace(record, trace)\n'
            '        record["failure_category"] = _execution_failure_category(exc)\n'
            '        if type(exc).__name__ == "DisabledRouteError":\n'
            '            record.update(route_allowed=False, executed_route=None)\n'
            '        exc.pipeline_context = record')
        book['cells'][29]['source'] = pipeline.splitlines(keepends=True)

        storage = replace_once(source_cells[33], '"final_ok", "classification_correct", "solution_correct"',
            '"final_ok", "classification_correct", "solution_correct", "route_allowed", "execution_ok", "generated_program_completed", "model_contract_recovered"')
        storage = replace_once(storage, '**dict.fromkeys(("final_objective", "label_objective"), float),',
            '**dict.fromkeys(("final_objective", "label_objective", "solver_runtime"), float),\n'
            '    **dict.fromkeys(("solver_status", "model_count", "classification_repair_count", "code_repair_count", "service_retry_count"), int),')
        storage = replace_once(storage, 'RECORD_ARTIFACT_FILES = {"generated_model": "model.md", "solve_code": "solve.py", "csvqa_observation": "data_overview.md"}',
            'RECORD_ARTIFACT_FILES = {"generated_model": "model.md", "solve_code": "solve.py", "csvqa_observation": "data_overview.md",\n'
            '                         "raw_generated_code": "generated_original.py", "execution_code": "executed.py",\n'
            '                         "execution_details": "execution.json"}')
        book['cells'][33]['source'] = storage.splitlines(keepends=True)
        if position:
            report = replace_once(source_cells[41], '"pipeline_stage", "execution_error_type", "execution_error", "cache_source", "cache_fingerprint",',
                '"pipeline_stage", "execution_error_type", "execution_error", "cache_source", "cache_fingerprint",\n'
                '        "selected_route", "executed_route", "execution_ok", "failure_category",\n'
                '        "model_interface_version", "model_return_strategy", "model_contract_recovered", "solver_status",')
            book['cells'][41]['source'] = report.splitlines(keepends=True)
        for cell in book['cells']:
            if cell['cell_type'] == 'code' and 'experiment' in cell.get('metadata', {}).get('tags', []):
                text = ''.join(cell['source'])
                import re
                text = re.sub(r'(?m)^(RUN_\w+) = True$', r'\1 = False', text)
                cell['source'] = text.splitlines(keepends=True)
        # The LOTO run cell may not have the experiment tag in historical copies.
        if position:
            book['cells'][45]['source'] = ''.join(book['cells'][45]['source']).replace('RUN_LOTO = True', 'RUN_LOTO = False').splitlines(keepends=True)
        introduction = (
            '# GPT-4.1 — Model Interface V2 copy\n\n'
            f'Copied from `{original}`; source SHA-256 `{hashlib.sha256(source_bytes).hexdigest()}`.\n\n'
            'Only the shared model-return contract, execution interface, trace artifacts, and isolated result configuration are changed. '
            'Classification, reference filtering, CSV modes, formulation, variable domains, objective scoring, and retry budgets remain unchanged. '
            'A unique constructed model may be recovered if a function forgets to return it; this is recorded as `captured_constructor`. '
            'No extra model call or optimization is made. Multiple models and missing/closed models fail explicitly. '
            'All automatic run switches are off. Historical outputs were cleared from this copy.\n\n'
        )
        book['cells'][0]['source'] = (introduction + source_cells[0]).splitlines(keepends=True)
        for i, cell in enumerate(book['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'{destination}:cell{i}', 'exec')
        target.write_text(json.dumps(book, ensure_ascii=False, indent=1) + '\n')
        if position == 0:
            base_copy, base_sha = destination, hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({'source': original, 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                         'copy': destination, 'copy_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'changed_source_cells': [i for i, cell in enumerate(book['cells']) if ''.join(cell['source']) != source_cells[i]],
                         'results_directory': f'outputs/gpt41_model_interface_v2/{variant}'})
        assert source_path.read_bytes() == source_bytes
    (OUT / 'copy_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
