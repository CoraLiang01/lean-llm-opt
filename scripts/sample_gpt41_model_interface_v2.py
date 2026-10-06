"""Paired real-solver replay of preselected, self-contained saved programs.

This isolates the executor change. It is NOT fresh classification/formulation or
LLM generation, and it never edits historical programs/results or reference data.
"""
import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from test_gpt41_model_interface_v2 import executor_namespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpt41_model_interface_v2_validation'
AUDIT = ROOT / 'outputs/loto_cross_model_audit_20261004/gpt41_audit'


def old_namespace(path):
    ns = executor_namespace(ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_Model_Interface_V2.ipynb')
    book = json.loads(path.read_text())
    tree = ast.parse(''.join(book['cells'][29]['source']))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_source_candidate', 'execute_code')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def worker(spec, version, repeat):
    spec = json.loads(Path(spec).read_text())
    code_path = Path(spec['code_path'])
    code = code_path.read_text()
    if hashlib.sha256(code.encode()).hexdigest() != spec['code_sha256']:
        raise ValueError('Saved sample changed after selection')
    notebook_name = spec['notebook'] if version == 'old' else spec['copy']
    expected = spec['notebook_sha256'] if version == 'old' else spec['copy_sha256']
    assert hashlib.sha256((ROOT / notebook_name).read_bytes()).hexdigest() == expected
    ns = old_namespace(ROOT / spec['notebook']) if version == 'old' else executor_namespace(ROOT / spec['copy'])
    result = {**spec, 'executor': version, 'repeat': repeat, 'solved': False,
              'objective_match': False, 'objective': None, 'error': None}
    try:
        if version == 'old':
            objective, solution = ns['execute_code'](code)
            trace = None
        else:
            objective, solution, trace = ns['execute_code'](code, return_trace=True)
        result.update(solved=True, objective=float(objective), variable_count=len(solution),
                      objective_match=math.isclose(float(objective), spec['reference_objective'], rel_tol=1e-4, abs_tol=1e-4))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        trace = getattr(exc, 'execution_trace', None)
    if trace is not None:
        trace = dict(trace)
        execution_code = trace.pop('executed_source', None)
        if execution_code:
            (OUT / 'replay' / f'{spec["sample_id"]}_{version}_{repeat}_executed.py').write_text(execution_code)
        result['execution_trace'] = trace
    (OUT / 'replay' / f'{spec["sample_id"]}_{version}_{repeat}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    (OUT / 'replay').mkdir(parents=True, exist_ok=True)
    samples = []
    specs = [
        ('examples_only', ['OR-004', 'OR-018', 'OR-036', 'OR-063', 'OR-002', 'OR-069', 'OR-001', 'OR-043']),
        ('examples_and_route', ['OR-011', 'OR-018']),
        ('full', ['OR-004', 'OR-036']),
    ]
    names = {
        'full': 'LEAN_LLM_OPT_4.1_Large-scale.ipynb',
        'examples_only': 'LOTO_Examples_Only_GPT4.1_Large-scale.ipynb',
        'examples_and_route': 'LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb',
    }
    references = {row['problem_id']: row for row in json.loads((AUDIT / 'examples_only_cases.json').read_text())}
    for variant, ids in specs:
        cases = {row['problem_id']: row for row in json.loads((AUDIT / f'{variant}_cases.json').read_text())} if variant != 'full' else references
        for case_id in ids:
            record = cases[case_id]
            if variant == 'full':
                matches = list((ROOT / 'outputs/final_101_V2/automatic/results_cases/AUTO' / case_id).glob('*/solve.py'))
                assert len(matches) == 1
                path = matches[0]
            else:
                path = Path(record['solution_path']).with_name('solve.py')
            code = path.read_text()
            # Reject non-self-contained or unexpected IO programs before any execution.
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                    assert all(m in {'gurobipy', 'math', 're', 'json', 'numpy', 'pandas'} for m in modules), modules
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        assert node.func.id not in {'exec', 'eval', 'compile', '__import__', 'open', 'input', 'getattr', 'setattr'}
                    if isinstance(node.func, ast.Attribute):
                        assert node.func.attr not in {'read_csv', 'read_excel', 'read_text', 'read_bytes', 'write', 'to_csv', 'system', 'popen', 'remove', 'unlink', 'dispose', 'close', 'computeIIS'}
            sample = {'sample_id': f'{variant}_{case_id}', 'variant': variant, 'problem_id': case_id,
                      'notebook': names[variant], 'copy': names[variant].replace('.ipynb', '_Model_Interface_V2.ipynb'),
                      'code_path': str(path), 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                      'reference_objective': references[case_id]['reference_objective']}
            sample['notebook_sha256'] = hashlib.sha256((ROOT / sample['notebook']).read_bytes()).hexdigest()
            sample['copy_sha256'] = hashlib.sha256((ROOT / sample['copy']).read_bytes()).hexdigest()
            samples.append(sample)
    manifest = {'scope': 'Frozen-program real Gurobi replay, not end-to-end inference',
                'selection': 'Four missing-return failures, two correct controls, one solved-wrong control, one uncalled-function negative control; two E2 and two full-baseline controls. Selected before execution.',
                'repeats_per_executor': 2, 'samples': samples}
    (OUT / 'sample_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    results = []
    for sample in samples:
        spec_path = OUT / 'replay' / f'{sample["sample_id"]}_spec.json'
        spec_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
        for version in ['old', 'new']:
            for repeat in [1, 2]:
                dest = OUT / 'replay' / f'{sample["sample_id"]}_{version}_{repeat}'
                with dest.with_suffix('.log').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(spec_path),
                                    '--version', version, '--repeat', str(repeat)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
                result = json.loads(dest.with_suffix('.json').read_text())
                results.append(result)
        print(f'Completed paired replay: {sample["sample_id"]}', flush=True)
    (OUT / 'replay_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    for version in ['old', 'new']:
        subset = [r for r in results if r['executor'] == version and r['repeat'] == 1]
        print(version, 'cases', len(subset), 'solved', sum(r['solved'] for r in subset), 'matched', sum(r['objective_match'] for r in subset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--version')
    parser.add_argument('--repeat', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.version, args.repeat)
    else:
        main()
