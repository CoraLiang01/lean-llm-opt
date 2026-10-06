"""Pipeline/guard/artifact integration checks with stub LLM output and real Gurobi."""
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/gpt41_model_interface_v2_validation'
CODE = '''import gurobipy as gp
def solve():
    local = gp.Model()
    local.Params.OutputFlag = 0
    x = local.addVar(ub=5)
    local.setObjective(x, gp.GRB.MAXIMIZE)
    local.optimize()
m = solve()
'''


def main():
    records = []
    for path in sorted(ROOT.glob('*Large-scale_Model_Interface_V2.ipynb')):
        book = json.loads(path.read_text())
        ns = {'__name__': '__integration_test__'}
        loto = path.name.startswith('LOTO_')
        with contextlib.redirect_stdout(io.StringIO()):
            for i, cell in enumerate(book['cells']):
                if cell['cell_type'] != 'code' or i > (41 if loto else 35):
                    continue
                exec(compile(''.join(cell['source']), f'{path}:cell{i}', 'exec'), ns)
        if loto:
            remove_route = ns['LOTO_REMOVE_ROUTE']
            ns['LOTO_STATE'] = {
                'held_out_type': 'TP', 'disabled_route': 'TP' if remove_route else None,
                'allowed_labels': [label for label in ns['CLASS_LABELS'] if label != 'TP'] if remove_route else ns['CLASS_LABELS'],
                'reference_sha256': 'fixture-reference', 'reference_count_after': 14,
            }
        visits = []
        def formulation(*args):
            visits.append(args)
            return {'formulation': 'Maximize x, 0 <= x <= 5', 'code': CODE,
                    'observation': 'fixture', 'trace': {'status': 'FIXTURE'}}
        ns['_LOTO_BASE_INVOKE_FORMULATION' if loto else 'invoke_formulation'] = formulation
        ns['invoke_classifier'] = lambda query: {'normalized_label': 'AP'}
        case = {'problem_id': 'fixture-001', 'Query': 'Fixture only, not benchmark data',
                'dataset_address': 'fixture.csv', 'true_label': 'TP', 'true_route': 'TP',
                'label_objective': 5.0}
        with contextlib.redirect_stdout(io.StringIO()):
            record = ns['finalize_record'](ns['execute_pipeline_case'](case))
        assert record['final_ok'] and record['solution_correct']
        assert record['selected_route'] == record['executed_route'] == 'AP'
        assert record['route_allowed'] and record['model_contract_recovered']
        assert record['model_return_strategy'] == 'captured_constructor'
        assert record['classification_repair_count'] == record['code_repair_count'] == record['service_retry_count'] == 0
        assert all(field in record for field in ('raw_generated_code', 'execution_code', 'execution_details'))
        state = {'notebook': path.name, 'definition_load': 'PASS', 'real_solver_pipeline': 'PASS',
                 'route_guard': 'not applicable', 'saved_artifacts_and_cache': None}
        try:
            with tempfile.TemporaryDirectory(prefix='lean-interface-v2-') as directory:
                csv_path = Path(directory) / 'results.csv'
                record['cache_fingerprint'] = 'fixture-v2'
                ns['save_record'](csv_path, record)
                row = ns['load_records'](csv_path)[0]
                restored = ns['reusable_record'](csv_path, row, 'fixture-v2')
                assert restored is not None
                for field in ('raw_generated_code', 'execution_code', 'execution_details'):
                    assert restored[field] == record[field]
                assert restored['route_allowed'] is True
                assert restored['execution_ok'] is True
                assert restored['model_contract_recovered'] is True
                assert ns['reusable_record'](csv_path, row, 'old-fingerprint') is None
                # A tampered executed artifact must invalidate reuse.
                (csv_path.parent / row['execution_code_path']).write_text('tampered')
                assert ns['reusable_record'](csv_path, row, 'fixture-v2') is None
                state['saved_artifacts_and_cache'] = 'PASS'
        except PermissionError as exc:
            state['saved_artifacts_and_cache'] = f'BLOCKED by filesystem protection: {exc}'
        if loto and ns['LOTO_REMOVE_ROUTE']:
            ns['invoke_classifier'] = lambda query: {'normalized_label': 'TP'}
            before = len(visits)
            try:
                ns['execute_pipeline_case'](case)
                raise AssertionError('Forbidden route was not rejected')
            except ns['DisabledRouteError'] as exc:
                failure = ns['error_record'](case, exc)
                assert failure['selected_route'] == 'TP'
                assert failure['executed_route'] is None and failure['route_allowed'] is False
                assert failure['failure_category'] == 'forbidden_route'
                assert len(visits) == before
            state['route_guard'] = 'PASS'
        # Missing function call stays a failure and records an actionable category.
        ns['invoke_classifier'] = lambda query: {'normalized_label': 'AP'}
        def no_model(*args):
            return {'formulation': 'fixture', 'code': 'def solve():\n    pass', 'observation': '', 'trace': {}}
        ns['_LOTO_BASE_INVOKE_FORMULATION' if loto else 'invoke_formulation'] = no_model
        try:
            ns['execute_pipeline_case'](case)
            raise AssertionError('No-model program unexpectedly succeeded')
        except ns['ModelContractError'] as exc:
            failed = ns['error_record'](case, exc)
            assert not failed['final_ok'] and failed['failure_category'] == 'model_interface'
            assert failed['executed_route'] == 'AP'
            assert json.loads(failed['execution_details'])['model_count'] == 0
        state['failure_record'] = 'PASS'
        records.append(state)
    (OUT / 'pipeline_record_tests.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
