import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path('/Users/cora/Documents/GitHub/lean-llm-opt')
BOOK = Path(__file__).parent / 'LEAN_LLM_OPT_4.1_Large-scale_Extraction_V6.ipynb'


def namespace():
    os.environ['LEAN_LLM_OPT_ROOT'] = str(ROOT)
    ns = {'__name__': 'notebook_checks'}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, cell in enumerate(json.loads(BOOK.read_text())['cells']):
            if cell['cell_type'] == 'code' and i < 37:
                exec(compile(''.join(cell['source']), f'cell{i}', 'exec'), ns)
    ns['NOTEBOOK_PATH'] = BOOK
    return ns


class AdditionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.ns = namespace()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_variant_loader_preserves_folder_ids_and_original_labels(self):
        import pandas as pd
        source = pd.DataFrame({'Query': ['q2', 'q1'],
            'dataset_address': ['/tmp/Variant29/inputs/one.csv', '/tmp/Variant6/inputs/two.csv'],
            'problem_id': ['OR-001', 'OR-002'], 'true_label': ['Mixture', 'Others'],
            'true_route': ['Others', 'Others'], 'label_objective': [10, 20]})
        load = self.ns.get('load_variants_benchmark')
        self.assertIsNotNone(load, 'Notebook must provide a variants loader')
        with patch.dict(self.ns, load_benchmark=lambda *args, **kwargs: source.copy()):
            result = load(ROOT / 'benchmark_dataset 3')
        self.assertEqual(result.problem_id.tolist(), ['Variant29', 'Variant6'])
        self.assertEqual(result.true_label.tolist(), ['Mixture', 'Others'])

    def test_variant_loader_rejects_duplicate_ids(self):
        import pandas as pd
        source = pd.DataFrame({'dataset_address': ['/tmp/Variant6/a.csv', '/tmp/Variant6/b.csv']})
        load = self.ns.get('load_variants_benchmark')
        self.assertIsNotNone(load)
        with patch.dict(self.ns, load_benchmark=lambda *args, **kwargs: source.copy()):
            with self.assertRaisesRegex(ValueError, 'Duplicate variant'):
                load(ROOT / 'benchmark_dataset 3')

    def test_sheet_loader_selects_only_requested_sheets_in_requested_order(self):
        import pandas as pd
        p = self.directory / 'columns.xlsx'
        with pd.ExcelWriter(p) as writer:
            for sheet in ['S1', 'S2', 'S3']:
                pd.DataFrame({'Query': [sheet], 'Label': [10]}).to_excel(writer, sheet_name=sheet, index=False)
        load = self.ns.get('load_column_sheets')
        self.assertIsNotNone(load, 'Notebook must provide a sheet selector')
        result = load(p, ['S3', 'S1'])
        self.assertEqual(result.benchmark_sheet.tolist(), ['S3', 'S1'])
        self.assertEqual(result.problem_id.tolist(), ['S3/OR-001', 'S1/OR-001'])
        with self.assertRaisesRegex(ValueError, 'Unknown sheet'):
            load(p, ['S4'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            load(p, ['S1', 'S1'])

    def test_resume_can_preserve_failed_attempt_without_api_or_artifact_rewrite(self):
        path = self.directory / 'results.csv'
        case = {'problem_id': 'one', 'Query': 'q', 'dataset_address': '', 'label_objective': 1}
        ns = self.ns
        with patch.dict(ns, _source_fingerprint=lambda: 'frozen', case_fingerprint=lambda *args, **kw: 'same'):
            error = ns['error_record'](case, ValueError('original failure'))
            error['cache_fingerprint'] = 'same'
            ns['save_record'](path, ns['finalize_record'](error))
            before = path.read_bytes()
            with patch.dict(ns, require_api_key=lambda: self.fail('Must not require API for a saved failure'),
                            execute_pipeline_case=lambda *args, **kw: self.fail('Failed attempt was rerun')):
                rows = ns['run_test'](__import__('pandas').DataFrame([case]), output_csv=path,
                                      reuse_completed=True, retry_failed=False)
            self.assertFalse(rows[0]['final_ok'])
            self.assertEqual(rows[0]['cache_source'], 'csv_failed_attempt')
            self.assertEqual(path.read_bytes(), before)

    def test_failure_in_formulation_retains_csvqa_state(self):
        ns = self.ns
        state = {'observation': '{"tables":[]}', 'trace': {'status': 'PLANNED', 'planner_attempt_count': 1}}
        with patch.dict(ns, build_csvqa_components=lambda *args, **kw: ('llm', 'tool', state),
                        initialize_agent=lambda **kw: 'agent',
                        invoke_react_with_required_csvqa=lambda *args: (_ for _ in ()).throw(ValueError('formulation failed'))):
            try: ns['formulate_with_csvqa']('q', 'path', 'RA', 'system', 'tool', 'prefix', 'suffix')
            except ValueError as exc:
                self.assertEqual(getattr(exc, 'csvqa_state', None), state)
            else: self.fail('Expected formulation failure')

    def test_pipeline_failure_context_includes_existing_csvqa_trace(self):
        ns = self.ns
        exc = ValueError('formulation failed')
        exc.csvqa_state = {'observation': '{"tables":[]}',
                           'trace': {'status': 'PLANNED', 'planner_attempt_count': 1}}
        case = {'problem_id': 'one', 'Query': 'q', 'dataset_address': '/tmp/source.csv'}
        with patch.dict(ns, invoke_formulation=lambda *args: (_ for _ in ()).throw(exc)):
            try: ns['execute_pipeline_case'](case, forced_route='RA')
            except ValueError as error:
                context = error.pipeline_context
                self.assertEqual(context.get('csvqa_observation'), '{"tables":[]}')
                self.assertEqual(context.get('csvqa_status'), 'PLANNED')
                self.assertEqual(json.loads(context.get('csvqa_trace', '{}'))['planner_attempt_count'], 1)
            else: self.fail('Expected pipeline failure')


if __name__ == '__main__': unittest.main()
