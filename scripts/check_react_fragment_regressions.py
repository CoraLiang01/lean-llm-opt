"""Exercise query-required CSV fragment coverage without external API calls."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import evaluate_react_revision_20261006 as runner


def namespace():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return runner.namespace(runner.ROOT / runner.NAMES['full'], 'full')


class FragmentCoverage(unittest.TestCase):
    query = 'Use tenant ALPHA records effective on or before 2025-06-30.'
    columns = ['table', 'tenant', 'record_id', 'revision', 'effective_date', 'value']

    def tables(self, tenant='ALPHA', date='2025-06-15', kind='cost', reorder=False):
        frames = [pd.DataFrame([['cost', 'ALPHA', 'a', '1', '2025-06-01', '7']], columns=self.columns),
                  pd.DataFrame([[kind, tenant, 'b', '2', date, '11']], columns=self.columns)]
        if reorder:
            frames[1] = frames[1][list(reversed(self.columns))]
        return [{'file_index': i, 'path': Path(f'part_{i}.csv'), 'frame': frame}
                for i, frame in enumerate(frames)]

    def plan(self, logic='and'):
        return {'route': 'RA', 'tables': [{'file_index': 0, 'columns': '*', 'filters': {
            'logic': logic, 'conditions': [
                {'column': 'tenant', 'operator': 'exact', 'dtype': 'string',
                 'value': 'ALPHA', 'evidence': 'tenant ALPHA'},
                {'column': 'effective_date', 'operator': 'le', 'dtype': 'date',
                 'value': '2025-06-30', 'evidence': 'effective on or before 2025-06-30'}]}}],
                'ignored_file_indices': [1], 'relationships': []}

    def test_matching_fragment_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'Ignored CSV fragment'):
            namespace()['_execute_plan'](self.plan(), self.tables(), self.query, 'RA')

    def test_column_order_does_not_hide_a_matching_fragment(self):
        with self.assertRaisesRegex(ValueError, 'Ignored CSV fragment'):
            namespace()['_execute_plan'](self.plan(), self.tables(reorder=True), self.query, 'RA')

    def test_explicit_filter_can_exclude_other_tenant_or_future_rows(self):
        ns = namespace()
        for tables in (self.tables(tenant='BETA'), self.tables(date='2025-07-01')):
            with self.subTest(rows=tables[1]['frame'].to_dict('records')):
                payload = ns['_execute_plan'](self.plan(), tables, self.query, 'RA')
                self.assertEqual(payload['validation']['status'], 'OK')
                self.assertEqual(payload['ignored_file_indices'], [1])

    def test_different_logical_table_with_same_headers_can_be_ignored(self):
        payload = namespace()['_execute_plan'](self.plan(), self.tables(kind='audit'), self.query, 'RA')
        self.assertEqual(payload['validation']['status'], 'OK')

    def test_or_filter_retains_matching_fragment(self):
        with self.assertRaisesRegex(ValueError, 'Ignored CSV fragment'):
            namespace()['_execute_plan'](self.plan(logic='or'), self.tables(tenant='BETA'), self.query, 'RA')

    def test_existing_fallback_preserves_exact_source_values_without_retry(self):
        ns = namespace()
        ns['make_llm'] = lambda **kwargs: FakeListChatModel(responses=[json.dumps(self.plan())])
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for table in self.tables():
                path = Path(directory) / table['path']
                table['frame'].to_csv(path, index=False)
                files.append(str(path))
            _, tool, state = ns['build_csvqa_components'](
                '\n'.join(files), '', '', route='RA', user_query=self.query)
            observation = json.loads(tool.invoke(self.query))
        self.assertEqual(state['trace']['status'], 'FALLBACK_FULL_DATA')
        self.assertEqual(state['trace']['planner_attempt_count'], 1)
        self.assertEqual(state['trace']['repair_count'], 0)
        self.assertEqual(state['trace']['retry_count'], 0)
        self.assertEqual(state['trace']['fallback_count'], 1)
        self.assertEqual(observation['tables'][1]['records'][0]['values']['value'], '11')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='v11')
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FragmentCoverage))
    (runner.OUT / f'fragment_regression_checks_{args.label}.txt').write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
