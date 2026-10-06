"""Real CSV execution and strict schema tests; only the external LLM is scripted."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'LEAN_LLM_OPT_4.1_Large-scale_Extraction_V6.ipynb'


def namespace():
    os.environ['LEAN_LLM_OPT_ROOT'] = str(ROOT)
    ns = {'__name__': 'extraction_tests'}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, cell in enumerate(json.loads(BOOK.read_text())['cells']):
            if cell['cell_type'] == 'code' and i < 37:
                exec(compile(''.join(cell['source']), f'{BOOK}:cell{i}', 'exec'), ns)
    return ns


class StructuredClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.options = None

    def with_structured_output(self, schema, **kwargs):
        self.schema, self.options = schema, kwargs
        return self

    def invoke(self, messages):
        self.requests.append(messages[0].content)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str):
            return {'raw': type('Message', (), {'content': value})(),
                    'parsed': None, 'parsing_error': ValueError('invalid plan')}
        parsed = self.schema.model_validate(value)
        return {'raw': type('Message', (), {'content': json.dumps(value)})(),
                'parsed': parsed, 'parsing_error': None}


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'items.csv'
        self.path.write_text('ID,Family,Profit,Noise\n001,F1,8,90\n002,F2,3,80\n')
        self.tables = self.ns['_load_tables'](str(self.path))
        self.query = 'Maximize Profit for all items.'
        self.plan = {'route': 'RA', 'tables': [{'file_index': 0, 'role': 'items',
            'columns': ['ID', 'Family', 'Profit'], 'filters': {'logic': 'and', 'conditions': []}}],
            'ignored_file_indices': [], 'relationships': [],
            'bindings': [{'parameter': 'profit', 'table_id': 'file_0_view_0',
                          'value_column': 'Profit', 'index_columns': ['ID']}]}

    def execute(self):
        return self.ns['_execute_plan'](self.plan, self.tables, self.query, 'RA')

    def invoke(self, responses, repair=False):
        client = StructuredClient(responses)
        with patch.dict(self.ns, make_llm=lambda: client, CSVQA_REPAIR_PLAN_ON_FAILURE=repair):
            _, tool, state = self.ns['build_csvqa_components'](str(self.path), 'Select needed data',
                'Get data', route='RA', user_query=self.query)
            data = json.loads(tool.invoke('Get data'))
        return data, state['trace'], client

    def test_planner_uses_strict_schema_and_preserves_raw_response(self):
        data, trace, client = self.invoke([self.plan])
        self.assertEqual(client.options, {'method': 'json_schema', 'strict': True, 'include_raw': True})
        self.assertEqual(trace['planner_outputs'], [json.dumps(self.plan)])
        self.assertEqual(data['bindings'], self.plan['bindings'])
        self.assertEqual(data['tables'][0]['records'][0]['values']['ID'], '001')

    def test_schema_rejects_unknown_fields_and_invalid_operator(self):
        schema = self.ns.get('CSVExtractionPlan')
        self.assertIsNotNone(schema, 'A strict extraction schema is required')
        with self.assertRaises(ValueError):
            schema.model_validate({**self.plan, 'python': 'frame.head(1)'})
        self.plan['tables'][0]['filters']['conditions'] = [{'column': 'ID', 'operator': 'python',
            'dtype': 'string', 'value': '001', 'evidence': '001', 'format': None, 'inclusive': 'both'}]
        with self.assertRaises(ValueError):
            schema.model_validate(self.plan)

    def test_schema_requires_all_fields_and_forbids_extras_recursively(self):
        schema = self.ns.get('CSVExtractionPlan')
        self.assertIsNotNone(schema, 'A strict extraction schema is required')
        def walk(value):
            if isinstance(value, dict):
                if value.get('type') == 'object':
                    self.assertFalse(value['additionalProperties'])
                    self.assertEqual(set(value['required']), set(value['properties']))
                for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(schema.model_json_schema())

    def test_invalid_structured_response_falls_back_without_brace_salvage(self):
        data, trace, _ = self.invoke(['prose ' + json.dumps(self.plan)])
        self.assertEqual(trace['status'], 'FALLBACK_FULL_DATA')
        self.assertEqual(data['tables'][0]['columns'], ['ID', 'Family', 'Profit', 'Noise'])
        self.assertEqual(data['tables'][0]['returned_rows'], 2)

    def test_optional_repair_reuses_structured_schema_after_unknown_column(self):
        bad = json.loads(json.dumps(self.plan)); bad['tables'][0]['columns'] = ['ID', 'Missing']
        _, trace, client = self.invoke([bad, self.plan], repair=True)
        self.assertEqual(trace['status'], 'PLANNED_REPAIRED')
        self.assertEqual(len(client.requests), 2)
        self.assertIn('Unknown columns', client.requests[1])

    def test_binding_rejects_unknown_value_column(self):
        self.plan['bindings'][0]['value_column'] = 'Noise'
        with self.assertRaisesRegex(ValueError, 'Binding.*column'):
            self.execute()

    def test_duplicate_binding_keys_cannot_silently_overwrite(self):
        self.tables[0]['frame'].loc[1, 'ID'] = '001'
        with self.assertRaisesRegex(ValueError, 'duplicate.*key'):
            self.execute()

    def test_composite_binding_keys_keep_repeated_option_labels(self):
        self.tables[0]['frame'].loc[:, 'ID'] = 'O1'
        self.plan['bindings'][0]['index_columns'] = ['Family', 'ID']
        data = self.execute()
        self.assertEqual(data.get('validation', {}).get('binding_checks', [])[0]['key_count'], 2)

    def test_empty_binding_id_is_rejected(self):
        self.tables[0]['frame'].loc[1, 'ID'] = ''
        with self.assertRaisesRegex(ValueError, 'empty.*key'):
            self.execute()

    def test_scalar_binding_cannot_choose_arbitrary_row(self):
        self.plan['bindings'][0]['index_columns'] = []
        with self.assertRaisesRegex(ValueError, 'scalar.*one row'):
            self.execute()

    def test_string_only_operator_on_numeric_data_is_validation_error(self):
        with self.assertRaisesRegex(ValueError, 'string'):
            self.ns['_apply_condition'](self.tables[0]['frame']['Profit'],
                {'operator': 'prefix', 'dtype': 'number', 'value': 8})

    def test_scalar_operator_rejects_list_instead_of_rowwise_comparison(self):
        with self.assertRaisesRegex(ValueError, 'scalar'):
            self.ns['_apply_condition'](self.tables[0]['frame']['ID'],
                {'operator': 'eq', 'dtype': 'string', 'value': ['001', '002']})

    def test_filter_evidence_must_be_literal_query_span(self):
        self.query = 'Only product A-1 is eligible.'
        self.tables[0]['frame'].loc[0, 'ID'] = 'A-1'
        self.plan['tables'][0]['filters']['conditions'] = [{'column': 'ID', 'operator': 'exact',
            'dtype': 'string', 'value': 'A-1', 'evidence': 'Only product A1', 'format': None, 'inclusive': 'both'}]
        with self.assertRaisesRegex(ValueError, 'query evidence'):
            self.execute()

    def test_valid_subset_returns_complete_source_order_with_leading_zero_ids(self):
        self.query = 'Only ID "002" is eligible.'
        self.plan['tables'][0]['filters']['conditions'] = [{'column': 'ID', 'operator': 'in',
            'dtype': 'string', 'value': ['002'], 'evidence': 'Only ID "002"', 'format': None, 'inclusive': 'both'}]
        data = self.execute()
        self.assertEqual(data['tables'][0]['records'], [{'source_row': 1,
            'values': {'ID': '002', 'Family': 'F2', 'Profit': '3'}}])

    def test_matrix_alignment_does_not_erase_business_id_punctuation(self):
        import pandas as pd
        self.tables = [
            {'file_index': 0, 'path': Path('rows.csv'), 'frame': pd.DataFrame({'ID': ['A-1']})},
            {'file_index': 1, 'path': Path('columns.csv'), 'frame': pd.DataFrame({'ID': ['B-1']})},
            {'file_index': 2, 'path': Path('matrix.csv'), 'frame': pd.DataFrame({'ID': ['A1'], 'B1': ['5']})}]
        self.plan['tables'] = [{'file_index': i, 'role': 'axis', 'columns': '*',
            'filters': {'logic': 'and', 'conditions': []}} for i in range(3)]
        self.plan['bindings'] = []
        self.plan['relationships'] = [{'type': 'matrix', 'matrix_table_id': 'file_2_view_0',
            'row_id_column': 'ID', 'row_axis': {'table_id': 'file_0_view_0', 'id_column': 'ID'},
            'column_axis': {'table_id': 'file_1_view_0', 'id_column': 'ID'}}]
        with self.assertRaisesRegex(ValueError, 'Matrix validation failed'):
            self.execute()


if __name__ == '__main__':
    unittest.main()
