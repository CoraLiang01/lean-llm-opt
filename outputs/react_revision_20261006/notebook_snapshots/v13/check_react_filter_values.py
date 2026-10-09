"""Test the declarative filter-value adapter without inference or optimization."""
import argparse
import contextlib
import io
import unittest

import pandas as pd

import evaluate_react_revision_20261006 as runner


class FilterValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ns = runner.namespace(runner.ROOT / runner.NAMES['full'], 'full')
        cls.apply = staticmethod(ns['_apply_condition'])

    def test_singleton_prefix_list_preserves_exact_requested_prefix(self):
        series = pd.Series(['AX.1_one', 'AX.1_two', 'AXx1_three', 'Other'])
        condition = {'operator': 'prefix', 'dtype': 'string', 'value': ['AX.1']}
        self.assertEqual(self.apply(series, condition).tolist(), [True, True, False, False])

    def test_singleton_equality_list_is_one_scalar_not_rowwise_comparison(self):
        series = pd.Series(['ALPHA', 'BETA', 'ALPHA'])
        self.assertEqual(self.apply(series, {'operator': 'eq', 'value': ['ALPHA']}).tolist(),
                         [True, False, True])

    def test_singleton_numeric_and_date_comparison(self):
        self.assertEqual(self.apply(pd.Series(['1', '3', '5']),
            {'operator': 'le', 'dtype': 'number', 'value': [3]}).tolist(), [True, True, False])
        self.assertEqual(self.apply(pd.Series(['2025-01-01', '2025-06-01']),
            {'operator': 'le', 'dtype': 'date', 'value': ['2025-03-01']}).tolist(), [True, False])

    def test_set_and_between_values_keep_their_list_semantics(self):
        self.assertEqual(self.apply(pd.Series(['A', 'B', 'C']),
            {'operator': 'in', 'value': ['A', 'C']}).tolist(), [True, False, True])
        self.assertEqual(self.apply(pd.Series(['1', '3', '5']),
            {'operator': 'between', 'dtype': 'number', 'value': [2, 4]}).tolist(), [False, True, False])

    def test_ambiguous_multi_value_scalar_operator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'requires one scalar value'):
            self.apply(pd.Series(['ALPHA', 'BETA']), {'operator': 'prefix', 'value': ['ALPHA', 'BETA']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(FilterValues))
    (runner.OUT / f'filter_value_regressions_{args.label}.txt').write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
