"""Verify the confirmed aggregate gate with isolated synthetic score fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import check_react_launch_gate as gate


class LaunchPolicy(unittest.TestCase):
    def fixture(self, root, version, counts, complete=True):
        folder = root / f'full_{version}'
        folder.mkdir()
        datasets = ['automatic', 'variants'] + [f'columns/{p}-S{s}'
                    for p in ('50pct', '100pct', '200pct') for s in (1, 2, 3)]
        denominators = [101, 36] + [35] * 9
        pd.DataFrame([{'dataset': d, 'expected': n, 'recorded': n, 'objective_match': m}
                      for d, n, m in zip(datasets, denominators, counts)]).to_csv(folder / 'summary.csv', index=False)
        pd.DataFrame([{'true_label': label, 'total': n, 'objective_match': m}
                      for label, n, m in [('AP', 5, 5), ('FLP', 14, 13), ('NRM', 25, 25),
                                          ('RA', 22, 22), ('TP', 9, 9), ('Others', 8, 5), ('Mixture', 18, 15)]
                      ]).to_csv(folder / 'accuracy_by_class.csv', index=False)
        (folder / 'run_status.json').write_text(json.dumps({'complete': complete}))
        (folder / 'manifest.json').write_text(json.dumps({'inputs': {'synthetic.csv': 'unchanged'}}))
        (folder / 'frozen_notebook.ipynb').write_text('{}')

    def evaluate(self, counts, complete=True, category_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'overall_launch_policy.json').write_text(json.dumps(
                {'metric': 'three_group_macro', 'preserve_original_required_floors': True}))
            self.fixture(root, 'v7', [94, 34, 33, 33, 33, 32, 30, 32, 32, 33, 31])
            self.fixture(root, 'candidate', counts, complete)
            if category_failure:
                path = root / 'full_candidate/accuracy_by_class.csv'
                data = pd.read_csv(path)
                data.loc[data.true_label.eq('AP'), 'objective_match'] = 4
                data.to_csv(path, index=False)
            old = gate.OUT
            try:
                gate.OUT = root
                with contextlib.redirect_stdout(io.StringIO()):
                    return gate.check('candidate')
            finally:
                gate.OUT = old

    def test_overall_improvement_allows_individual_dataset_decline(self):
        result = self.evaluate([98, 33] + [35] * 9)
        self.assertTrue(result['passed'])
        self.assertFalse(next(r for r in result['datasets'] if r['dataset'] == 'variants')['no_decline'])
        self.assertEqual(result['overall']['denominators'], [101, 36, 315])

    def test_necessary_variant_floor_blocks_even_when_macro_improves(self):
        result = self.evaluate([101, 31] + [35] * 9)
        self.assertTrue(result['overall_improved'])
        self.assertFalse(result['passed'])

    def test_necessary_category_floor_blocks_launch(self):
        self.assertFalse(self.evaluate([98, 35] + [35] * 9, category_failure=True)['passed'])

    def test_incomplete_pass_blocks_launch(self):
        self.assertFalse(self.evaluate([98, 35] + [35] * 9, complete=False)['passed'])


if __name__ == '__main__':
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(LaunchPolicy))
    (gate.OUT / 'launch_policy_regression_checks.txt').write_text(stream.getvalue())
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
