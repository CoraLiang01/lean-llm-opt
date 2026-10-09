"""Complete-pass gate using the user's confirmed overall comparison policy."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from evaluate_react_revision_20261006 import OUT


def check(version, baseline="v7"):
    root, old = OUT / f"full_{version}", OUT / f"full_{baseline}"
    report = OUT / f"report_{version}"
    report.mkdir(exist_ok=True)
    state = json.loads((root / 'run_status.json').read_text()) if (root / 'run_status.json').exists() else {}
    passed = state.get('complete') is True
    rows = []
    if (root / 'summary.csv').exists():
        current = pd.read_csv(root / 'summary.csv').set_index('dataset')
        previous = pd.read_csv(old / 'summary.csv').set_index('dataset')
        for group, reference in previous.iterrows():
            value = current.loc[group] if group in current.index else {}
            complete = int(value.get('recorded', 0)) == int(reference['expected'])
            match = int(value.get('objective_match', 0))
            no_decline = match >= int(reference['objective_match'])
            rows.append({'dataset': group, 'expected': int(reference['expected']),
                         'recorded': int(value.get('recorded', 0)), 'baseline_match': int(reference['objective_match']),
                         'current_match': match, 'delta': match - int(reference['objective_match']),
                         'complete': complete, 'no_decline': no_decline})
            passed = passed and complete
    else:
        passed = False
    classes = []
    if (root / 'accuracy_by_class.csv').exists():
        current = pd.read_csv(root / 'accuracy_by_class.csv').set_index('true_label')
        previous = pd.read_csv(old / 'accuracy_by_class.csv').set_index('true_label')
        for label, reference in previous.iterrows():
            match = int(current.loc[label, 'objective_match']) if label in current.index else 0
            no_decline = match >= int(reference['objective_match'])
            classes.append({'class': label, 'baseline_match': int(reference['objective_match']),
                            'current_match': match, 'no_decline': no_decline})
    else:
        passed = False
    same_inputs = False
    if (root / 'manifest.json').exists():
        same_inputs = json.loads((root / 'manifest.json').read_text())['inputs'] == json.loads((old / 'manifest.json').read_text())['inputs']
    policy_path = OUT / 'overall_launch_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    def totals(folder):
        path = folder / 'summary.csv'
        if not path.exists():
            return None
        data = pd.read_csv(path).set_index('dataset')
        if 'automatic' not in data.index or 'variants' not in data.index:
            return None
        columns = data.loc[data.index.str.startswith('columns/')]
        if len(columns) != 9:
            return None
        matches = [int(data.loc['automatic', 'objective_match']),
                   int(data.loc['variants', 'objective_match']), int(columns['objective_match'].sum())]
        denominators = [101, 36, 315]
        return {'matches': matches, 'denominators': denominators,
                'macro_accuracy': sum(m / n for m, n in zip(matches, denominators)) / 3,
                'pooled_matches': sum(matches), 'pooled_accuracy': sum(matches) / 452}
    overall, reference_overall = totals(root), totals(old)
    overall_improved = bool(overall and reference_overall and
                            overall['macro_accuracy'] > reference_overall['macro_accuracy'])
    required_floors = {}
    if overall:
        required_floors = {'101_at_least_93': overall['matches'][0] >= 93,
                           'variants_at_least_32': overall['matches'][1] >= 32}
        if (root / 'accuracy_by_class.csv').exists():
            frame = pd.read_csv(root / 'accuracy_by_class.csv').set_index('true_label')
            required_floors['main_categories_at_least_92_percent'] = all(
                label in frame.index and float(frame.loc[label, 'objective_match']) / int(frame.loc[label, 'total']) >= .92
                for label in ('AP', 'FLP', 'NRM', 'RA', 'TP'))
    if policy.get('metric') == 'three_group_macro':
        passed = passed and overall_improved
        if policy.get('preserve_original_required_floors', True):
            passed = passed and bool(required_floors) and all(required_floors.values())
    else:
        passed = passed and all(r['no_decline'] for r in rows) and all(r['no_decline'] for r in classes)
    passed = passed and same_inputs
    result = {'version': version, 'baseline': baseline, 'complete': state.get('complete') is True,
              'inputs_unchanged': same_inputs, 'datasets': rows, 'classes': classes,
              'passed': bool(passed), 'comparison_metric': 'Objective Match; fixed denominators/tolerances',
              'policy': policy or 'Legacy per-group no-decline policy',
              'overall': overall, 'baseline_overall': reference_overall,
              'macro_delta_percentage_points': (overall['macro_accuracy'] - reference_overall['macro_accuracy']) * 100 if overall and reference_overall else None,
              'pooled_delta_matches': overall['pooled_matches'] - reference_overall['pooled_matches'] if overall and reference_overall else None,
              'overall_improved': overall_improved, 'required_floors': required_floors,
              'full_notebook_sha256': hashlib.sha256((root / 'frozen_notebook.ipynb').read_bytes()).hexdigest() if (root / 'frozen_notebook.ipynb').exists() else None}
    (report / 'launch_gate.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(report / 'comparison_vs_baseline.csv', index=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v8')
    parser.add_argument('--baseline', default='v7')
    args = parser.parse_args()
    check(args.version, args.baseline)
