"""Prespecified complete-pass gate before any ablation/LOTO API experiment."""
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
            passed = passed and complete and no_decline
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
            passed = passed and no_decline
    else:
        passed = False
    same_inputs = False
    if (root / 'manifest.json').exists():
        same_inputs = json.loads((root / 'manifest.json').read_text())['inputs'] == json.loads((old / 'manifest.json').read_text())['inputs']
    passed = passed and same_inputs
    result = {'version': version, 'baseline': baseline, 'complete': state.get('complete') is True,
              'inputs_unchanged': same_inputs, 'datasets': rows, 'classes': classes,
              'passed': bool(passed), 'comparison_metric': 'Objective Match; fixed denominators/tolerances',
              'policy': 'No decline for either dataset, any sheet or any 101 class; no case selection',
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
