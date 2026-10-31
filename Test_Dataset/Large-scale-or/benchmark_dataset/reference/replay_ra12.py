"""Replay the user's existing RA12 solve.py with only its input paths replaced.

Pass --source /absolute/path/to/OR-004/solve.py. No LLM or network calls.
"""
import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import warnings
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent

def replay(source, paths):
    tree = ast.parse(source)
    changed = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'csv_paths' for t in node.targets):
            node.value = ast.parse(repr([str(p) for p in paths]), mode='eval').body
            changed += 1
    assert changed == 1
    ast.fix_missing_locations(tree)
    ns = {'__name__': '__replay__'}
    with contextlib.redirect_stdout(io.StringIO()) as out, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exec(compile(tree, '<saved-user-RA12-solve-paths-only>', 'exec'), ns)
    m = ns['m']
    assert m.Status == 2
    return {'objective_usd_cents': round(m.ObjVal),
            'resource_capacity': ns['resource_capacity'],
            'available_items': len(ns['item_refs'])}, out.getvalue()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    def rows(p):
        with p.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    old = rows(BASE / 'reference/copy_before_consolidation.csv')[3]
    historical = ROOT / 'benchmark_archive/before_ra_expansion'
    new = rows(historical / 'questions.csv' if historical.exists() else BASE / 'questions.csv')[3]
    oldpaths = [ROOT / p for p in old['Dataset_address'].splitlines()]
    oldpaths = [p if p.exists() else ROOT / 'benchmark_archive' / p.relative_to(ROOT) for p in oldpaths]
    old_result, old_log = replay(source, oldpaths)
    newpaths = [ROOT / p for p in new['Dataset_address'].splitlines()]
    if historical.exists():
        newpaths = [historical / 'RA12_previous_active' / p.relative_to(BASE / 'RA12') for p in newpaths]
    new_result, new_log = replay(source, newpaths)
    report = {'source': str(args.source), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'only_change_to_saved_code': 'csv_paths replaced with corresponding local input paths',
              'old_inputs': old_result, 'new_inputs': new_result,
              'expected_objective_usd_cents': int(new['Label-objective']),
              'new_result_correct': new_result['objective_usd_cents'] == int(new['Label-objective'])}
    (BASE / 'reference/ra12_replay.json').write_text(json.dumps(report, indent=2))
    (BASE / 'reference/ra12_replay_old.log').write_text(old_log)
    (BASE / 'reference/ra12_replay_new.log').write_text(new_log)
    print(json.dumps(report, indent=2))
    assert old_result['objective_usd_cents'] == 25771
    assert report['new_result_correct']

if __name__ == '__main__':
    main()
