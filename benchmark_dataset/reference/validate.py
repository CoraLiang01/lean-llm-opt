"""Validate all six cases from the exact input CSVs listed in questions.csv."""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent if HERE.name == 'reference' else HERE / 'benchmark_dataset'
ROOT = BASE.parent
spec = importlib.util.spec_from_file_location('solver_core', BASE / 'reference/solver_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
MANIFEST = json.loads((BASE / 'case_manifest.json').read_text()) if (BASE / 'case_manifest.json').exists() else None
NAMES = [c['case'] for c in MANIFEST] if MANIFEST else ['AP1', 'AP2', 'AP3', 'RA12', 'RA1', 'RA2']

def ap_tables(paths):
    data = [core.read_csv(p) for p in paths]
    wr = next(r for r in data if 'skill' in r[0])
    pr = next(r for r in data if 'required_skill' in r[0])
    cost_parts = [r for r in data if 'worker_id' in r[0] and 'skill' not in r[0]]
    matrix = [r for part in cost_parts for r in part]
    assert len({r['worker_id'] for r in wr}) == len(wr)
    assert len({r['project_id'] for r in pr}) == len(pr)
    long_form = 'cost_cents' in matrix[0]
    assert {r['worker_id'] for r in matrix} <= {r['worker_id'] for r in wr}
    ps = {r['project_id'] for r in pr}
    assert set(matrix[0]) == ({'worker_id','project_id','cost_cents'} if long_form else ps | {'worker_id'})
    t = {'worker': [{'worker_ref': r['worker_id'], 'skill': r['skill'], 'on_leave': r['on_leave']} for r in wr],
         'project': [{'project_ref': r['project_id'], 'required_skill': r['required_skill']} for r in pr],
         'route': [], 'charge': [], 'fx': [{'currency': 'USD', 'usd_cents_numerator': '1', 'denominator': '1'}]}
    for r in matrix:
        for p in ([r['project_id']] if long_form else sorted(ps)):
            if long_form or r[p] != '':
                assert p in ps
                cost = int(r['cost_cents'] if long_form else r[p])
                e = r['worker_id'] + '_' + p
                t['route'].append({'edge_id': e, 'worker_ref': r['worker_id'], 'project_ref': p})
                t['charge'].append({'edge_id': e, 'amount': str(cost), 'currency': 'USD'})
    return t

def ra_tables(paths, cfg):
    if cfg.get('input_style', 'events') == 'events':
        t = core.decode(paths, **{k: cfg[k] for k in ['scope_column','scope_value','asof']})
    else:
        from collections import defaultdict
        t = defaultdict(list)
        for p in paths:
            for row in core.read_csv(p):
                table = row.pop('table')
                t[table].append(row)
    if 'fx' not in t:
        t['fx']=[{'currency':'USD','usd_cents_numerator':'1','denominator':'1'}]
    if t.get('item') and 'unit_benefit_cents' in t['item'][0]:
        t['benefit']=[];t['item_fee']=[]
        for row in t['item']:
            t['benefit'].append({'item_ref':row['item_ref'],'amount':row.pop('unit_benefit_cents'),'currency':'USD'})
            t['item_fee'].append({'item_ref':row['item_ref'],'activation_fee_cents':row.pop('item_fee_cents')})
    for row in t.get('benefit',[]):
        if 'amount_cents' in row:
            row['amount']=row.pop('amount_cents');row['currency']='USD'
    return t

def main():
    publish = '--publish' in sys.argv
    questions = BASE / 'questions.csv'
    rows = core.read_csv(questions)
    assert len(rows) == len(NAMES)
    assert list(rows[0]) == ['Type by size', 'Token', 'Problem Type', 'Query', 'Dataset_address', 'Label-objective', 'Label-model']
    report = []
    hashes = {}
    for i, (name, row) in enumerate(zip(NAMES, rows)):
        paths = [ROOT / p for p in row['Dataset_address'].splitlines()]
        assert paths and len(paths) == len(set(paths))
        assert len({p.name for p in paths}) == len(paths), 'Ambiguous basename'
        assert set(paths) == set((BASE / name / 'inputs').rglob('*.csv'))
        for p in paths:
            assert p.is_file() and p.is_relative_to(BASE / name / 'inputs')
            with p.open(encoding='utf-8-sig', newline='') as f:
                records = list(csv.reader(f))
            assert len(records) > 1 and all(len(r) == len(records[0]) for r in records)
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        if MANIFEST[i].get('validation_mode') == 'copy_integrity':
            # Variants are copied unchanged; no claim of a fresh optimization audit.
            merge = json.loads((BASE/'reference/variants_merge.json').read_text())
            entry = next(e for e in merge['variants'] if e['case']==name)
            assert entry['row']==i+1
            assert {e['destination'] for e in entry['files']} == {str(p.relative_to(ROOT)) for p in paths}
            for e in entry['files']:
                assert hashes[e['destination']]==e['sha256']
            report.append({'case':name,'input_files':len(paths),'label_objective':row['Label-objective'],'status':'COPY_INTEGRITY_VERIFIED','solver_rerun':False})
            continue
        t = ap_tables(paths) if i < 3 else ra_tables(paths, MANIFEST[i])
        ref = BASE / name / 'reference'
        if i < 3:
            m, s, obj = core.solve_ap(t)
        else:
            m, s = core.build_model(t, 'RA')
            obj = core.audit(t, 'RA', s)
        if publish:
            if i >= 3:
                assert obj == int(row['Label-objective'])
            row['Label-objective'] = str(obj)
            (ref / 'normalized_tables.json').write_text(json.dumps(t, indent=2))
            (ref / 'solution.json').write_text(json.dumps({'objective_usd_cents': obj, 'decisions': s}, indent=2))
            m.write(str(ref / 'model.lp'))
        else:
            assert obj == int(row['Label-objective'])
        assert row['Label-model'] == 'Reference LP file: ' + str((ref / 'model.lp').relative_to(ROOT))
        lp = core.gp.read(str(ref / 'model.lp'))
        lp.Params.OutputFlag = 0
        lp.Params.MIPGap = 0
        lp.optimize()
        assert lp.Status == core.GRB.OPTIMAL and abs(lp.ObjVal - obj) < 1e-6
        report.append({'case': name, 'input_files': len(paths), 'objective_usd_cents': obj,
                       'input_bytes': sum(p.stat().st_size for p in paths), 'status': 'OPTIMAL',
                       'variables': m.NumVars, 'constraints': m.NumConstrs,
                       'independent_min_cost_flow': i < 3})
    if publish:
        with questions.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)
        target = ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101_冗余列35个 copy.csv'
        main = target.with_name('Large-scale-or-101_冗余列35个.csv')
        expected = json.loads((BASE / 'reference/consolidation.json').read_text())['main_csv_sha256']
        assert hashlib.sha256(main.read_bytes()).hexdigest() == expected
        target.write_bytes(questions.read_bytes())
        (BASE / 'reference/verification.json').write_text(json.dumps(report, indent=2))
        (BASE / 'reference/input_sha256.json').write_text(json.dumps(hashes, indent=2))
    with questions.open(encoding='utf-8-sig', newline='') as f:
        physical = list(csv.reader(f))
    assert len(physical) == len(NAMES) + 1 and all(len(r) == 7 for r in physical)
    assert all(len(c) < 32767 for r in physical for c in r)
    assert 'benchmark_hard_v1/' not in questions.read_text(encoding='utf-8-sig')
    assert 'benchmark_calibrated_v2/' not in questions.read_text(encoding='utf-8-sig')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
