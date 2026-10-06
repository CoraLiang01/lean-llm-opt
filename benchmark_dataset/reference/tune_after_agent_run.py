"""Target four failures while preserving all reference objectives."""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'benchmark_dataset'
COPY=ROOT/'Test_Dataset/Large-scale-or/Large-scale-or-101_冗余列35个 copy.csv'
spec=importlib.util.spec_from_file_location('core',BASE/'reference/solver_core.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def write(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,quoting=csv.QUOTE_ALL)
        w.writeheader();w.writerows(rows)

def canonical(t):
    return {k:sorted(json.dumps(r,sort_keys=True) for r in v) for k,v in t.items()}

def main():
    backup=ROOT/'benchmark_archive/before_targeted_agent_tuning'
    assert not backup.exists(), 'Already applied.'
    backup.mkdir()
    shutil.copyfile(COPY,backup/'questions.csv')
    shutil.copytree(BASE/'RA7',backup/'RA7')
    shutil.copyfile(BASE/'reference/input_sha256.json',backup/'input_sha256.json')
    rows=core.read_csv(COPY)
    manifest=json.loads((BASE/'case_manifest.json').read_text())
    original=json.loads(json.dumps(rows))
    scope=manifest[8]
    paths=[ROOT/p for p in rows[8]['Dataset_address'].splitlines()]
    kwargs={k:scope[k] for k in ['scope_column','scope_value','asof']}
    before=core.decode(paths,**kwargs)
    snapshot_tables={'benefit','usage','capacity_ledger','fx'}
    changed=[]
    for p in paths:
        records=core.read_csv(p)
        table=records[0]['table']
        if table not in snapshot_tables:continue
        chosen={}
        for r in records:
            if r[scope['scope_column']]!=scope['scope_value'] or r['effective_date']>scope['asof']:continue
            if r['record_id'] not in chosen or int(r['revision'])>int(chosen[r['record_id']]['revision']):chosen[r['record_id']]=r
        live=[r for r in chosen.values() if r['action']!='DELETE']
        assert live
        write(p,list(records[0]),live)
        changed.append({'path':str(p.relative_to(ROOT)),'table':table,'before_rows':len(records),'after_rows':len(live)})
    assert canonical(core.decode(paths,**kwargs))==canonical(before)
    uniform=('Apply the same record-selection rule to every logical table, including usage, capacity_ledger, benefit and fx. '
             'Use the stated business unit and planning date throughout. Select by (table, record_id) after filtering the business unit and date; '
             'keep record_id available until version selection is complete. Only then join tables or aggregate surviving records.')
    for i in [3,9]:
        rows[i]['Query'] += ' '+uniform
    rows[6]['Query'] += (' For a binary selected indicator y and integer quantity x, the item selection rule is '
        'minimum_lot*y <= x <= maximum_order*y. All minimum_lot values are positive, so these bounds already make y=1 exactly when x is positive.')
    rows[8]['Query'] += (' The benefit, usage, capacity_ledger and fx exports are finalized snapshots for this planning date and portfolio, '
        'with one row per surviving record. Sum their signed entries once. The remaining tables use the event-selection rule above.')
    rows[9]['Query'] += (' An unauthorized option has quantity and selected status zero. A bundle earns zero if either option is unauthorized or unselected. '
        'If b indicates a bundle, enforce b<=y_a, b<=y_b and b>=y_a+y_b-1, treating unavailable options as y=0. '
        'Each category minimum is required even when no option has yet been selected.')
    assert all(rows[i]==original[i] for i in range(12) if i not in [3,6,8,9])
    assert all(r['Label-objective']==o['Label-objective'] and r['Dataset_address']==o['Dataset_address'] for r,o in zip(rows,original))
    write(BASE/'questions.csv',list(rows[0]),rows)
    COPY.write_bytes((BASE/'questions.csv').read_bytes())
    (BASE/'reference/targeted_tuning.json').write_text(json.dumps({'changed_cases':['RA12','RA3','RA7','RA10'],'labels_unchanged':True,'RA7_effective_tables_unchanged':True,'modified_files':changed},indent=2))
    print(json.dumps({'changed_queries':4,'changed_RA7_files':len(changed),'all_labels_unchanged':True}))

if __name__=='__main__':main()
