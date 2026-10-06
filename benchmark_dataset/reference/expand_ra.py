"""Build contextual RA cases from the accepted RA template; no model API calls."""
import csv
import hashlib
import importlib.util
import json
import random
import shutil
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'benchmark_dataset'
COPY = ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101_冗余列35个 copy.csv'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

core = load('core', BASE / 'reference/solver_core.py')
old = load('generator', ROOT / 'benchmark_archive/benchmark_hard_v1/reference/build.py')

# Scenario identifiers are fictional benchmark planning units.
SETTINGS = [
    ('RA3', 25, 'dealership_id', 'RIVERSIDE_AUTO', 'HARBOR_AUTO', '2026-04-16',
     'A vehicle dealership is planning its daily replenishment. One order unit is one vehicle of the listed model/configuration. Space is shared vehicle storage volume; labor and power are shared preparation resources.'),
    ('RA4', 28, 'dealership_id', 'OSLO_NEW_CARS', 'BERGEN_NEW_CARS', '2026-05-07',
     'A new-car dealership in Norway is planning an inventory replenishment. One unit is one vehicle of the listed model/configuration. Space is shared inventory storage volume; labor and power are shared vehicle preparation resources.'),
    ('RA7', 24, 'portfolio_id', 'NYC_REDEVELOPMENT', 'NJ_REDEVELOPMENT', '2026-06-18',
     'A New York property developer is selecting standardized development modules in named neighborhoods. One quantity unit is one indivisible development module. Space refers to the shared volume of construction materials in storage; labor and power are shared construction resources.'),
    ('RA12', 24, 'store_id', 'CENTRAL_FRESH', 'WEST_FRESH', '2026-07-09',
     'A supermarket is planning produce replenishment. One order unit is one case of the listed produce. Space, labor and power are the shared storage volume, handling time and refrigeration energy for this plan.'),
    ('RA10', 30, 'store_id', 'MARKET_SQUARE', 'RIVER_MARKET', '2026-08-13',
     'A supermarket is planning stock across three display sections. Each item_ref identifies a product-and-section option, and its section is given by location_id. One unit is one pack of that product placed in that section. Each section has its own display-volume capacity; capacity cannot be transferred between sections. Different section options for the same product are separate decisions.'),
    ('RA13', 30, 'catalog_id', 'ARCADE_CATALOG', 'INDIE_CATALOG', '2026-10-08',
     'A digital game store is allocating downloadable game editions across three platforms. Each item_ref identifies a game-edition-and-platform option, and location_id gives its platform. One unit is one deployable licensed edition package. Each platform has a separate memory capacity; capacity cannot be transferred between platforms. Different platform options for the same title are separate decisions. Monetary benefits and fees describe licensing economics.'),
    ('RA14', 28, 'fulfillment_center_id', 'FC_EAST_HVAC', 'FC_WEST_HVAC', '2026-11-12',
     'An online retailer is allocating air conditioners across three storage areas in one fulfillment center. Each item_ref identifies an air-conditioner-model-and-area option, with its area in location_id. One quantity unit is one air conditioner placed in that area. Each area has its own storage-volume capacity; capacity cannot be transferred between areas. Different area options for the same model are separate decisions.'),
]

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)

def canonical(t):
    return {k: sorted(json.dumps({a: str(b) for a,b in r.items()}, sort_keys=True) for r in rows) for k,rows in t.items()}

def make_tables(name, n, seed):
    if name == 'RA12':
        return json.loads((BASE / name / 'reference/normalized_tables.json').read_text())
    c = old.Case(name, seed)
    old.generate_ra(c, n)
    for currency, denom in [('USD',1),('EUR',2),('JPY',100)]:
        c.add('fx', currency=currency, usd_cents_numerator=1, denominator=denom)
    t = dict(c.tables)
    source = next((ROOT / f'Test_Dataset/Large-scale-or/RA_testing/{name}').rglob('products.csv'))
    products = core.read_csv(source)
    display = [r.get('ProductName', r.get('item_name')) for r in products]
    ids = {r['ref']: r for r in t['identity']}
    # Keep the original scenario's product names, with explicit configuration IDs.
    for i,r in enumerate(t['item']):
        ident = ids[r['item_ref']]
        ident['entity_id'] = f'{name}_OPTION_{i+1:02d}'
        ident['display_name'] = display[i % len(display)]
        r['configuration_id'] = f'CFG_{i // len(display)+1:02d}'
    if name in ['RA10','RA13','RA14']:
        locs = ['SECTION_A','SECTION_B','SECTION_C'] if name == 'RA10' else (['PC','CONSOLE','MOBILE'] if name == 'RA13' else ['AREA_A','AREA_B','AREA_C'])
        original = ['space','labor','power']
        locmap = {r['item_ref']: original[i % 3] for i,r in enumerate(t['item'])}
        for r in t['item']:
            r['location_id'] = locs[original.index(locmap[r['item_ref']])]
        t['usage'] = [r for r in t['usage'] if r['resource'] == locmap[r['item_ref']]]
        for r in t['usage']:
            r['resource'] = locs[original.index(r['resource'])]
            r['unit'] = 'GB' if name == 'RA13' else 'liter'
        for r in t['capacity_ledger']:
            divisor = 60 if r['resource'] == 'labor' else 1000
            assert int(r['amount']) % divisor == 0
            r['amount'] = int(r['amount']) // divisor * 1000
            r['resource'] = locs[original.index(r['resource'])]
            r['unit'] = 'MB' if name == 'RA13' else 'ml'
    return t

def publish_events(out, tables, scope, target, other, asof, seed):
    rng = random.Random(seed)
    day = date.fromisoformat(asof)
    dt = lambda days: (day + timedelta(days=days)).isoformat()
    paths = []
    for table, records in sorted(tables.items()):
        groups = defaultdict(list)
        for i,r in enumerate(records):
            base = dict(table=table, **{scope:target}, record_id=f'{table}:{i:04d}')
            current = dict(base, revision=2, effective_date=dt(-7), action='UPSERT', **r)
            prev = dict(r)
            for k,v in r.items():
                if str(v).lstrip('-').isdigit(): prev[k] = str(int(v)+7)
            events = [dict(base, revision=1, effective_date=dt(-35), action='UPSERT', **prev), current]
            if i % 7 == 0: events.append(dict(current))
            if i % 5 == 0: events.append(dict(base, revision=3, effective_date=dt(12), action='UPSERT', **prev))
            if i % 4 == 0: events.append(dict(current, **{scope:other}))
            for e in events: groups[e[scope], e['record_id']].append(e)
        base = dict(table=table, **{scope:target}, record_id=f'{table}:withdrawn')
        groups[target, base['record_id']] = [dict(base, revision=1,effective_date=dt(-35),action='UPSERT',**records[0]),
                                            dict(base,revision=2,effective_date=dt(-2),action='DELETE')]
        count = 1 if table == 'fx' else (3 if table in ['benefit','usage'] else 2)
        chunks = [[] for _ in range(count)]
        for i,key in enumerate(sorted(groups)): chunks[i % count].extend(groups[key])
        for chunk in chunks:
            rng.shuffle(chunk)
            i = len(paths)
            p = out / 'inputs' / f'batch_{i%6+1:02d}' / f'export_{i+1:02d}.csv'
            write_csv(p, chunk)
            paths.append(p)
    decoded = core.decode(paths, scope, target, asof)
    assert canonical(decoded) == canonical(tables)
    locations = {}
    for p in paths:
        for r in core.read_csv(p):
            key = r[scope],r['table'],r['record_id']
            assert key not in locations or locations[key] == p
            locations[key] = p
    return paths, decoded

def query(background, scope, target, asof, name):
    text = (background + f' Plan only for {scope}={target} as of {asof}. '
        f'The CSV exports identify their logical table in table. For each ({scope}, table, record_id), discard events after the planning date and use the greatest integer revision. '
        'Identical retransmissions count once. A selected DELETE removes the record without falling back to an older version. '
        'All versions of one record are in the same CSV. Distinct surviving record_ids are additive records. '
        f'References are scoped to {scope}; identity resolves item references. '
        'Choose nonnegative integer quantities to maximize net benefit. Only authorized=1 items may be selected. '
        'Each item quantity is either zero or between minimum_lot and maximum_order inclusive. '
        'Compute per-unit benefit by summing signed benefit components after currency conversion: amount * usd_cents_numerator / denominator from fx. '
        'Deduct item_fee once for each item with positive quantity. For every resource, total per-unit usage times quantity must not exceed the sum of its signed capacity_ledger entries. ')
    text += ('Convert GB to MB by 1000; MB is the base unit. ' if name == 'RA13' else
             ('Convert liter to ml by 1000; ml is the base unit. ' if name in ['RA10','RA14'] else
              'Convert liter to ml by 1000, hour to minute by 60, and kwh to wh by 1000. '))
    return text + ('Every category total must satisfy its minimum_quantity and maximum_quantity, unconditionally; deduct its activation fee once when any item in that category is selected. '
        'Incompatible items cannot both have positive quantities. A requires row permits ordering item_ref only if prerequisite_ref has positive quantity, with no quantity ratio. '
        'Each bundle adds bonus_cents once if both listed items have positive quantities. '
        'Report the maximum net benefit in USD cents. All monetary fees and bonuses are already in USD cents.')

def main():
    backup = ROOT / 'benchmark_archive/before_ra_expansion'
    assert not backup.exists(), 'Expansion already run; do not overwrite backup.'
    backup.mkdir()
    shutil.copyfile(COPY, backup / 'questions.csv')
    shutil.copytree(BASE / 'RA12', backup / 'RA12')
    rows = core.read_csv(COPY)
    assert len(rows) == 6
    preserved = [dict(rows[i]) for i in [0,1,2,4,5]]
    mainpath = COPY.with_name('Large-scale-or-101_冗余列35个.csv')
    mainhash = hashlib.sha256(mainpath.read_bytes()).hexdigest()
    sources = core.read_csv(mainpath)
    stage = BASE / '_expansion_stage'
    stage.mkdir()
    manifests = {name: {'case':name, 'scope_column':'tenant', 'scope_value':'NORTH', 'asof':'2026-09-20'} for name in ['RA12','RA1','RA2']}
    reports = []
    for j,(name,n,scope,target,other,asof,background) in enumerate(SETTINGS):
        t = make_tables(name,n,92000+j)
        paths,t = publish_events(stage/name,t,scope,target,other,asof,95000+j)
        model,solution = core.build_model(t,'RA')
        obj = core.audit(t,'RA',solution)
        ref = stage/name/'reference'
        ref.mkdir()
        model.write(str(ref/'model.lp'))
        (ref/'solution.json').write_text(json.dumps({'objective_usd_cents':obj,'decisions':solution},indent=2))
        (ref/'normalized_tables.json').write_text(json.dumps(t,indent=2))
        if name == 'RA12':
            row = rows[3]
            assert obj == int(row['Label-objective'])
        else:
            row = dict(next(r for r in sources if f'/RA_testing/{name}/' in r['Dataset_address']))
            rows.append(row)
        row.update({'Query':query(background,scope,target,asof,name), 'Token':'', 'Label-objective':str(obj),
            'Dataset_address':'\n'.join(str((BASE/name/p.relative_to(stage/name)).relative_to(ROOT)) for p in paths),
            'Label-model':f'Reference LP file: benchmark_dataset/{name}/reference/model.lp'})
        manifests[name] = {'case':name,'scope_column':scope,'scope_value':target,'asof':asof}
        reports.append({'case':name,'scope_column':scope,'scope_value':target,'asof':asof,'items':len(t['item']),'input_files':len(paths),'objective_usd_cents':obj})
    assert preserved == [rows[i] for i in [0,1,2,4,5]]
    assert hashlib.sha256(mainpath.read_bytes()).hexdigest() == mainhash
    for name,*_ in SETTINGS:
        if name == 'RA12': shutil.move(str(BASE/name),str(backup/'RA12_previous_active'))
        shutil.move(str(stage/name),str(BASE/name))
    stage.rmdir()
    order = ['AP1','AP2','AP3','RA12','RA1','RA2','RA3','RA4','RA7','RA10','RA13','RA14']
    assert len(rows) == len(order)
    (BASE/'case_manifest.json').write_text(json.dumps([manifests.get(n,{'case':n}) for n in order],indent=2))
    write_csv(BASE/'questions.csv',rows)
    COPY.write_bytes((BASE/'questions.csv').read_bytes())
    (BASE/'reference/ra_expansion.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))

if __name__ == '__main__': main()
