import gurobipy as gp
import pandas as pd
import numpy as np
import re

def read_csv(path):
    return pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
paths = {'benefit': 'benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', 'bundle': 'benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', 'capacity_ledger': 'benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', 'category': 'benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', 'identity': 'benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', 'incompatible': 'benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', 'item_1': 'benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', 'item_2': 'benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', 'item_fee': 'benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', 'market': 'benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', 'requires': 'benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', 'usage_1': 'benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', 'usage_2': 'benchmark_dataset/Variant34/inputs/batch_01/export_13.csv'}
dfs = {k: read_csv(v) for (k, v) in paths.items()}
item_tables = [dfs['item_1'], dfs['item_2']]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows['category'] = item_rows['category'].astype(str)
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
item_rows['location_id'] = item_rows['location_id'].astype(str)
item_options = []
item_option_keys = set()
for (idx, row) in item_rows.iterrows():
    key = (row['item_ref'], row['location_id'])
    if key not in item_option_keys:
        item_options.append({'item_ref': row['item_ref'], 'location_id': row['location_id'], 'category': row['category'], 'authorized': row['authorized'], 'minimum_lot': row['minimum_lot'], 'maximum_order': row['maximum_order']})
        item_option_keys.add(key)
item_option_keys = set(((x['item_ref'], x['location_id']) for x in item_options))
item2cat = {(x['item_ref'], x['location_id']): x['category'] for x in item_options}
item2auth = {(x['item_ref'], x['location_id']): x['authorized'] for x in item_options}
item2minlot = {(x['item_ref'], x['location_id']): x['minimum_lot'] for x in item_options}
item2maxorder = {(x['item_ref'], x['location_id']): x['maximum_order'] for x in item_options}
cat_rows = dfs['category']
cat_rows['minimum_quantity'] = cat_rows['minimum_quantity'].astype(int)
cat_rows['maximum_quantity'] = cat_rows['maximum_quantity'].astype(int)
cat_rows['activation_fee_cents'] = cat_rows['activation_fee_cents'].astype(int)
categories = list(cat_rows['category'].unique())
cat2minqty = dict(zip(cat_rows['category'], cat_rows['minimum_quantity']))
cat2maxqty = dict(zip(cat_rows['category'], cat_rows['maximum_quantity']))
cat2fee = dict(zip(cat_rows['category'], cat_rows['activation_fee_cents']))
benefit_rows = dfs['benefit']
benefit_rows['amount_cents'] = benefit_rows['amount_cents'].astype(int)
benefit_sum = benefit_rows.groupby('item_ref')['amount_cents'].sum().to_dict()
item2benefit = {}
for key in item_option_keys:
    item_ref = key[0]
    item2benefit[key] = benefit_sum.get(item_ref, 0)
item_fee_rows = dfs['item_fee']
item_fee_rows['activation_fee_cents'] = item_fee_rows['activation_fee_cents'].astype(int)
itemfee_map = dict(zip(item_fee_rows['item_ref'], item_fee_rows['activation_fee_cents']))
item2fee = {}
for key in item_option_keys:
    item_ref = key[0]
    item2fee[key] = itemfee_map.get(item_ref, 0)
bundle_rows = dfs['bundle']
bundle_rows['bonus_cents'] = bundle_rows['bonus_cents'].astype(int)
bundle_pairs = []
bundle2bonus = {}
for (idx, row) in bundle_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_pairs.append((a, b))
    bundle2bonus[a, b] = row['bonus_cents']
incomp_rows = dfs['incompatible']
incomp_pairs = []
for (idx, row) in incomp_rows.iterrows():
    incomp_pairs.append((row['item_a'], row['item_b']))
req_rows = dfs['requires']
prereq_pairs = []
for (idx, row) in req_rows.iterrows():
    prereq_pairs.append((row['item_ref'], row['prerequisite_ref']))
cap_rows = dfs['capacity_ledger']
cap_rows['amount'] = cap_rows['amount'].astype(int)
cap_rows['unit'] = cap_rows['unit'].str.strip().str.casefold()
resource_caps = cap_rows.groupby('resource')['amount'].sum().to_dict()
usage_rows = pd.concat([dfs['usage_1'], dfs['usage_2']], ignore_index=True)
usage_rows['amount'] = usage_rows['amount'].astype(int)
usage_rows['unit'] = usage_rows['unit'].str.strip().str.casefold()

def to_ml(row):
    if row['unit'] == 'ml':
        return row['amount']
    elif row['unit'] == 'liter':
        return row['amount'] * 1000
    else:
        raise ValueError(f"Unknown unit: {row['unit']}")
usage_rows['amount_ml'] = usage_rows.apply(to_ml, axis=1)
itemres2usage = {}
for (idx, row) in usage_rows.iterrows():
    item_ref = row['item_ref']
    resource = row['resource']
    itemres2usage[item_ref, resource] = row['amount_ml']
item2resource = {(x['item_ref'], x['location_id']): x['location_id'] for x in item_options}
item2usage = {}
for key in item_option_keys:
    (item_ref, location_id) = key
    item2usage[key] = itemres2usage.get((item_ref, location_id), 0)
resource2items = {}
for key in item_option_keys:
    resource = key[1]
    resource2items.setdefault(resource, []).append(key)
cat2items = {}
for key in item_option_keys:
    cat = item2cat[key]
    cat2items.setdefault(cat, []).append(key)
itemref2options = {}
for key in item_option_keys:
    (item_ref, location_id) = key
    itemref2options.setdefault(item_ref, []).append(key)
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_option_keys, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_option_keys, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for key in item_option_keys:
    auth = item2auth[key]
    minlot = item2minlot[key]
    maxorder = item2maxorder[key]
    if auth == 0:
        m.addConstr(x_vars[key] == 0)
        m.addConstr(y_vars[key] == 0)
    else:
        m.addConstr(x_vars[key] >= minlot * y_vars[key])
        m.addConstr(x_vars[key] <= maxorder * y_vars[key])
        m.addConstr(x_vars[key] >= 0)
        m.addConstr(y_vars[key] <= 1)
        m.addConstr(y_vars[key] >= 0)
        m.addConstr((x_vars[key] >= 1) >> (y_vars[key] == 1))
        m.addConstr((y_vars[key] == 0) >> (x_vars[key] == 0))
for (resource, cap) in resource_caps.items():
    items = resource2items.get(resource, [])
    m.addConstr(gp.quicksum((item2usage[key] * x_vars[key] for key in items)) <= cap)
for cat in categories:
    items = cat2items.get(cat, [])
    m.addConstr(gp.quicksum((x_vars[key] for key in items)) >= cat2minqty[cat])
    m.addConstr(gp.quicksum((x_vars[key] for key in items)) <= cat2maxqty[cat])
for cat in categories:
    items = cat2items.get(cat, [])
    for key in items:
        m.addConstr(y_vars[key] <= z_vars[cat])
    m.addConstr(z_vars[cat] <= gp.quicksum((y_vars[key] for key in items)))
for (item_a, item_b) in bundle_pairs:
    a_options = itemref2options.get(item_a, [])
    b_options = itemref2options.get(item_b, [])
    m.addConstr(b_vars[item_a, item_b] <= gp.quicksum((y_vars[key] for key in a_options)))
    m.addConstr(b_vars[item_a, item_b] <= gp.quicksum((y_vars[key] for key in b_options)))
    m.addConstr(b_vars[item_a, item_b] >= gp.quicksum((y_vars[key] for key in a_options)) + gp.quicksum((y_vars[key] for key in b_options)) - 1)
for (item_a, item_b) in incomp_pairs:
    a_options = itemref2options.get(item_a, [])
    b_options = itemref2options.get(item_b, [])
    if a_options and b_options:
        m.addConstr(gp.quicksum((y_vars[key] for key in a_options)) + gp.quicksum((y_vars[key] for key in b_options)) <= 1)
for (item_ref, prereq_ref) in prereq_pairs:
    item_options_ = itemref2options.get(item_ref, [])
    prereq_options = itemref2options.get(prereq_ref, [])
    if item_options_ and prereq_options:
        for key in item_options_:
            m.addConstr(y_vars[key] <= gp.quicksum((y_vars[k] for k in prereq_options)))
benefit_expr = gp.quicksum((item2benefit[key] * x_vars[key] for key in item_option_keys))
itemfee_expr = gp.quicksum((item2fee[key] * y_vars[key] for key in item_option_keys))
catfee_expr = gp.quicksum((cat2fee[cat] * z_vars[cat] for cat in categories))
bundle_expr = gp.quicksum((bundle2bonus[bundle] * b_vars[bundle] for bundle in bundle_pairs))
total_expr = benefit_expr - itemfee_expr - catfee_expr + bundle_expr
m.setObjective(total_expr, gp.GRB.MAXIMIZE)
m.optimize()