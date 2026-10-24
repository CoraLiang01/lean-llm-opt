import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_paths = ['benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant28/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant28/inputs/batch_03/export_09.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
bundle_df = dfs[0]
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundles = list(bundle_df.index)
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
bundle_bonus = {k: v for (k, v) in zip(bundles, bundle_df['bonus_cents'])}
bundle_a = {k: v for (k, v) in zip(bundles, bundle_df['item_a'])}
bundle_b = {k: v for (k, v) in zip(bundles, bundle_df['item_b'])}
cap_df = dfs[1]
cap_df = cap_df[cap_df['table'].str.strip().str.casefold() == 'capacity_ledger']
cap_df['amount'] = cap_df['amount'].astype(int)
cap_df['resource_norm'] = cap_df['resource'].str.strip().str.casefold()
cap_df['unit_norm'] = cap_df['unit'].str.strip().str.casefold()
resource_units = {'space': 'ml', 'power': 'wh', 'labor': 'minute'}
unit_conversion = {('liter', 'ml'): 1000, ('ml', 'ml'): 1, ('kwh', 'wh'): 1000, ('wh', 'wh'): 1, ('hour', 'minute'): 60, ('minute', 'minute'): 1}
resources = sorted(cap_df['resource_norm'].unique())
resource_capacity = {}
for r in resources:
    base_unit = resource_units[r]
    rows = cap_df[cap_df['resource_norm'] == r]
    total = 0
    for (_, row) in rows.iterrows():
        from_unit = row['unit_norm']
        to_unit = base_unit
        key = (from_unit, to_unit)
        if key not in unit_conversion:
            raise ValueError(f'Unknown unit conversion: {from_unit} to {to_unit}')
        total += int(row['amount']) * unit_conversion[key]
    resource_capacity[r] = total
cat_df = dfs[2]
cat_df = cat_df[cat_df['table'].str.strip().str.casefold() == 'category']
cat_df['minimum_quantity'] = cat_df['minimum_quantity'].astype(int)
cat_df['maximum_quantity'] = cat_df['maximum_quantity'].astype(int)
cat_df['activation_fee_cents'] = cat_df['activation_fee_cents'].astype(int)
categories = list(cat_df['category'])
cat_min = dict(zip(cat_df['category'], cat_df['minimum_quantity']))
cat_max = dict(zip(cat_df['category'], cat_df['maximum_quantity']))
cat_fee = dict(zip(cat_df['category'], cat_df['activation_fee_cents']))
id_df = dfs[3]
id_df = id_df[id_df['table'].str.strip().str.casefold() == 'identity']
item_ref_to_entity_id = dict(zip(id_df['ref'], id_df['entity_id']))
incomp_df = dfs[4]
incomp_df = incomp_df[incomp_df['table'].str.strip().str.casefold() == 'incompatible']
incomp_pairs = [(row['item_a'], row['item_b']) for (_, row) in incomp_df.iterrows()]
item_df = dfs[5]
item_df = item_df[item_df['table'].str.strip().str.casefold() == 'item']
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['unit_benefit_cents'] = item_df['unit_benefit_cents'].astype(int)
item_df['item_fee_cents'] = item_df['item_fee_cents'].astype(int)
item_df = item_df[item_df['authorized'] == 1]
items = list(item_df['item_ref'])
item_category = dict(zip(item_df['item_ref'], item_df['category']))
item_minlot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_maxorder = dict(zip(item_df['item_ref'], item_df['maximum_order']))
item_benefit = dict(zip(item_df['item_ref'], item_df['unit_benefit_cents']))
item_fee = dict(zip(item_df['item_ref'], item_df['item_fee_cents']))
req_df = dfs[7]
req_df = req_df[req_df['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in req_df.iterrows()]
usage_df = dfs[8]
usage_df = usage_df[usage_df['table'].str.strip().str.casefold() == 'usage']
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['resource_norm'] = usage_df['resource'].str.strip().str.casefold()
usage_df['unit_norm'] = usage_df['unit'].str.strip().str.casefold()
usage_df = usage_df[usage_df['item_ref'].isin(items)]
item_resource_usage = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource_norm']
    from_unit = row['unit_norm']
    to_unit = resource_units[r]
    key = (from_unit, to_unit)
    if key not in unit_conversion:
        raise ValueError(f'Unknown unit conversion: {from_unit} to {to_unit}')
    amt = int(row['amount']) * unit_conversion[key]
    item_resource_usage[i, r] = amt
bundle_indices = []
bundle_bonus_cents = {}
bundle_item_a = {}
bundle_item_b = {}
for (idx, (a, b)) in enumerate(bundle_pairs):
    if a in items and b in items:
        bundle_indices.append(idx)
        bundle_bonus_cents[idx] = bundle_df.iloc[idx]['bonus_cents']
        bundle_item_a[idx] = a
        bundle_item_b[idx] = b
incomp_indices = []
for (a, b) in incomp_pairs:
    if a in items and b in items:
        incomp_indices.append((a, b))
prereq_indices = []
for (i, pre) in prereq_pairs:
    if i in items and pre in items:
        prereq_indices.append((i, pre))
m = gp.Model('central_fresh_produce_order')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_indices, vtype=gp.GRB.BINARY, name='')
for i in items:
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    m.addConstr(q_vars[i] >= minlot * z_vars[i], name=f'minlot_{i}')
    m.addConstr(q_vars[i] <= maxorder * z_vars[i], name=f'maxorder_{i}')
    m.addConstr(q_vars[i] >= 0, name=f'nonneg_{i}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    if items_in_c:
        for i in items_in_c:
            m.addConstr(z_vars[i] <= y_vars[c], name=f'catact1_{i}_{c}')
        m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)), name=f'catact2_{c}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    if items_in_c:
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= cat_min[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= cat_max[c], name=f'catmax_{c}')
for r in resources:
    total_usage = gp.quicksum((item_resource_usage.get((i, r), 0) * q_vars[i] for i in items))
    m.addConstr(total_usage <= resource_capacity[r], name=f'rescap_{r}')
for (a, b) in incomp_indices:
    m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incomp_{a}_{b}')
for (i, pre) in prereq_indices:
    m.addConstr(z_vars[i] <= z_vars[pre], name=f'prereq_{i}_{pre}')
for idx in bundle_indices:
    a = bundle_item_a[idx]
    b = bundle_item_b[idx]
    m.addConstr(b_vars[idx] <= z_vars[a], name=f'bundle1_{idx}')
    m.addConstr(b_vars[idx] <= z_vars[b], name=f'bundle2_{idx}')
    m.addConstr(b_vars[idx] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{idx}')
obj = gp.quicksum((item_benefit[i] * q_vars[i] for i in items)) - gp.quicksum((item_fee[i] * z_vars[i] for i in items)) - gp.quicksum((cat_fee[c] * y_vars[c] for c in categories)) + gp.quicksum((bundle_bonus_cents[idx] * b_vars[idx] for idx in bundle_indices))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()