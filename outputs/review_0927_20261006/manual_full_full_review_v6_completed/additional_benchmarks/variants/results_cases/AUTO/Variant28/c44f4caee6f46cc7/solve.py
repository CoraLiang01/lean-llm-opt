import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
items_df = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_ids = list(items_df['item_ref'])
category_of = items_df.set_index('item_ref')['category'].to_dict()
authorized_of = items_df.set_index('item_ref')['authorized'].astype(int).to_dict()
min_lot_of = items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
max_order_of = items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
unit_benefit_of = items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_of = items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
categories_df = df_category[df_category['table'].str.strip().str.casefold() == 'category']
category_ids = list(categories_df['category'])
min_qty_of = categories_df.set_index('category')['minimum_quantity'].astype(int).to_dict()
max_qty_of = categories_df.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_of = categories_df.set_index('category')['activation_fee_cents'].astype(int).to_dict()
bundles_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_keys = list(bundles_df.index)
bundle_item_a = bundles_df['item_a'].to_dict()
bundle_item_b = bundles_df['item_b'].to_dict()
bundle_bonus = bundles_df['bonus_cents'].astype(int).to_dict()
incompat_df = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = list(zip(incompat_df['item_a'], incompat_df['item_b']))
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = list(zip(requires_df['item_ref'], requires_df['prerequisite_ref']))
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage'].copy()
usage_resources = set(usage_df['resource'])
capacity_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_resources = set(capacity_df['resource'])
resource_ids = sorted(usage_resources | capacity_resources)
unit_to_canonical = {'ml': ('space', 1), 'liter': ('space', 1000), 'wh': ('power', 1), 'kwh': ('power', 1000), 'minute': ('labor', 1), 'hour': ('labor', 60)}
usage = {i: {r: 0 for r in resource_ids} for i in item_ids}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_to_canonical:
        raise ValueError(f'Unknown unit {unit} in usage table')
    canonical_r, factor = unit_to_canonical[unit]
    if canonical_r != r:
        raise ValueError(f'Resource/unit mismatch: {r} vs {unit}')
    usage[i][r] += amt * factor
capacity = {r: 0 for r in resource_ids}
for _, row in capacity_df.iterrows():
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_to_canonical:
        raise ValueError(f'Unknown unit {unit} in capacity_ledger')
    canonical_r, factor = unit_to_canonical[unit]
    if canonical_r != r:
        raise ValueError(f'Resource/unit mismatch: {r} vs {unit}')
    capacity[r] += amt * factor
items_in_category = {c: [] for c in category_ids}
for i in item_ids:
    c = category_of[i]
    items_in_category[c].append(i)
m = gp.Model('central_fresh_order')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if authorized_of[i] != 1:
        m.addConstr(x_vars[i] == 0, name=f'unauth_x_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= min_lot_of[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= max_order_of[i] * y_vars[i], name=f'maxorder_{i}')
        m.addConstr(x_vars[i] <= max_order_of[i] * y_vars[i], name=f'link_x_y_{i}')
for c in category_ids:
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_category[c])) >= min_qty_of[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_category[c])) <= max_qty_of[c], name=f'cat_max_{c}')
for c in category_ids:
    for i in items_in_category[c]:
        m.addConstr(y_vars[i] <= z_vars[c], name=f'catact1_{c}_{i}')
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_category[c])), name=f'catact2_{c}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage[i][r] * x_vars[i] for i in item_ids)) <= capacity[r], name=f'rescap_{r}')
for i, j in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for i, prereq in prereq_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(y_vars[i] <= y_vars[prereq], name=f'prereq_{i}_{prereq}')
for b in bundle_keys:
    ia = bundle_item_a[b]
    ib = bundle_item_b[b]
    if ia in item_ids and ib in item_ids:
        m.addConstr(w_vars[b] <= y_vars[ia], name=f'bundle1_{b}')
        m.addConstr(w_vars[b] <= y_vars[ib], name=f'bundle2_{b}')
        m.addConstr(w_vars[b] >= y_vars[ia] + y_vars[ib] - 1, name=f'bundle3_{b}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle0_{b}')
obj = gp.quicksum((unit_benefit_of[i] * x_vars[i] - item_fee_of[i] * y_vars[i] for i in item_ids)) + gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_keys)) - gp.quicksum((activation_fee_of[c] * z_vars[c] for c in category_ids))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()