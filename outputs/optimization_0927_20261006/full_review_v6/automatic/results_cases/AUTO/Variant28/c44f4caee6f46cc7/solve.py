import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_07.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv'
df_bundle = read_csv(f_bundle)
df_capacity = read_csv(f_capacity)
df_category = read_csv(f_category)
df_identity = read_csv(f_identity)
df_incompat = read_csv(f_incompat)
df_item = read_csv(f_item)
df_market = read_csv(f_market)
df_requires = read_csv(f_requires)
df_usage = read_csv(f_usage)
df_item['authorized'] = df_item['authorized'].astype(int)
authorized_items = df_item[df_item['authorized'] == 1].copy()
item_ids = authorized_items['item_ref'].tolist()
item_ids_set = set(item_ids)
category_ids = df_category['category'].tolist()
category_ids_set = set(category_ids)
usage_resources = set(df_usage['resource'])
capacity_resources = set(df_capacity['resource'])
resource_ids = sorted(usage_resources | capacity_resources)
resource_ids_set = set(resource_ids)
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_ids_set and b in item_ids_set:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in item_ids_set and p in item_ids_set:
        prereq_pairs.append((i, p))
bundle_pairs = []
bundle_bonus = dict()
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_ids_set and b in item_ids_set:
        bundle_pairs.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])

def get_int_map(df, key_col, val_col, restrict_keys=None):
    d = {}
    for (_, row) in df.iterrows():
        k = row[key_col]
        if restrict_keys is None or k in restrict_keys:
            d[k] = int(row[val_col])
    return d
minimum_lot = get_int_map(authorized_items, 'item_ref', 'minimum_lot')
maximum_order = get_int_map(authorized_items, 'item_ref', 'maximum_order')
unit_benefit_cents = get_int_map(authorized_items, 'item_ref', 'unit_benefit_cents')
item_fee_cents = get_int_map(authorized_items, 'item_ref', 'item_fee_cents')
item_category = dict(zip(authorized_items['item_ref'], authorized_items['category']))
minimum_quantity = get_int_map(df_category, 'category', 'minimum_quantity')
maximum_quantity = get_int_map(df_category, 'category', 'maximum_quantity')
activation_fee_cents = get_int_map(df_category, 'category', 'activation_fee_cents')
items_by_category = {g: [] for g in category_ids}
for i in item_ids:
    g = item_category[i]
    items_by_category[g].append(i)
unit_map = {'ml': ('space', 1), 'liter': ('space', 1000), 'wh': ('power', 1), 'kwh': ('power', 1000), 'minute': ('labor', 1), 'hour': ('labor', 60)}
usage = {i: {r: 0 for r in resource_ids} for i in item_ids}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    if i not in item_ids_set:
        continue
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_map:
        raise ValueError(f'Unknown unit {unit} in usage table')
    (base_r, factor) = unit_map[unit]
    if base_r != r:
        raise ValueError(f'Resource/unit mismatch: {r} vs {unit}')
    usage[i][r] += amt * factor
capacity = {r: 0 for r in resource_ids}
for (_, row) in df_capacity.iterrows():
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_map:
        raise ValueError(f'Unknown unit {unit} in capacity_ledger')
    (base_r, factor) = unit_map[unit]
    if base_r != r:
        raise ValueError(f'Resource/unit mismatch: {r} vs {unit}')
    capacity[r] += amt * factor
m = gp.Model('central_fresh_order')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
    m.addConstr(x_vars[i] >= 0)
for g in category_ids:
    items = items_by_category[g]
    for i in items:
        m.addConstr(z_vars[g] >= y_vars[i])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items)))
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) >= minimum_quantity[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) <= maximum_quantity[g])
for r in resource_ids:
    m.addConstr(gp.quicksum((usage[i][r] * x_vars[i] for i in item_ids)) <= capacity[r])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, p) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[p])
for (i, j) in bundle_pairs:
    m.addConstr(b_vars[i, j] <= y_vars[i])
    m.addConstr(b_vars[i, j] <= y_vars[j])
    m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in item_ids)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in item_ids)) - gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in category_ids)) + gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()