import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def to_base_unit(amount, unit):
    u = unit.strip().casefold()
    if u == 'ml':
        return amount
    elif u == 'liter':
        return amount * 1000
    elif u == 'wh':
        return amount
    elif u == 'kwh':
        return amount * 1000
    elif u == 'minute':
        return amount
    elif u == 'hour':
        return amount * 60
    else:
        raise ValueError(f'Unknown unit: {unit}')
item_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
category_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
usage_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
incompat_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
requires_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
bundle_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
identity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
item_df['authorized'] = item_df['authorized'].astype(int)
items_df = item_df[item_df['authorized'] == 1].copy()
items = list(items_df['item_ref'])
categories = list(category_df['category'])
item_to_category = items_df.set_index('item_ref')['category'].to_dict()
items_df['minimum_lot'] = items_df['minimum_lot'].astype(int)
items_df['maximum_order'] = items_df['maximum_order'].astype(int)
items_df['unit_benefit_cents'] = items_df['unit_benefit_cents'].astype(int)
items_df['item_fee_cents'] = items_df['item_fee_cents'].astype(int)
minimum_lot = items_df.set_index('item_ref')['minimum_lot'].to_dict()
maximum_order = items_df.set_index('item_ref')['maximum_order'].to_dict()
unit_benefit_cents = items_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = items_df.set_index('item_ref')['item_fee_cents'].to_dict()
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
minimum_quantity = category_df.set_index('category')['minimum_quantity'].to_dict()
maximum_quantity = category_df.set_index('category')['maximum_quantity'].to_dict()
activation_fee_cents = category_df.set_index('category')['activation_fee_cents'].to_dict()
usage_df = usage_df[usage_df['item_ref'].isin(items)]
usage_df['amount'] = usage_df['amount'].astype(int)
resources = sorted(usage_df['resource'].unique())
item_resource_usage = {}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit']
    amt_base = to_base_unit(amt, unit)
    item_resource_usage[i, r] = amt_base
capacity_df['amount'] = capacity_df['amount'].astype(int)
resource_capacity = {}
for r in capacity_df['resource'].unique():
    sub = capacity_df[capacity_df['resource'] == r]
    total = 0
    for _, row in sub.iterrows():
        amt = int(row['amount'])
        unit = row['unit']
        amt_base = to_base_unit(amt, unit)
        total += amt_base
    resource_capacity[r] = total
incompat_pairs = []
for _, row in incompat_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        incompat_pairs.append((a, b))
requires_pairs = []
for _, row in requires_df.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in items and p in items:
        requires_pairs.append((i, p))
bundle_pairs = []
bundle_bonus = {}
for _, row in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        bundle_pairs.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
category_items = {g: [] for g in categories}
for i in items:
    g = item_to_category[i]
    category_items[g].append(i)
m = gp.Model('central_fresh_order')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in items:
    m.addConstr(x_vars[i] >= 0, name=f'x_nonneg_{i}')
    m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i], name=f'x_max_{i}')
    m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i], name=f'x_min_{i}')
    m.addConstr(y_vars[i] <= 1, name=f'y_bin_{i}')
for g in categories:
    for i in category_items[g]:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'cat_act1_{g}_{i}')
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in category_items[g])), name=f'cat_act2_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) >= minimum_quantity[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) <= maximum_quantity[g], name=f'cat_max_{g}')
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * x_vars[i] for i in items)) <= resource_capacity[r], name=f'res_cap_{r}')
for i, j in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for i, p in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[p], name=f'prereq_{i}_{p}')
for i, j in bundle_pairs:
    m.addConstr(w_vars[i, j] <= y_vars[i], name=f'bundle1_{i}_{j}')
    m.addConstr(w_vars[i, j] <= y_vars[j], name=f'bundle2_{i}_{j}')
    m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{i}_{j}')
obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in items)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in items)) - gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in categories)) + gp.quicksum((bundle_bonus[i, j] * w_vars[i, j] for i, j in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()