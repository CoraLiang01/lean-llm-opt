import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def convert_to_base(amount, from_unit, resource):
    from_unit = from_unit.strip().casefold()
    resource = resource.strip().casefold()
    if resource == 'space':
        if from_unit == 'liter':
            return amount * 1000
        elif from_unit == 'ml':
            return amount
        else:
            raise ValueError(f'Unknown unit for space: {from_unit}')
    elif resource == 'power':
        if from_unit == 'kwh':
            return amount * 1000
        elif from_unit == 'wh':
            return amount
        else:
            raise ValueError(f'Unknown unit for power: {from_unit}')
    elif resource == 'labor':
        if from_unit == 'hour':
            return amount * 60
        elif from_unit == 'minute':
            return amount
        else:
            raise ValueError(f'Unknown unit for labor: {from_unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
items_df = df_item[df_item['authorized'] == 1].copy()
items = list(items_df['item_ref'])
if len(items) == 0:
    raise ValueError('No authorized items found.')
categories = list(df_category['category'])
usage_resources = set(df_usage['resource'].str.strip())
capacity_resources = set(df_capacity['resource'].str.strip())
resources = sorted(usage_resources.union(capacity_resources))
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    j = row['prerequisite_ref']
    if i in items and j in items:
        prereq_pairs.append((i, j))
bundles = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        key = (a, b)
        bundles.append(key)
        bundle_bonus[key] = int(row['bonus_cents'])
cat_min_qty = df_category.set_index('category')['minimum_quantity'].astype(int).to_dict()
cat_max_qty = df_category.set_index('category')['maximum_quantity'].astype(int).to_dict()
cat_activation_fee = df_category.set_index('category')['activation_fee_cents'].astype(int).to_dict()
item_min_lot = items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_max_order = items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_unit_benefit = items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee = items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
item_category = items_df.set_index('item_ref')['category'].to_dict()
cat_items = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    cat_items[c].append(i)
usage_dict = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource'].strip()
    if i in items:
        amt = int(row['amount'])
        unit = row['unit']
        amt_base = convert_to_base(amt, unit, r)
        usage_dict[i, r] = amt_base
for i in items:
    for r in resources:
        if (i, r) not in usage_dict:
            usage_dict[i, r] = 0
capacity_dict = {}
for r in resources:
    df_r = df_capacity[df_capacity['resource'].str.strip() == r]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = int(row['amount'])
        unit = row['unit']
        amt_base = convert_to_base(amt, unit, r)
        total += amt_base
    capacity_dict[r] = total
m = gp.Model('central_fresh_order')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    min_lot = item_min_lot[i]
    max_ord = item_max_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i])
    m.addConstr(x_vars[i] <= max_ord * y_vars[i])
    m.addConstr(x_vars[i] == 0, name=f'x0_{i}').setAttr('Lazy', 1)
for c in categories:
    m.addConstr(gp.quicksum((x_vars[i] for i in cat_items[c])) >= cat_min_qty[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in cat_items[c])) <= cat_max_qty[c])
for c in categories:
    for i in cat_items[c]:
        m.addConstr(z_vars[c] >= y_vars[i])
for r in resources:
    m.addConstr(gp.quicksum((x_vars[i] * usage_dict[i, r] for i in items)) <= capacity_dict[r])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[j])
for (i, j) in bundles:
    m.addConstr(w_vars[i, j] <= y_vars[i])
    m.addConstr(w_vars[i, j] <= y_vars[j])
    m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
obj = gp.quicksum((item_unit_benefit[i] * x_vars[i] - item_fee[i] * y_vars[i] for i in items)) + gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundles)) - gp.quicksum((cat_activation_fee[c] * z_vars[c] for c in categories))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()