import gurobipy as gp
import pandas as pd
import numpy as np
import re

def to_base_unit(amount, unit):
    u = unit.strip().casefold()
    if u == 'ml':
        return int(amount)
    elif u == 'liter':
        return int(amount) * 1000
    elif u == 'wh':
        return int(amount)
    elif u == 'kwh':
        return int(amount) * 1000
    elif u == 'minute':
        return int(amount)
    elif u == 'hour':
        return int(amount) * 60
    else:
        raise ValueError(f'Unknown unit: {unit}')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].tolist()
item_params = {}
for (_, row) in df_item.iterrows():
    i = row['item_ref']
    if i in authorized_items:
        item_params[i] = {'category': row['category'], 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'unit_benefit_cents': int(row['unit_benefit_cents']), 'item_fee_cents': int(row['item_fee_cents'])}
categories = df_category['category'].tolist()
cat_params = {}
for (_, row) in df_category.iterrows():
    g = row['category']
    cat_params[g] = {'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity']), 'activation_fee_cents': int(row['activation_fee_cents'])}
cat_items = {g: [] for g in categories}
for i in authorized_items:
    g = item_params[i]['category']
    cat_items[g].append(i)
usage_resources = df_usage['resource'].str.strip().str.casefold().unique()
capacity_resources = df_capacity['resource'].str.strip().str.casefold().unique()
resources = sorted(set(usage_resources).union(set(capacity_resources)))
item_resource_usage = {(i, r): 0 for i in authorized_items for r in resources}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource'].strip().casefold()
    if i in authorized_items and r in resources:
        amt = to_base_unit(row['amount'], row['unit'])
        item_resource_usage[i, r] += amt
resource_capacity = {r: 0 for r in resources}
for r in resources:
    df_r = df_capacity[df_capacity['resource'].str.strip().str.casefold() == r]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = to_base_unit(row['amount'], row['unit'])
        total += amt
    resource_capacity[r] = total
bundles = []
bundle_params = {}
for (_, row) in df_bundle.iterrows():
    if row['table'].strip().casefold() == 'bundle':
        a = row['item_a']
        b = row['item_b']
        if a in authorized_items and b in authorized_items:
            key = (a, b)
            bundles.append(key)
            bundle_params[key] = int(row['bonus_cents'])
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    if row['table'].strip().casefold() == 'incompatible':
        a = row['item_a']
        b = row['item_b']
        if a in authorized_items and b in authorized_items:
            incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    if row['table'].strip().casefold() == 'requires':
        i = row['item_ref']
        prereq = row['prerequisite_ref']
        if i in authorized_items and prereq in authorized_items:
            prereq_pairs.append((i, prereq))
m = gp.Model('central_fresh_order')
x_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    min_lot = item_params[i]['minimum_lot']
    max_ord = item_params[i]['maximum_order']
    m.addConstr(x_vars[i] == 0, name=f'x0_{i}').setAttr('Lazy', 1)
    m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'x_minlot_{i}')
    m.addConstr(x_vars[i] <= max_ord * y_vars[i], name=f'x_maxord_{i}')
for g in categories:
    for i in cat_items[g]:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'cat_act_{g}_{i}')
for g in categories:
    items_in_g = cat_items[g]
    min_q = cat_params[g]['minimum_quantity']
    max_q = cat_params[g]['maximum_quantity']
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= min_q, name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= max_q, name=f'cat_max_{g}')
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * x_vars[i] for i in authorized_items)) <= resource_capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[prereq], name=f'prereq_{i}_{prereq}')
for (a, b) in bundles:
    m.addConstr(w_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
    m.addConstr(w_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
    m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
obj_benefit = gp.quicksum((item_params[i]['unit_benefit_cents'] * x_vars[i] for i in authorized_items))
obj_item_fees = gp.quicksum((item_params[i]['item_fee_cents'] * y_vars[i] for i in authorized_items))
obj_cat_fees = gp.quicksum((cat_params[g]['activation_fee_cents'] * z_vars[g] for g in categories))
obj_bundles = gp.quicksum((bundle_params[b] * w_vars[b] for b in bundles))
m.setObjective(obj_benefit - obj_item_fees - obj_cat_fees + obj_bundles, gp.GRB.MAXIMIZE)
m.optimize()