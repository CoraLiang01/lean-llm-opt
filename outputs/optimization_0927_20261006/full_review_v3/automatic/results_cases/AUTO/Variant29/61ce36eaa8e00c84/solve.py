import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].tolist()
categories = df_category['category'].tolist()
item_to_category = df_item.set_index('item_ref')['category'].to_dict()
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
item_minimum_lot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
item_maximum_order = df_item.set_index('item_ref')['maximum_order'].to_dict()
df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
item_unit_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
category_min_quantity = df_category.set_index('category')['minimum_quantity'].to_dict()
category_max_quantity = df_category.set_index('category')['maximum_quantity'].to_dict()
category_activation_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
resources = sorted(df_usage['resource'].unique().tolist())
usage_rows = df_usage[df_usage['item_ref'].isin(authorized_items)].copy()
usage_rows['amount'] = usage_rows['amount'].astype(int)

def normalize_usage(row):
    amt = row['amount']
    unit = row['unit'].strip().casefold()
    if unit == 'kwh':
        return amt * 1000
    elif unit == 'wh':
        return amt
    elif unit == 'hour':
        return amt * 60
    elif unit == 'minute':
        return amt
    elif unit == 'liter':
        return amt * 1000
    elif unit == 'ml':
        return amt
    else:
        raise ValueError(f'Unknown unit in usage: {unit}')
usage_rows['amount_norm'] = usage_rows.apply(normalize_usage, axis=1)
item_resource_usage = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    r = row['resource']
    item_resource_usage[i, r] = row['amount_norm']
df_capacity['amount'] = df_capacity['amount'].astype(int)

def normalize_capacity(row):
    amt = row['amount']
    unit = row['unit'].strip().casefold()
    if unit == 'kwh':
        return amt * 1000
    elif unit == 'wh':
        return amt
    elif unit == 'hour':
        return amt * 60
    elif unit == 'minute':
        return amt
    elif unit == 'liter':
        return amt * 1000
    elif unit == 'ml':
        return amt
    else:
        raise ValueError(f'Unknown unit in capacity: {unit}')
df_capacity['amount_norm'] = df_capacity.apply(normalize_capacity, axis=1)
resource_capacity = df_capacity.groupby('resource')['amount_norm'].sum().to_dict()
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in authorized_items and p in authorized_items:
        prereq_pairs.append((i, p))
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_list = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in authorized_items and b in authorized_items:
        bundle_list.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
category_items = {c: [] for c in categories}
for i in authorized_items:
    c = item_to_category[i]
    category_items[c].append(i)
m = gp.Model('BakeryOrderNetBenefit')
x_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    min_lot = item_minimum_lot[i]
    max_order = item_maximum_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i])
    m.addConstr(x_vars[i] <= max_order * y_vars[i])
    m.addConstr(y_vars[i] <= 1)
for c in categories:
    items_in_c = category_items[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_min_quantity[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_max_quantity[c])
    for i in items_in_c:
        m.addConstr(z_vars[c] >= y_vars[i])
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * x_vars[i] for i in authorized_items)) <= resource_capacity.get(r, 0))
for (a, b) in incompat_pairs:
    m.addConstr(y_vars[a] + y_vars[b] <= 1)
for (i, p) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[p])
for (a, b) in bundle_list:
    m.addConstr(w_vars[a, b] <= y_vars[a])
    m.addConstr(w_vars[a, b] <= y_vars[b])
    m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
item_benefit_expr = gp.quicksum((item_unit_benefit[i] * x_vars[i] - item_fee[i] * y_vars[i] for i in authorized_items))
category_fee_expr = gp.quicksum((category_activation_fee[c] * z_vars[c] for c in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus[a, b] * w_vars[a, b] for (a, b) in bundle_list))
m.setObjective(item_benefit_expr - category_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.optimize()