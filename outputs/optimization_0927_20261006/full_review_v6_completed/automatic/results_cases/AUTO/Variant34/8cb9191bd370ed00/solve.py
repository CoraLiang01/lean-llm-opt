import gurobipy as gp
import pandas as pd
import numpy as np
import re

def read_csv(path):
    return pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
df_benefit = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv')
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
df_bundle = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv')
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_capacity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv')
df_capacity['amount'] = df_capacity['amount'].astype(int)
df_category = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv')
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
df_identity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv')
df_incompat = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv')
df_item1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv')
df_item2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv')
df_item = pd.concat([df_item1, df_item2], ignore_index=True)
df_item['authorized'] = df_item['authorized'].astype(int)
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
df_item_fee = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv')
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv')
df_usage1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv')
df_usage2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv')
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['amount'] = df_usage['amount'].astype(int)
item_keys = []
item_info = dict()
for (_, row) in df_item.iterrows():
    key = (row['item_ref'], row['location_id'])
    item_keys.append(key)
    item_info[key] = {'item_ref': row['item_ref'], 'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id']}
categories = list(df_category['category'].unique())
resources = list(df_capacity['resource'].unique())
bundle_keys = []
bundle_bonus = dict()
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_keys.append((a, b))
    bundle_bonus[a, b] = int(row['bonus_cents'])
incompat_keys = []
for (_, row) in df_incompat.iterrows():
    incompat_keys.append((row['item_a'], row['item_b']))
prereq_keys = []
for (_, row) in df_requires.iterrows():
    prereq_keys.append((row['item_ref'], row['prerequisite_ref']))
benefit_per_itemref = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit_per_item = dict()
for key in item_keys:
    item_ref = key[0]
    benefit_per_item[key] = benefit_per_itemref.get(item_ref, 0)
item_fee_per_itemref = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee_per_item = dict()
for key in item_keys:
    item_ref = key[0]
    item_fee_per_item[key] = item_fee_per_itemref.get(item_ref, 0)
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
usage_per_itemref_resource = dict()
for (_, row) in df_usage.iterrows():
    item_ref = row['item_ref']
    resource = row['resource']
    amount = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit == 'liter':
        amount_ml = amount * 1000
    elif unit == 'ml':
        amount_ml = amount
    else:
        raise ValueError(f'Unknown unit {unit} in usage table')
    usage_per_itemref_resource[item_ref, resource] = amount_ml
usage_per_item_resource = dict()
for key in item_keys:
    (item_ref, location_id) = key
    for resource in resources:
        usage_per_item_resource[key, resource] = usage_per_itemref_resource.get((item_ref, resource), 0)
capacity_per_resource = dict()
for resource in resources:
    df_r = df_capacity[df_capacity['resource'] == resource]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'liter':
            amt_ml = amt * 1000
        elif unit == 'ml':
            amt_ml = amt
        else:
            raise ValueError(f'Unknown unit {unit} in capacity_ledger')
        total += amt_ml
    capacity_per_resource[resource] = total
cat_min = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max = df_category.set_index('category')['maximum_quantity'].to_dict()
item_category = {key: item_info[key]['category'] for key in item_keys}
cat_items = {c: [] for c in categories}
for key in item_keys:
    c = item_category[key]
    cat_items[c].append(key)
itemref_to_keys = dict()
for key in item_keys:
    item_ref = key[0]
    itemref_to_keys.setdefault(item_ref, []).append(key)
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for key in item_keys:
    if item_info[key]['authorized'] == 0:
        m.addConstr(x_vars[key] == 0)
        m.addConstr(y_vars[key] == 0)
    else:
        min_lot = item_info[key]['minimum_lot']
        max_order = item_info[key]['maximum_order']
        m.addConstr(x_vars[key] >= min_lot * y_vars[key])
        m.addConstr(x_vars[key] <= max_order * y_vars[key])
for resource in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource[key, resource] * x_vars[key] for key in item_keys)) <= capacity_per_resource[resource])
for c in categories:
    m.addConstr(gp.quicksum((x_vars[key] for key in cat_items[c])) >= cat_min[c])
    m.addConstr(gp.quicksum((x_vars[key] for key in cat_items[c])) <= cat_max[c])
for c in categories:
    for key in cat_items[c]:
        m.addConstr(y_vars[key] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[key] for key in cat_items[c])))
for key in item_keys:
    pass
for (item_a, item_b) in bundle_keys:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    m.addConstr(b_vars[item_a, item_b] <= gp.quicksum((y_vars[key] for key in keys_a)))
    m.addConstr(b_vars[item_a, item_b] <= gp.quicksum((y_vars[key] for key in keys_b)))
    for key_a in keys_a:
        for key_b in keys_b:
            m.addConstr(b_vars[item_a, item_b] >= y_vars[key_a] + y_vars[key_b] - 1)
for (item_a, item_b) in incompat_keys:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    for key_a in keys_a:
        for key_b in keys_b:
            m.addConstr(y_vars[key_a] + y_vars[key_b] <= 1)
for (item_ref, prereq_ref) in prereq_keys:
    keys_i = itemref_to_keys.get(item_ref, [])
    keys_p = itemref_to_keys.get(prereq_ref, [])
    if not keys_p:
        for key_i in keys_i:
            m.addConstr(y_vars[key_i] == 0)
    else:
        for key_i in keys_i:
            m.addConstr(y_vars[key_i] <= gp.quicksum((y_vars[key_p] for key_p in keys_p)))
obj_benefit = gp.quicksum((benefit_per_item[key] * x_vars[key] for key in item_keys))
obj_item_fee = gp.quicksum((item_fee_per_item[key] * y_vars[key] for key in item_keys))
obj_cat_fee = gp.quicksum((cat_fee[c] * z_vars[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus[bundle] * b_vars[bundle] for bundle in bundle_keys))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()