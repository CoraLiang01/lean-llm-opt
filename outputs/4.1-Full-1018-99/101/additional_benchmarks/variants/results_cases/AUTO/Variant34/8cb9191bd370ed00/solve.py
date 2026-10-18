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
df_item1['authorized'] = df_item1['authorized'].astype(int)
df_item1['minimum_lot'] = df_item1['minimum_lot'].astype(int)
df_item1['maximum_order'] = df_item1['maximum_order'].astype(int)
df_item2['authorized'] = df_item2['authorized'].astype(int)
df_item2['minimum_lot'] = df_item2['minimum_lot'].astype(int)
df_item2['maximum_order'] = df_item2['maximum_order'].astype(int)
df_items = pd.concat([df_item1, df_item2], ignore_index=True)
df_itemfee = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv')
df_itemfee['activation_fee_cents'] = df_itemfee['activation_fee_cents'].astype(int)
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv')
df_usage1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv')
df_usage2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv')
df_usage1['amount'] = df_usage1['amount'].astype(int)
df_usage2['amount'] = df_usage2['amount'].astype(int)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
item_keys = []
item_data = {}
for _, row in df_items.iterrows():
    key = (row['item_ref'], row['location_id'])
    item_keys.append(key)
    item_data[key] = {'item_ref': row['item_ref'], 'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id']}
item_refs_in_items = set((row['item_ref'] for row in df_items.itertuples()))
categories = set(df_category['category'])
resources = set(df_capacity['resource'])
benefit_per_itemref = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit_per_item = {}
for key in item_keys:
    item_ref = key[0]
    benefit_per_item[key] = benefit_per_itemref.get(item_ref, 0)
itemfee_per_itemref = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
itemfee_per_item = {}
for key in item_keys:
    item_ref = key[0]
    itemfee_per_item[key] = itemfee_per_itemref.get(item_ref, 0)
catfee = df_category.set_index('category')['activation_fee_cents'].to_dict()
bundle_keys = []
bundle_bonus = {}
for _, row in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_keys.append((a, b))
    bundle_bonus[a, b] = row['bonus_cents']
capacity_per_resource = {}
for resource in resources:
    df_r = df_capacity[df_capacity['resource'] == resource]
    total = 0
    for _, row in df_r.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'ml':
            total += amt
        elif unit == 'liter':
            total += amt * 1000
        else:
            raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
    capacity_per_resource[resource] = total
usage_per_item_resource = {}
for _, row in df_usage.iterrows():
    item_ref = row['item_ref']
    resource = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit == 'ml':
        amt_ml = amt
    elif unit == 'liter':
        amt_ml = amt * 1000
    else:
        raise ValueError(f'Unknown unit in usage: {unit}')
    usage_per_item_resource[item_ref, resource] = amt_ml
cat_min = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max = df_category.set_index('category')['maximum_quantity'].to_dict()
incompat_pairs = set()
for _, row in df_incompat.iterrows():
    incompat_pairs.add((row['item_a'], row['item_b']))
    incompat_pairs.add((row['item_b'], row['item_a']))
prereq_pairs = []
for _, row in df_requires.iterrows():
    prereq_pairs.append((row['item_ref'], row['prerequisite_ref']))
itemref_to_keys = {}
for key in item_keys:
    item_ref = key[0]
    itemref_to_keys.setdefault(item_ref, []).append(key)
cat_to_itemkeys = {}
for key in item_keys:
    cat = item_data[key]['category']
    cat_to_itemkeys.setdefault(cat, []).append(key)
resource_to_itemkeys = {}
for key in item_keys:
    loc = key[1]
    resource_to_itemkeys.setdefault(loc, []).append(key)
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for key in item_keys:
    auth = item_data[key]['authorized']
    min_lot = item_data[key]['minimum_lot']
    max_order = item_data[key]['maximum_order']
    if auth == 0:
        m.addConstr(x_vars[key] == 0)
        m.addConstr(y_vars[key] == 0)
    else:
        m.addConstr(x_vars[key] >= min_lot * y_vars[key])
        m.addConstr(x_vars[key] <= max_order * y_vars[key])
for resource in resources:
    itemkeys_in_resource = [key for key in item_keys if key[1] == resource]
    if not itemkeys_in_resource:
        continue
    m.addConstr(gp.quicksum((usage_per_item_resource.get((key[0], resource), 0) * x_vars[key] for key in itemkeys_in_resource)) <= capacity_per_resource[resource])
for cat in categories:
    itemkeys_in_cat = cat_to_itemkeys.get(cat, [])
    if not itemkeys_in_cat:
        m.addConstr(z_vars[cat] == 0)
        continue
    m.addConstr(gp.quicksum((x_vars[key] for key in itemkeys_in_cat)) >= cat_min[cat])
    m.addConstr(gp.quicksum((x_vars[key] for key in itemkeys_in_cat)) <= cat_max[cat])
    for key in itemkeys_in_cat:
        m.addConstr(y_vars[key] <= z_vars[cat])
    m.addConstr(z_vars[cat] <= gp.quicksum((y_vars[key] for key in itemkeys_in_cat)))
for item_a, item_b in incompat_pairs:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    for ka in keys_a:
        for kb in keys_b:
            if ka != kb:
                m.addConstr(y_vars[ka] + y_vars[kb] <= 1)
for item_ref, prereq_ref in prereq_pairs:
    keys_item = itemref_to_keys.get(item_ref, [])
    keys_prereq = itemref_to_keys.get(prereq_ref, [])
    if not keys_prereq:
        for ki in keys_item:
            m.addConstr(y_vars[ki] == 0)
    else:
        for ki in keys_item:
            m.addConstr(y_vars[ki] <= gp.quicksum((y_vars[kp] for kp in keys_prereq)))
for item_a, item_b in bundle_keys:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    for ka in keys_a:
        for kb in keys_b:
            if ka != kb:
                m.addConstr(b_vars[item_a, item_b] <= y_vars[ka])
                m.addConstr(b_vars[item_a, item_b] <= y_vars[kb])
    if keys_a and keys_b:
        m.addConstr(b_vars[item_a, item_b] >= gp.quicksum((y_vars[ka] for ka in keys_a)) + gp.quicksum((y_vars[kb] for kb in keys_b)) - 1)
    else:
        m.addConstr(b_vars[item_a, item_b] == 0)
obj_benefit = gp.quicksum((benefit_per_item[key] * x_vars[key] for key in item_keys))
obj_itemfee = gp.quicksum((itemfee_per_item[key] * y_vars[key] for key in item_keys))
obj_catfee = gp.quicksum((catfee[cat] * z_vars[cat] for cat in categories))
obj_bundle = gp.quicksum((bundle_bonus[bkey] * b_vars[bkey] for bkey in bundle_keys))
m.setObjective(obj_benefit - obj_itemfee - obj_catfee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()