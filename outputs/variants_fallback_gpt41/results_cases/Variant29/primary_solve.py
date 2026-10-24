import gurobipy as gp
import pandas as pd
import numpy as np
import re
f_requires = 'benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = 'benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = 'benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = 'benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = 'benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompatible = 'benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = 'benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = 'benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = 'benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
item_df = pd.read_csv(f_item, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(f_usage, dtype=str, keep_default_na=False)
category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
incompatible_df = pd.read_csv(f_incompatible, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(f_capacity, dtype=str, keep_default_na=False)
item_df['authorized'] = item_df['authorized'].astype(int)
authorized_items_df = item_df[item_df['authorized'] == 1].copy()
authorized_items = list(authorized_items_df['item_ref'])
categories = list(category_df['category'])
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
item_to_category = authorized_items_df.set_index('item_ref')['category'].to_dict()
category_to_items = {c: authorized_items_df[authorized_items_df['category'] == c]['item_ref'].tolist() for c in categories}
authorized_items_df['minimum_lot'] = authorized_items_df['minimum_lot'].astype(int)
authorized_items_df['maximum_order'] = authorized_items_df['maximum_order'].astype(int)
authorized_items_df['unit_benefit_cents'] = authorized_items_df['unit_benefit_cents'].astype(int)
authorized_items_df['item_fee_cents'] = authorized_items_df['item_fee_cents'].astype(int)
minimum_lot = authorized_items_df.set_index('item_ref')['minimum_lot'].to_dict()
maximum_order = authorized_items_df.set_index('item_ref')['maximum_order'].to_dict()
unit_benefit_cents = authorized_items_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = authorized_items_df.set_index('item_ref')['item_fee_cents'].to_dict()
unit_to_base = {'liter': 1000, 'ml': 1, 'hour': 60, 'minute': 1, 'kwh': 1000, 'wh': 1}
resource_set = set(usage_df['resource'].unique()).union(capacity_df['resource'].unique())
usage_df = usage_df[usage_df['item_ref'].isin(authorized_items)].copy()
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['amount_base'] = usage_df.apply(lambda row: row['amount'] * unit_to_base[row['unit'].strip().casefold()], axis=1)
item_resource_usage = {}
for (_, row) in usage_df.iterrows():
    item_resource_usage[row['item_ref'], row['resource']] = row['amount_base']
capacity_df['amount'] = capacity_df['amount'].astype(int)
capacity_df['amount_base'] = capacity_df.apply(lambda row: row['amount'] * unit_to_base[row['unit'].strip().casefold()], axis=1)
resource_capacity = {}
for resource in resource_set:
    resource_capacity[resource] = capacity_df[capacity_df['resource'] == resource]['amount_base'].sum()
incompat_pairs = []
for (_, row) in incompatible_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
requires_pairs = []
for (_, row) in requires_df.iterrows():
    (i, prereq) = (row['item_ref'], row['prerequisite_ref'])
    if i in authorized_items and prereq in authorized_items:
        requires_pairs.append((i, prereq))
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in authorized_items and b in authorized_items:
        key = (a, b)
        bundle_pairs.append(key)
        bundle_bonus[key] = row['bonus_cents']
m = gp.Model('BakeryOrder')
q_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(q_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(q_vars[i] <= maximum_order[i] * y_vars[i])
    m.addConstr(q_vars[i] >= 0)
    m.addConstr(q_vars[i] <= maximum_order[i])
for c in categories:
    for i in category_to_items[c]:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in category_to_items[c])))
for c in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in category_to_items[c])) >= category_df.loc[category_df['category'] == c, 'minimum_quantity'].iloc[0])
    m.addConstr(gp.quicksum((q_vars[i] for i in category_to_items[c])) <= category_df.loc[category_df['category'] == c, 'maximum_quantity'].iloc[0])
for resource in resource_set:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, resource), 0) * q_vars[i] for i in authorized_items)) <= resource_capacity[resource])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, prereq) in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[prereq])
for (i_a, i_b) in bundle_pairs:
    m.addConstr(b_vars[i_a, i_b] <= y_vars[i_a])
    m.addConstr(b_vars[i_a, i_b] <= y_vars[i_b])
    m.addConstr(b_vars[i_a, i_b] >= y_vars[i_a] + y_vars[i_b] - 1)
obj = gp.quicksum((unit_benefit_cents[i] * q_vars[i] - item_fee_cents[i] * y_vars[i] for i in authorized_items)) - gp.quicksum((category_df.loc[category_df['category'] == c, 'activation_fee_cents'].iloc[0] * z_vars[c] for c in categories)) + gp.quicksum((bundle_bonus[p] * b_vars[p] for p in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()