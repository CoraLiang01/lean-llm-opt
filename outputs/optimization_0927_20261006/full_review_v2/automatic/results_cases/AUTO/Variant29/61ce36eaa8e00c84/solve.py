import gurobipy as gp
import pandas as pd
import numpy as np
import re
UNIT_CONV = {'ml': 1, 'liter': 1000, 'minute': 1, 'hour': 60, 'wh': 1, 'kwh': 1000}
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
items_df = df_item[df_item['authorized'] == 1].copy()
item_ids = items_df['item_ref'].tolist()
item_set = set(item_ids)
categories_df = df_category.copy()
category_ids = categories_df['category'].tolist()
category_set = set(category_ids)
item_to_category = dict(zip(items_df['item_ref'], items_df['category']))
usage_resources = set(df_usage['resource'].str.strip().str.casefold())
capacity_resources = set(df_capacity['resource'].str.strip().str.casefold())
resource_set = usage_resources.union(capacity_resources)
items_df['minimum_lot'] = items_df['minimum_lot'].astype(int)
items_df['maximum_order'] = items_df['maximum_order'].astype(int)
items_df['unit_benefit_cents'] = items_df['unit_benefit_cents'].astype(int)
items_df['item_fee_cents'] = items_df['item_fee_cents'].astype(int)
minimum_lot = dict(zip(items_df['item_ref'], items_df['minimum_lot']))
maximum_order = dict(zip(items_df['item_ref'], items_df['maximum_order']))
unit_benefit_cents = dict(zip(items_df['item_ref'], items_df['unit_benefit_cents']))
item_fee_cents = dict(zip(items_df['item_ref'], items_df['item_fee_cents']))
categories_df['minimum_quantity'] = categories_df['minimum_quantity'].astype(int)
categories_df['maximum_quantity'] = categories_df['maximum_quantity'].astype(int)
categories_df['activation_fee_cents'] = categories_df['activation_fee_cents'].astype(int)
minimum_quantity = dict(zip(categories_df['category'], categories_df['minimum_quantity']))
maximum_quantity = dict(zip(categories_df['category'], categories_df['maximum_quantity']))
activation_fee_cents = dict(zip(categories_df['category'], categories_df['activation_fee_cents']))
usage_df = df_usage[df_usage['item_ref'].isin(item_set)].copy()
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['resource'] = usage_df['resource'].str.strip().str.casefold()
usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()
item_resource_usage = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    unit = row['unit']
    if unit not in UNIT_CONV:
        raise ValueError(f'Unknown unit {unit} for resource {r}')
    amt_base = amt * UNIT_CONV[unit]
    item_resource_usage[i, r] = amt_base
df_capacity['amount'] = df_capacity['amount'].astype(int)
df_capacity['resource'] = df_capacity['resource'].str.strip().str.casefold()
df_capacity['unit'] = df_capacity['unit'].str.strip().str.casefold()
resource_capacity = {}
for r in resource_set:
    df_r = df_capacity[df_capacity['resource'] == r]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = row['amount']
        unit = row['unit']
        if unit not in UNIT_CONV:
            raise ValueError(f'Unknown unit {unit} for resource {r}')
        amt_base = amt * UNIT_CONV[unit]
        total += amt_base
    resource_capacity[r] = total
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in item_set and p in item_set:
        prereq_pairs.append((i, p))
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        bundle_pairs.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
m = gp.Model('BakeryOrder')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
for c in category_ids:
    items_in_c = [i for i in item_ids if item_to_category[i] == c]
    for i in items_in_c:
        m.addConstr(z_vars[c] >= y_vars[i])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for c in category_ids:
    items_in_c = [i for i in item_ids if item_to_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= maximum_quantity[c])
for r in resource_set:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * x_vars[i] for i in item_ids)) <= resource_capacity[r])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, p) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[p])
for (i, j) in bundle_pairs:
    m.addConstr(b_vars[i, j] <= y_vars[i])
    m.addConstr(b_vars[i, j] <= y_vars[j])
    m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
objective = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in item_ids)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in item_ids)) - gp.quicksum((activation_fee_cents[c] * z_vars[c] for c in category_ids)) + gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.optimize()