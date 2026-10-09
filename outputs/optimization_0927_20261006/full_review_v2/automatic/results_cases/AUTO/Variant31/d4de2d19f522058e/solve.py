import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)

def norm(s):
    return s.strip().casefold()
items = df_item[df_item['table'].str.strip().str.casefold() == 'item']['item_ref'].map(norm).tolist()
item_set = set(items)
categories = df_category[df_category['table'].str.strip().str.casefold() == 'category']['category'].map(norm).tolist()
category_set = set(categories)
resources = pd.concat([df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']['resource'].map(norm), df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']['resource'].map(norm)]).unique().tolist()
resource_set = set(resources)
bundles = []
for (_, row) in df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle'].iterrows():
    a = norm(row['item_a'])
    b = norm(row['item_b'])
    bundles.append((a, b))
bundle_set = set(bundles)
incompat_pairs = []
for (_, row) in df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible'].iterrows():
    a = norm(row['item_a'])
    b = norm(row['item_b'])
    incompat_pairs.append((a, b))
    incompat_pairs.append((b, a))
requires_pairs = []
for (_, row) in df_requires[df_requires['table'].str.strip().str.casefold() == 'requires'].iterrows():
    i = norm(row['item_ref'])
    prereq = norm(row['prerequisite_ref'])
    requires_pairs.append((i, prereq))
item_df = df_item[df_item['table'].str.strip().str.casefold() == 'item'].copy()
item_df['item_ref'] = item_df['item_ref'].map(norm)
item_df['category'] = item_df['category'].map(norm)
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['unit_benefit_cents'] = item_df['unit_benefit_cents'].astype(int)
item_df['item_fee_cents'] = item_df['item_fee_cents'].astype(int)
authorized = dict(zip(item_df['item_ref'], item_df['authorized']))
minimum_lot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
maximum_order = dict(zip(item_df['item_ref'], item_df['maximum_order']))
unit_benefit_cents = dict(zip(item_df['item_ref'], item_df['unit_benefit_cents']))
item_fee_cents = dict(zip(item_df['item_ref'], item_df['item_fee_cents']))
item_category = dict(zip(item_df['item_ref'], item_df['category']))
cat_df = df_category[df_category['table'].str.strip().str.casefold() == 'category'].copy()
cat_df['category'] = cat_df['category'].map(norm)
cat_df['minimum_quantity'] = cat_df['minimum_quantity'].astype(int)
cat_df['maximum_quantity'] = cat_df['maximum_quantity'].astype(int)
cat_df['activation_fee_cents'] = cat_df['activation_fee_cents'].astype(int)
minimum_quantity = dict(zip(cat_df['category'], cat_df['minimum_quantity']))
maximum_quantity = dict(zip(cat_df['category'], cat_df['maximum_quantity']))
activation_fee_cents = dict(zip(cat_df['category'], cat_df['activation_fee_cents']))
bundle_bonus_cents = {}
for (_, row) in df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle'].iterrows():
    a = norm(row['item_a'])
    b = norm(row['item_b'])
    bundle_bonus_cents[a, b] = int(row['bonus_cents'])
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage'].copy()
usage_df['item_ref'] = usage_df['item_ref'].map(norm)
usage_df['resource'] = usage_df['resource'].map(norm)
usage_df['amount'] = usage_df['amount'].astype(int)
usage = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    usage[i, r] = row['amount']
cap_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger'].copy()
cap_df['resource'] = cap_df['resource'].map(norm)
cap_df['entry'] = cap_df['entry'].map(norm)
cap_df['amount'] = cap_df['amount'].astype(int)
resource_capacity = {}
for r in resource_set:
    total = cap_df[cap_df['resource'] == r]['amount'].sum()
    resource_capacity[r] = total
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    if authorized[i] == 1:
        m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i], name=f'maxorder_{i}')
    else:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
for i in items:
    m.addConstr(x_vars[i] >= y_vars[i], name=f'link1_{i}')
for r in resource_set:
    m.addConstr(gp.quicksum((usage.get((i, r), 0) * x_vars[i] for i in items)) <= resource_capacity[r], name=f'resource_{r}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= minimum_quantity[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= maximum_quantity[c], name=f'cat_max_{c}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c], name=f'cat_link_{i}_{c}')
for (i, j) in incompat_pairs:
    if i in item_set and j in item_set:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in requires_pairs:
    if i in item_set and prereq in item_set:
        m.addConstr(y_vars[i] <= y_vars[prereq], name=f'requires_{i}_{prereq}')
for (a, b) in bundles:
    if a in item_set and b in item_set:
        m.addConstr(b_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(b_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle0_{a}_{b}')
obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in items)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in items)) - gp.quicksum((activation_fee_cents[c] * z_vars[c] for c in categories)) + gp.quicksum((bundle_bonus_cents[a, b] * b_vars[a, b] for (a, b) in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()