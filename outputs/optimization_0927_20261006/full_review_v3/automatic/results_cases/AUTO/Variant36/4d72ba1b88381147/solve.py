import gurobipy as gp
import pandas as pd
import numpy as np
import re
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv']
dfs = [pd.read_csv(p, sep=',', dtype=str, keep_default_na=False) for p in paths]
bundle_df = dfs[0]
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_tuples = list(bundle_df[['item_a', 'item_b']].itertuples(index=False, name=None))
bundle_bonus = {(row.item_a, row.item_b): row.bonus_cents for row in bundle_df.itertuples(index=False)}
capacity_df = dfs[1]
capacity_df = capacity_df[capacity_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_df['amount'] = capacity_df['amount'].astype(int)
capacity_by_resource = capacity_df.groupby('resource')['amount'].sum().to_dict()
resources = set(capacity_by_resource.keys())
category_df = dfs[2]
category_df = category_df[category_df['table'].str.strip().str.casefold() == 'category']
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = set(category_df['category'])
cat_min_qty = category_df.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = category_df.set_index('category')['maximum_quantity'].to_dict()
cat_activation_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
identity_df = dfs[3]
identity_df = identity_df[identity_df['table'].str.strip().str.casefold() == 'identity']
ref_to_category = {}
incompat_df = dfs[4]
incompat_df = incompat_df[incompat_df['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = set()
for row in incompat_df.itertuples(index=False):
    incompat_pairs.add((row.item_a, row.item_b))
    incompat_pairs.add((row.item_b, row.item_a))
item_dfs = []
for idx in [5, 6]:
    df = dfs[idx]
    df = df[df['table'].str.strip().str.casefold() == 'item']
    df['authorized'] = df['authorized'].astype(int)
    df['minimum_lot'] = df['minimum_lot'].astype(int)
    df['maximum_order'] = df['maximum_order'].astype(int)
    df['unit_benefit_cents'] = df['unit_benefit_cents'].astype(int)
    df['item_fee_cents'] = df['item_fee_cents'].astype(int)
    item_dfs.append(df)
item_df = pd.concat(item_dfs, ignore_index=True)
item_df = item_df[item_df['authorized'] == 1].copy()
items = set(item_df['item_ref'])
item_to_category = item_df.set_index('item_ref')['category'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_unit_benefit = item_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = item_df.set_index('item_ref')['item_fee_cents'].to_dict()
item_location = item_df.set_index('item_ref')['location_id'].to_dict()
requires_df = dfs[8]
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
requires_pairs = set()
for row in requires_df.itertuples(index=False):
    requires_pairs.add((row.item_ref, row.prerequisite_ref))
usage_dfs = []
for idx in [9, 10]:
    df = dfs[idx]
    df = df[df['table'].str.strip().str.casefold() == 'usage']
    df['amount'] = df['amount'].astype(int)
    usage_dfs.append(df)
usage_df = pd.concat(usage_dfs, ignore_index=True)
usage_df = usage_df[usage_df['item_ref'].isin(items)]
usage_dict = {}
for row in usage_df.itertuples(index=False):
    usage_dict[row.item_ref, row.resource] = row.amount
item_resources = set(usage_df['resource'])
m = gp.Model('FC_EAST_HVAC_Placement')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
obj = gp.quicksum((item_unit_benefit[i] * x_vars[i] - item_fee[i] * y_vars[i] for i in items))
obj += gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_tuples))
obj -= gp.quicksum((cat_activation_fee[c] * z_vars[c] for c in categories))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for r in resources:
    relevant_items = [i for i in items if (i, r) in usage_dict]
    m.addConstr(gp.quicksum((usage_dict[i, r] * x_vars[i] for i in relevant_items)) <= capacity_by_resource[r], name=f'capacity_{r}')
for i in items:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'minlot_{i}')
    m.addConstr(x_vars[i] <= max_order * y_vars[i], name=f'maxorder_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
for c in categories:
    items_in_c = [i for i in items if item_to_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= cat_min_qty[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= cat_max_qty[c], name=f'cat_max_{c}')
for c in categories:
    items_in_c = [i for i in items if item_to_category[i] == c]
    for i in items_in_c:
        m.addConstr(z_vars[c] >= y_vars[i], name=f'catact_{c}_{i}')
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)), name=f'catactsum_{c}')
for (i, j) in incompat_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] <= y_vars[j], name=f'requires_{i}_{j}')
for (a, b) in bundle_tuples:
    if a in items and b in items:
        m.addConstr(b_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(b_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle_forbidden_{a}_{b}')
m.optimize()