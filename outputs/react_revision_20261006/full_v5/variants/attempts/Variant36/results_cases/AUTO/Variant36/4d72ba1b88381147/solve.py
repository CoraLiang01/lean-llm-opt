import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', sep=',')
df_incompatible = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', sep=',')
df_item_6 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', sep=',')
df_item_7 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', sep=',')
df_usage_10 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', sep=',')
df_usage_11 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv', sep=',')
df_items = pd.concat([df_item_6, df_item_7], ignore_index=True)
df_items = df_items.drop_duplicates(subset=['item_ref'])
df_items['authorized'] = df_items['authorized'].astype(int)
items = df_items[df_items['authorized'] == 1]['item_ref'].tolist()
item_min_lot = df_items.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = df_items.set_index('item_ref')['maximum_order'].to_dict()
item_category = df_items.set_index('item_ref')['category'].to_dict()
item_location = df_items.set_index('item_ref')['location_id'].to_dict()
item_unit_benefit = df_items.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = df_items.set_index('item_ref')['item_fee_cents'].to_dict()
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
categories = df_category['category'].tolist()
cat_min_qty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_activation_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_capacity['amount'] = df_capacity['amount'].astype(int)
resource_capacity = df_capacity.groupby('resource')['amount'].sum().to_dict()
resources = list(resource_capacity.keys())
df_usage = pd.concat([df_usage_10, df_usage_11], ignore_index=True)
df_usage['amount'] = df_usage['amount'].astype(int)
df_usage = df_usage[df_usage['item_ref'].isin(items)]
usage = {}
for (_, row) in df_usage.iterrows():
    usage[row['resource'], row['item_ref']] = row['amount']
incompat_pairs = []
for (_, row) in df_incompatible.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        incompat_pairs.append((a, b))
requires_pairs = []
for (_, row) in df_requires.iterrows():
    (i, j) = (row['item_ref'], row['prerequisite_ref'])
    if i in items and j in items:
        requires_pairs.append((i, j))
bundles = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        bundles.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
cat_items = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    if c in categories:
        cat_items[c].append(i)
resource_items = {r: [] for r in resources}
for i in items:
    loc = item_location[i]
    if loc in resources:
        resource_items[loc].append(i)
m = gp.Model('FC_EAST_HVAC_Placement')
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, ub=[item_max_order[i] for i in items], name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(x[i] >= min_lot * y[i], name='')
    m.addConstr(x[i] <= max_order * y[i], name='')
for r in resources:
    expr = gp.LinExpr()
    for i in resource_items[r]:
        if (r, i) in usage:
            expr += usage[r, i] * x[i]
        else:
            continue
    m.addConstr(expr <= resource_capacity[r], name='')
for c in categories:
    items_in_c = cat_items[c]
    min_qty = cat_min_qty[c]
    max_qty = cat_max_qty[c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= min_qty * z[c], name='')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= max_qty * z[c], name='')
    for i in items_in_c:
        m.addConstr(y[i] <= z[c], name='')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name='')
for (i, j) in requires_pairs:
    m.addConstr(y[i] <= y[j], name='')
for (i, j) in bundles:
    m.addConstr(b[i, j] <= y[i], name='')
    m.addConstr(b[i, j] <= y[j], name='')
    m.addConstr(b[i, j] >= y[i] + y[j] - 1, name='')
obj = gp.LinExpr()
obj += gp.quicksum((item_unit_benefit[i] * x[i] for i in items))
obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
obj -= gp.quicksum((cat_activation_fee[c] * z[c] for c in categories))
if bundles:
    obj += gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()