import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
(df_benefit, df_bundle, df_capacity_ledger, df_category, df_identity, df_incompatible, df_item1, df_item2, df_item_fee, df_market, df_requires, df_usage1, df_usage2) = dfs
df_items = pd.concat([df_item1, df_item2], ignore_index=True)
df_items['authorized'] = df_items['authorized'].astype(int)
df_items['minimum_lot'] = df_items['minimum_lot'].astype(int)
df_items['maximum_order'] = df_items['maximum_order'].astype(int)
authorized_items = df_items[df_items['authorized'] == 1].copy()
item_refs = authorized_items['item_ref'].tolist()
item_to_category = authorized_items.set_index('item_ref')['category'].to_dict()
item_min_lot = authorized_items.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = authorized_items.set_index('item_ref')['maximum_order'].to_dict()
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
benefit_by_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit_by_item = {i: benefit_by_item.get(i, 0) for i in item_refs}
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
item_fee = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee = {i: item_fee.get(i, 0) for i in item_refs}
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
categories = df_category['category'].tolist()
cat_min_qty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_usage1['amount'] = df_usage1['amount'].astype(int)
df_usage2['amount'] = df_usage2['amount'].astype(int)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
resources = sorted(df_usage['resource'].unique())
usage_by_item_resource = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    if i in item_refs:
        usage_by_item_resource[i, r] = amt
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
resource_capacity = {r: resource_capacity[r] for r in resources if r in resource_capacity}
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        bundle_pairs.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
incompat_pairs = []
for (_, row) in df_incompatible.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        incompat_pairs.append((a, b))
requires_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    j = row['prerequisite_ref']
    if i in item_refs and j in item_refs:
        requires_pairs.append((i, j))
cat_to_items = {g: [] for g in categories}
for i in item_refs:
    g = item_to_category[i]
    cat_to_items[g].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i])
    m.addConstr(x_vars[i] <= max_order * y_vars[i])
for r in resources:
    m.addConstr(gp.quicksum((usage_by_item_resource.get((i, r), 0) * x_vars[i] for i in item_refs)) <= resource_capacity[r])
for g in categories:
    items_in_g = cat_to_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_min_qty[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_max_qty[g])
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[j])
for (a, b) in bundle_pairs:
    m.addConstr(w_vars[a, b] <= y_vars[a])
    m.addConstr(w_vars[a, b] <= y_vars[b])
    m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
obj = gp.quicksum((benefit_by_item[i] * x_vars[i] for i in item_refs)) - gp.quicksum((item_fee[i] * y_vars[i] for i in item_refs)) - gp.quicksum((cat_fee[g] * z_vars[g] for g in categories)) + gp.quicksum((bundle_bonus[a, b] * w_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()