import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
benefit_df = dfs[0]
bundle_df = dfs[1]
capacity_ledger_df = dfs[2]
category_df = dfs[3]
identity_df = dfs[4]
incompatible_df = dfs[5]
item1_df = dfs[6]
item2_df = dfs[7]
item_fee_df = dfs[8]
requires_df = dfs[10]
usage1_df = dfs[11]
usage2_df = dfs[12]
item_df = pd.concat([item1_df, item2_df], ignore_index=True)
item_df['authorized'] = item_df['authorized'].astype(int)
authorized_items_df = item_df[item_df['authorized'] == 1].copy()
item_refs = authorized_items_df['item_ref'].unique().tolist()
item_category = authorized_items_df.set_index('item_ref')['category'].to_dict()
item_min_lot = authorized_items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_max_order = authorized_items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
benefit_df = benefit_df[benefit_df['table'].str.strip().str.casefold() == 'benefit']
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_benefit = {i: item_benefit.get(i, 0) for i in item_refs}
item_fee_df = item_fee_df[item_fee_df['table'].str.strip().str.casefold() == 'item_fee']
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_activation_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
item_activation_fee = {i: item_activation_fee.get(i, 0) for i in item_refs}
category_df = category_df[category_df['table'].str.strip().str.casefold() == 'category']
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = category_df['category'].unique().tolist()
category_min_qty = category_df.set_index('category')['minimum_quantity'].to_dict()
category_max_qty = category_df.set_index('category')['maximum_quantity'].to_dict()
category_activation_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
usage_df = pd.concat([usage1_df, usage2_df], ignore_index=True)
usage_df = usage_df[usage_df['table'].str.strip().str.casefold() == 'usage']
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df = usage_df[usage_df['item_ref'].isin(item_refs)]
resources = usage_df['resource'].unique().tolist()
item_resource_usage = {i: {r: 0 for r in resources} for i in item_refs}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    item_resource_usage[i][r] = int(amt)
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
resource_capacity = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
resource_capacity = {r: resource_capacity.get(r, 0) for r in resources}
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    i = row['item_a']
    j = row['item_b']
    if i in item_refs and j in item_refs:
        bundle_pairs.append((i, j))
        bundle_bonus[i, j] = row['bonus_cents']
incompatible_df = incompatible_df[incompatible_df['table'].str.strip().str.casefold() == 'incompatible']
incompatible_pairs = []
for (_, row) in incompatible_df.iterrows():
    i = row['item_a']
    j = row['item_b']
    if i in item_refs and j in item_refs:
        incompatible_pairs.append((i, j))
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
requires_pairs = []
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    j = row['prerequisite_ref']
    if i in item_refs and j in item_refs:
        requires_pairs.append((i, j))
category_items = {g: [] for g in categories}
for i in item_refs:
    g = item_category[i]
    if g in categories:
        category_items[g].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    m.addConstr(x_vars[i] >= item_min_lot[i] * y_vars[i])
    m.addConstr(x_vars[i] <= item_max_order[i] * y_vars[i])
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i][r] * x_vars[i] for i in item_refs)) <= resource_capacity[r])
for g in categories:
    items_in_g = category_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= category_min_qty[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= category_max_qty[g])
for g in categories:
    items_in_g = category_items[g]
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for (i, j) in incompatible_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[j])
for (i, j) in bundle_pairs:
    m.addConstr(b_vars[i, j] <= y_vars[i])
    m.addConstr(b_vars[i, j] <= y_vars[j])
    m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
obj = gp.quicksum((item_benefit[i] * x_vars[i] for i in item_refs)) - gp.quicksum((item_activation_fee[i] * y_vars[i] for i in item_refs)) - gp.quicksum((category_activation_fee[g] * z_vars[g] for g in categories)) + gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()