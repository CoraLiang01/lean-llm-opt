import gurobipy as gp
import pandas as pd
import numpy as np
import re
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
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
item_tables = [item1_df, item2_df]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['authorized'] = item_rows['authorized'].astype(int)
authorized_items_df = item_rows[item_rows['authorized'] == 1].copy()
item_refs = authorized_items_df['item_ref'].unique().tolist()
item_category = dict(zip(authorized_items_df['item_ref'], authorized_items_df['category']))
item_min_lot = dict(zip(authorized_items_df['item_ref'], authorized_items_df['minimum_lot'].astype(int)))
item_max_order = dict(zip(authorized_items_df['item_ref'], authorized_items_df['maximum_order'].astype(int)))
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_benefit = {i: item_benefit.get(i, 0) for i in item_refs}
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_setup_fee = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
item_setup_fee = {i: item_setup_fee.get(i, 0) for i in item_refs}
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = category_df['category'].unique().tolist()
category_min_qty = dict(zip(category_df['category'], category_df['minimum_quantity']))
category_max_qty = dict(zip(category_df['category'], category_df['maximum_quantity']))
category_activation_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
usage_df = pd.concat([usage1_df, usage2_df], ignore_index=True)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df = usage_df[usage_df['item_ref'].isin(item_refs)]
resources = usage_df['resource'].unique().tolist()
item_resource_usage = {(row['item_ref'], row['resource']): row['amount'] for (_, row) in usage_df.iterrows()}
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
resource_capacity = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
incompatible_pairs = []
for (_, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        incompatible_pairs.append((a, b))
requires_pairs = []
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    if i in item_refs and prereq in item_refs:
        requires_pairs.append((i, prereq))
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_tuples = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
category_items = {g: [] for g in categories}
for i in item_refs:
    g = item_category[i]
    category_items[g].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * x_vars[i] for i in item_refs)) <= resource_capacity[r], name=f'res_{r}')
for i in item_refs:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'xmin_{i}')
    m.addConstr(x_vars[i] <= max_order * y_vars[i], name=f'xmax_{i}')
for g in categories:
    items_in_g = category_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= category_min_qty[g], name=f'catmin_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= category_max_qty[g], name=f'catmax_{g}')
for g in categories:
    items_in_g = category_items[g]
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'catlink_{g}_{i}')
for (i, j) in incompatible_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, prereq) in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[prereq], name=f'req_{i}_{prereq}')
for (i, j) in bundle_tuples:
    m.addConstr(w_vars[i, j] <= y_vars[i], name=f'bundle1_{i}_{j}')
    m.addConstr(w_vars[i, j] <= y_vars[j], name=f'bundle2_{i}_{j}')
    m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{i}_{j}')
benefit_term = gp.quicksum((item_benefit[i] * x_vars[i] for i in item_refs))
item_fee_term = gp.quicksum((item_setup_fee[i] * y_vars[i] for i in item_refs))
cat_fee_term = gp.quicksum((category_activation_fee[g] * z_vars[g] for g in categories))
bundle_term = gp.quicksum((bundle_bonus[i, j] * w_vars[i, j] for (i, j) in bundle_tuples))
m.setObjective(benefit_term - item_fee_term - cat_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()