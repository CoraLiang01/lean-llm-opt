import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
benefit_df = dfs[0]
benefit_df = benefit_df[benefit_df['table'].str.strip().str.casefold() == 'benefit']
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
bundle_df = dfs[1]
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_tuples = list(bundle_df[['item_a', 'item_b']].itertuples(index=False, name=None))
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
capacity_df = dfs[2]
capacity_df = capacity_df[capacity_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_df['amount'] = capacity_df['amount'].astype(int)
resource_capacity = capacity_df.groupby('resource')['amount'].sum().to_dict()
resources = sorted(resource_capacity.keys())
category_df = dfs[3]
category_df = category_df[category_df['table'].str.strip().str.casefold() == 'category']
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = category_df['category'].tolist()
category_min = category_df.set_index('category')['minimum_quantity'].to_dict()
category_max = category_df.set_index('category')['maximum_quantity'].to_dict()
category_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
incompat_df = dfs[5]
incompat_df = incompat_df[incompat_df['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = list(incompat_df[['item_a', 'item_b']].itertuples(index=False, name=None))
item_dfs = []
for idx in [6, 7]:
    df = dfs[idx]
    df = df[df['table'].str.strip().str.casefold() == 'item']
    df['authorized'] = df['authorized'].astype(int)
    df['minimum_lot'] = df['minimum_lot'].astype(int)
    df['maximum_order'] = df['maximum_order'].astype(int)
    item_dfs.append(df)
item_df = pd.concat(item_dfs, ignore_index=True)
item_df = item_df.drop_duplicates(subset=['item_ref'])
item_ref_set = set(item_df['item_ref'])
item_fee_df = dfs[8]
item_fee_df = item_fee_df[item_fee_df['table'].str.strip().str.casefold() == 'item_fee']
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
requires_df = dfs[10]
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
requires_pairs = list(requires_df[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
usage_dfs = []
for idx in [11, 12]:
    df = dfs[idx]
    df = df[df['table'].str.strip().str.casefold() == 'usage']
    df['amount'] = df['amount'].astype(int)
    usage_dfs.append(df)
usage_df = pd.concat(usage_dfs, ignore_index=True)
usage_dict = {}
for (_, row) in usage_df.iterrows():
    usage_dict[row['item_ref'], row['resource']] = row['amount']
items = sorted(item_df['item_ref'].tolist())
item_category = item_df.set_index('item_ref')['category'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_benefit_full = {i: item_benefit.get(i, 0) for i in items}
item_fee_full = {i: item_fee.get(i, 0) for i in items}
item_resource_usage = {(i, r): usage_dict.get((i, r), 0) for i in items for r in resources}
category_items = {g: [] for g in categories}
for i in items:
    g = item_category[i]
    if g in category_items:
        category_items[g].append(i)
    else:
        continue
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in items:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth == 1:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i])
        m.addConstr(x_vars[i] <= max_order * y_vars[i])
        m.addConstr(x_vars[i] >= 0)
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for g in categories:
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) >= category_min[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) <= category_max[g])
for g in categories:
    for i in category_items[g]:
        m.addConstr(z_vars[g] >= y_vars[i])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in category_items[g])))
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * x_vars[i] for i in items)) <= resource_capacity[r])
for (i, j) in incompat_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, pr) in requires_pairs:
    if i in items and pr in items:
        m.addConstr(y_vars[i] <= y_vars[pr])
for (i, j) in bundle_tuples:
    if i in items and j in items:
        m.addConstr(w_vars[i, j] <= y_vars[i])
        m.addConstr(w_vars[i, j] <= y_vars[j])
        m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(w_vars[i, j] == 0)
objective = gp.quicksum((item_benefit_full[i] * x_vars[i] for i in items)) - gp.quicksum((item_fee_full[i] * y_vars[i] for i in items)) - gp.quicksum((category_fee[g] * z_vars[g] for g in categories)) + gp.quicksum((bundle_bonus[i, j] * w_vars[i, j] for (i, j) in bundle_tuples))
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.optimize()