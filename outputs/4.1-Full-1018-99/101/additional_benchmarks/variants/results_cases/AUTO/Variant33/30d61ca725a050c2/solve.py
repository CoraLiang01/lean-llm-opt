import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
benefit_df = dfs[0]
bundle_df = dfs[1]
capacity_ledger_df = dfs[2]
category_df = dfs[3]
identity_df = dfs[4]
incompatible_df = dfs[5]
item_df1 = dfs[6]
item_df2 = dfs[7]
item_df = pd.concat([item_df1, item_df2], ignore_index=True)
item_fee_df = dfs[8]
requires_df = dfs[10]
usage_df1 = dfs[11]
usage_df2 = dfs[12]
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
item_df['item_ref'] = item_df['item_ref'].astype(str)
item_refs = item_df['item_ref'].unique().tolist()
category_df['category'] = category_df['category'].astype(str)
categories = category_df['category'].unique().tolist()
capacity_ledger_df['resource'] = capacity_ledger_df['resource'].astype(str)
usage_df['resource'] = usage_df['resource'].astype(str)
resources = sorted(set(capacity_ledger_df['resource'].unique()).union(usage_df['resource'].unique()))
bundle_df['item_a'] = bundle_df['item_a'].astype(str)
bundle_df['item_b'] = bundle_df['item_b'].astype(str)
bundles = list(bundle_df[['item_a', 'item_b']].itertuples(index=False, name=None))
incompatible_df['item_a'] = incompatible_df['item_a'].astype(str)
incompatible_df['item_b'] = incompatible_df['item_b'].astype(str)
incompatible_pairs = list(incompatible_df[['item_a', 'item_b']].itertuples(index=False, name=None))
requires_df['item_ref'] = requires_df['item_ref'].astype(str)
requires_df['prerequisite_ref'] = requires_df['prerequisite_ref'].astype(str)
requires_pairs = list(requires_df[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
benefit_df['item_ref'] = benefit_df['item_ref'].astype(str)
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_fee_df['item_ref'] = item_fee_df['item_ref'].astype(str)
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_activation_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for _, row in bundle_df.iterrows()}
usage_df['item_ref'] = usage_df['item_ref'].astype(str)
usage_df['resource'] = usage_df['resource'].astype(str)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_per_item_resource = {}
for _, row in usage_df.iterrows():
    key = (row['item_ref'], row['resource'])
    usage_per_item_resource[key] = row['amount']
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
resource_capacity = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_min = category_df.set_index('category')['minimum_quantity'].to_dict()
category_max = category_df.set_index('category')['maximum_quantity'].to_dict()
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['category'] = item_df['category'].astype(str)
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_category = item_df.set_index('item_ref')['category'].to_dict()
category_items = {g: [] for g in categories}
for i in item_refs:
    g = item_category.get(i)
    if g in category_items:
        category_items[g].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_authorized.get(i, 0)
    min_lot = item_min_lot.get(i, 0)
    max_order = item_max_order.get(i, 0)
    if auth == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i])
        m.addConstr(x_vars[i] <= max_order * y_vars[i])
        m.addConstr(x_vars[i] >= 0)
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource.get((i, r), 0) * x_vars[i] for i in item_refs)) <= resource_capacity.get(r, 0))
for g in categories:
    items_in_g = category_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= category_min[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= category_max[g])
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for i, j in incompatible_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for i, j in requires_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] <= y_vars[j])
for i, j in bundles:
    if i in y_vars and j in y_vars:
        m.addConstr(w_vars[i, j] <= y_vars[i])
        m.addConstr(w_vars[i, j] <= y_vars[j])
        m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(w_vars[i, j] == 0)
obj = gp.quicksum((benefit_per_item.get(i, 0) * x_vars[i] for i in item_refs)) - gp.quicksum((item_fee.get(i, 0) * y_vars[i] for i in item_refs)) - gp.quicksum((category_activation_fee.get(g, 0) * z_vars[g] for g in categories)) + gp.quicksum((bundle_bonus.get((i, j), 0) * w_vars[i, j] for i, j in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()