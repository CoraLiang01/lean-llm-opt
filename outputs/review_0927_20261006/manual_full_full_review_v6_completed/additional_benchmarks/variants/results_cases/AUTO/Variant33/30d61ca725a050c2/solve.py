import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
df_benefit = dfs[0]
df_bundle = dfs[1]
df_capacity_ledger = dfs[2]
df_category = dfs[3]
df_identity = dfs[4]
df_incompatible = dfs[5]
df_item_1 = dfs[6]
df_item_2 = dfs[7]
df_item_fee = dfs[8]
df_requires = dfs[10]
df_usage_1 = dfs[11]
df_usage_2 = dfs[12]
item_tables = [df_item_1, df_item_2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_ref_set = set(item_rows['item_ref'])
items = sorted(item_ref_set)
item_to_category = item_rows.set_index('item_ref')['category'].to_dict()
item_to_authorized = item_rows.set_index('item_ref')['authorized'].to_dict()
item_to_minlot = item_rows.set_index('item_ref')['minimum_lot'].to_dict()
item_to_maxorder = item_rows.set_index('item_ref')['maximum_order'].to_dict()
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
categories = sorted(df_category['category'].unique())
cat_to_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_to_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_to_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
resources = sorted(df_capacity_ledger['resource'].unique())
resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
item_benefit = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in item_benefit:
        item_benefit[i] = 0
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
item_fee = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in items:
    if i not in item_fee:
        item_fee[i] = 0
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = list(df_bundle[['item_a', 'item_b']].itertuples(index=False, name=None))
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for _, row in df_bundle.iterrows()}
bundles = list(bundle_bonus.keys())
df_usage_1['amount'] = df_usage_1['amount'].astype(int)
df_usage_2['amount'] = df_usage_2['amount'].astype(int)
df_usage = pd.concat([df_usage_1, df_usage_2], ignore_index=True)
df_usage = df_usage[df_usage['item_ref'].isin(items)]
item_resource_usage = {}
for _, row in df_usage.iterrows():
    item_resource_usage[row['item_ref'], row['resource']] = int(row['amount'])
for i in items:
    for r in resources:
        if (i, r) not in item_resource_usage:
            item_resource_usage[i, r] = 0
incompat_pairs = set()
for _, row in df_incompatible.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        incompat_pairs.add((a, b))
        incompat_pairs.add((b, a))
prereq_pairs = []
for _, row in df_requires.iterrows():
    i = row['item_ref']
    j = row['prerequisite_ref']
    if i in items and j in items:
        prereq_pairs.append((i, j))
cat_to_items = {g: [] for g in categories}
for i in items:
    g = item_to_category[i]
    cat_to_items[g].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    auth = item_to_authorized[i]
    minlot = item_to_minlot[i]
    maxorder = item_to_maxorder[i]
    if auth == 1:
        m.addConstr(x_vars[i] >= minlot * y_vars[i])
        m.addConstr(x_vars[i] <= maxorder * y_vars[i])
        m.addConstr(x_vars[i] <= maxorder)
        m.addConstr(x_vars[i] >= 0)
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * x_vars[i] for i in items)) <= resource_capacity[r])
for g in categories:
    items_in_g = cat_to_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_to_minqty[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_to_maxqty[g])
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for i, j in incompat_pairs:
    if i < j:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for i, j in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[j])
for i, j in bundles:
    m.addConstr(w_vars[i, j] <= y_vars[i])
    m.addConstr(w_vars[i, j] <= y_vars[j])
    m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
benefit_expr = gp.quicksum((item_benefit[i] * x_vars[i] for i in items))
item_fee_expr = gp.quicksum((item_fee[i] * y_vars[i] for i in items))
cat_fee_expr = gp.quicksum((cat_to_fee[g] * z_vars[g] for g in categories))
bundle_expr = gp.quicksum((bundle_bonus[i, j] * w_vars[i, j] for i, j in bundles))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_expr, gp.GRB.MAXIMIZE)
m.optimize()