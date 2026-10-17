import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
items_df = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_ids = items_df['item_ref'].tolist()
authorized = items_df.set_index('item_ref')['authorized'].astype(int).to_dict()
minimum_lot = items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_category = items_df.set_index('item_ref')['category'].to_dict()
unit_benefit_cents = items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
categories_df = df_category[df_category['table'].str.strip().str.casefold() == 'category']
category_ids = categories_df['category'].tolist()
minimum_quantity = categories_df.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = categories_df.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = categories_df.set_index('category')['activation_fee_cents'].astype(int).to_dict()
resources = sorted(df_capacity['resource'].unique())
capacity_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
resource_capacity = {}
for r in resources:
    r_df = capacity_df[capacity_df['resource'] == r]
    total = r_df['amount'].astype(int).sum()
    resource_capacity[r] = total
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
item_resource_usage = {}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    item_resource_usage[i, r] = amt
bundles_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_tuples = []
bundle_bonus_cents = {}
for _, row in bundles_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_tuples.append((a, b))
    bundle_bonus_cents[a, b] = int(row['bonus_cents'])
incompat_df = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = []
for _, row in incompat_df.iterrows():
    i = row['item_a']
    j = row['item_b']
    incompat_pairs.append((i, j))
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = []
for _, row in requires_df.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    requires_pairs.append((i, prereq))
category_items = {c: [] for c in category_ids}
for i in item_ids:
    c = item_category[i]
    if c in category_items:
        category_items[c].append(i)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if authorized[i] == 1:
        m.addConstr(q_vars[i] >= minimum_lot[i] * z_vars[i])
        m.addConstr(q_vars[i] <= maximum_order[i] * z_vars[i])
        m.addConstr(q_vars[i] >= 0)
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * q_vars[i] for i in item_ids)) <= resource_capacity[r])
for c in category_ids:
    items_in_c = category_items[c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= maximum_quantity[c])
    for i in items_in_c:
        m.addConstr(y_vars[c] >= z_vars[i])
for i, j in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for i, prereq in requires_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(z_vars[i] <= z_vars[prereq])
for a, b in bundle_tuples:
    if a in item_ids and b in item_ids:
        m.addConstr(b_vars[a, b] <= z_vars[a])
        m.addConstr(b_vars[a, b] <= z_vars[b])
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
obj = gp.quicksum((unit_benefit_cents[i] * q_vars[i] for i in item_ids)) - gp.quicksum((item_fee_cents[i] * z_vars[i] for i in item_ids)) - gp.quicksum((activation_fee_cents[c] * y_vars[c] for c in category_ids)) + gp.quicksum((bundle_bonus_cents[a, b] * b_vars[a, b] for a, b in bundle_tuples))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()