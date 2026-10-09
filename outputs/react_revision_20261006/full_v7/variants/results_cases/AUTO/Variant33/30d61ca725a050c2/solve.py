import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
(df_benefit, df_bundle, df_capacity_ledger, df_category, df_identity, df_incompatible, df_item1, df_item2, df_item_fee, df_market, df_requires, df_usage1, df_usage2) = dfs
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
df_item1['authorized'] = df_item1['authorized'].astype(int)
df_item1['minimum_lot'] = df_item1['minimum_lot'].astype(int)
df_item1['maximum_order'] = df_item1['maximum_order'].astype(int)
df_item2['authorized'] = df_item2['authorized'].astype(int)
df_item2['minimum_lot'] = df_item2['minimum_lot'].astype(int)
df_item2['maximum_order'] = df_item2['maximum_order'].astype(int)
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
df_usage1['amount'] = df_usage1['amount'].astype(int)
df_usage2['amount'] = df_usage2['amount'].astype(int)
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows = item_rows[item_rows['authorized'] == 1].copy()
item_refs = item_rows['item_ref'].unique().tolist()
item_category = dict(zip(item_rows['item_ref'], item_rows['category']))
item_min_lot = dict(zip(item_rows['item_ref'], item_rows['minimum_lot']))
item_max_order = dict(zip(item_rows['item_ref'], item_rows['maximum_order']))
benefit_rows = df_benefit[df_benefit['table'].str.casefold() == 'benefit']
item_benefit = benefit_rows.groupby('item_ref')['amount_cents'].sum().to_dict()
item_benefit = {i: item_benefit.get(i, 0) for i in item_refs}
item_fee_rows = df_item_fee[df_item_fee['table'].str.casefold() == 'item_fee']
item_activation_fee = dict(zip(item_fee_rows['item_ref'], item_fee_rows['activation_fee_cents']))
item_activation_fee = {i: item_activation_fee.get(i, 0) for i in item_refs}
category_rows = df_category[df_category['table'].str.casefold() == 'category']
categories = category_rows['category'].unique().tolist()
category_min = dict(zip(category_rows['category'], category_rows['minimum_quantity']))
category_max = dict(zip(category_rows['category'], category_rows['maximum_quantity']))
category_activation_fee = dict(zip(category_rows['category'], category_rows['activation_fee_cents']))
category_items = {c: [i for i in item_refs if item_category[i] == c] for c in categories}
resource_rows = df_capacity_ledger[df_capacity_ledger['table'].str.casefold() == 'capacity_ledger']
resources = resource_rows['resource'].unique().tolist()
resource_total = resource_rows.groupby('resource')['amount'].sum().to_dict()
usage_rows = pd.concat([df_usage1, df_usage2], ignore_index=True)
usage_rows = usage_rows[usage_rows['table'].str.casefold() == 'usage']
item_resource_usage = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    if i in item_refs:
        item_resource_usage[i, r] = amt
for i in item_refs:
    for r in resources:
        if (i, r) not in item_resource_usage:
            item_resource_usage[i, r] = 0
incompat_rows = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
incompat_pairs = []
for (_, row) in incompat_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        incompat_pairs.append((a, b))
requires_rows = df_requires[df_requires['table'].str.casefold() == 'requires']
prereq_pairs = []
for (_, row) in requires_rows.iterrows():
    i = row['item_ref']
    pre = row['prerequisite_ref']
    if i in item_refs and pre in item_refs:
        prereq_pairs.append((i, pre))
bundle_rows = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
bundle_tuples = []
bundle_bonus = {}
for (_, row) in bundle_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs and b in item_refs:
        key = (a, b)
        bundle_tuples.append(key)
        bundle_bonus[key] = int(row['bonus_cents'])
m = gp.Model('NY_Dev_Module_Portfolio')
quantity_vars = m.addVars(item_refs, lb=0, ub=[item_max_order[i] for i in item_refs], vtype=gp.GRB.INTEGER, name='')
item_active_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
category_active_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
bundle_active_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(quantity_vars[i] >= min_lot * item_active_vars[i], name='')
    m.addConstr(quantity_vars[i] <= max_order * item_active_vars[i], name='')
for i in item_refs:
    m.addConstr(quantity_vars[i] >= item_active_vars[i], name='')
    m.addConstr(quantity_vars[i] <= item_max_order[i] * item_active_vars[i], name='')
for c in categories:
    items_in_c = category_items[c]
    for i in items_in_c:
        m.addConstr(item_active_vars[i] <= category_active_vars[c], name='')
    if items_in_c:
        m.addConstr(gp.quicksum((item_active_vars[i] for i in items_in_c)) >= category_active_vars[c], name='')
for c in categories:
    items_in_c = category_items[c]
    minq = category_min[c]
    maxq = category_max[c]
    m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) >= minq * category_active_vars[c], name='')
    m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) <= maxq * category_active_vars[c], name='')
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * quantity_vars[i] for i in item_refs)) <= resource_total[r], name='')
for (a, b) in incompat_pairs:
    m.addConstr(item_active_vars[a] + item_active_vars[b] <= 1, name='')
for (i, pre) in prereq_pairs:
    m.addConstr(item_active_vars[i] <= item_active_vars[pre], name='')
for (a, b) in bundle_tuples:
    m.addConstr(bundle_active_vars[a, b] <= item_active_vars[a], name='')
    m.addConstr(bundle_active_vars[a, b] <= item_active_vars[b], name='')
    m.addConstr(bundle_active_vars[a, b] >= item_active_vars[a] + item_active_vars[b] - 1, name='')
obj = gp.quicksum((item_benefit[i] * quantity_vars[i] for i in item_refs))
obj -= gp.quicksum((item_activation_fee[i] * item_active_vars[i] for i in item_refs))
obj -= gp.quicksum((category_activation_fee[c] * category_active_vars[c] for c in categories))
obj += gp.quicksum((bundle_bonus[a, b] * bundle_active_vars[a, b] for (a, b) in bundle_tuples))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()