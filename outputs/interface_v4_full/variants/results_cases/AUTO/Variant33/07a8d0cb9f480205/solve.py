import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_02/export_02.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_02/export_08.csv'
f_itemfee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant33/inputs/batch_01/export_13.csv'
df_benefit = pd.read_csv(f_benefit)
df_bundle = pd.read_csv(f_bundle)
df_capacity = pd.read_csv(f_capacity)
df_category = pd.read_csv(f_category)
df_identity = pd.read_csv(f_identity)
df_incompat = pd.read_csv(f_incompat)
df_item1 = pd.read_csv(f_item1)
df_item2 = pd.read_csv(f_item2)
df_itemfee = pd.read_csv(f_itemfee)
df_market = pd.read_csv(f_market)
df_requires = pd.read_csv(f_requires)
df_usage1 = pd.read_csv(f_usage1)
df_usage2 = pd.read_csv(f_usage2)
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows = item_rows[item_rows['authorized'] == 1].copy()
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
items = sorted(item_rows['item_ref'].unique())
item2cat = item_rows.set_index('item_ref')['category'].to_dict()
item2minlot = item_rows.set_index('item_ref')['minimum_lot'].to_dict()
item2maxorder = item_rows.set_index('item_ref')['maximum_order'].to_dict()
df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit = {i: benefit_per_item.get(i, 0) for i in items}
df_itemfee['item_ref'] = df_itemfee['item_ref'].astype(str)
itemfee = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee = {i: itemfee.get(i, 0) for i in items}
df_category['category'] = df_category['category'].astype(str)
cat2fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
cat2min = df_category.set_index('category')['minimum_quantity'].to_dict()
cat2max = df_category.set_index('category')['maximum_quantity'].to_dict()
categories = sorted(df_category['category'].unique())
df_usage1['item_ref'] = df_usage1['item_ref'].astype(str)
df_usage2['item_ref'] = df_usage2['item_ref'].astype(str)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['resource'] = df_usage['resource'].astype(str)
resources = sorted(df_usage['resource'].unique())
usage = {i: {r: 0 for r in resources} for i in items}
for _, row in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    if i in items:
        usage[i][r] += amt
df_capacity['resource'] = df_capacity['resource'].astype(str)
cap_ledger = df_capacity.groupby('resource')['amount'].sum().to_dict()
capacity = {r: cap_ledger.get(r, 0) for r in resources}
df_incompat['item_a'] = df_incompat['item_a'].astype(str)
df_incompat['item_b'] = df_incompat['item_b'].astype(str)
incompat_pairs = []
for _, row in df_incompat.iterrows():
    a, b = (row['item_a'], row['item_b'])
    if a in items and b in items:
        incompat_pairs.append((a, b))
df_requires['item_ref'] = df_requires['item_ref'].astype(str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
requires_pairs = []
for _, row in df_requires.iterrows():
    i, prereq = (row['item_ref'], row['prerequisite_ref'])
    if i in items and prereq in items:
        requires_pairs.append((i, prereq))
df_bundle['item_a'] = df_bundle['item_a'].astype(str)
df_bundle['item_b'] = df_bundle['item_b'].astype(str)
bundles = []
bundle_bonus = {}
for _, row in df_bundle.iterrows():
    a, b = (row['item_a'], row['item_b'])
    if a in items and b in items:
        key = (a, b)
        bundles.append(key)
        bundle_bonus[key] = row['bonus_cents']
m = gp.Model('NY_Module_Portfolio')
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
bvar = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    minlot = item2minlot[i]
    maxorder = item2maxorder[i]
    m.addConstr(x[i] <= maxorder * y[i], name=f'xmax_{i}')
    m.addConstr(x[i] >= minlot * y[i], name=f'xmin_{i}')
    m.addConstr(x[i] <= maxorder, name=f'xub_{i}')
    m.addConstr(x[i] >= 0, name=f'xlb_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage[i][r] * x[i] for i in items)) <= capacity[r], name=f'res_{r}')
for c in categories:
    items_in_c = [i for i in items if item2cat[i] == c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat2min[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat2max[c], name=f'catmax_{c}')
    for i in items_in_c:
        m.addConstr(x[i] <= item2maxorder[i] * z[c], name=f'catz_{c}_{i}')
for a, b in incompat_pairs:
    m.addConstr(y[a] + y[b] <= 1, name=f'incompat_{a}_{b}')
for i, prereq in requires_pairs:
    m.addConstr(y[i] <= y[prereq], name=f'req_{i}_{prereq}')
for a, b in bundles:
    m.addConstr(bvar[a, b] <= y[a], name=f'bundle1_{a}_{b}')
    m.addConstr(bvar[a, b] <= y[b], name=f'bundle2_{a}_{b}')
    m.addConstr(bvar[a, b] >= y[a] + y[b] - 1, name=f'bundle3_{a}_{b}')
obj = gp.quicksum((benefit[i] * x[i] for i in items)) - gp.quicksum((item_fee[i] * y[i] for i in items)) - gp.quicksum((cat2fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[a, b] * bvar[a, b] for a, b in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()