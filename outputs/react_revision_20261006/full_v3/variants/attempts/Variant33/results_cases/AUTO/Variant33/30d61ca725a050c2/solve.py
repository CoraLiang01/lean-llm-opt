import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv'
f_itemfee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv'
df_benefit = pd.read_csv(f_benefit, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_item1 = pd.read_csv(f_item1, sep=',')
df_item2 = pd.read_csv(f_item2, sep=',')
df_itemfee = pd.read_csv(f_itemfee, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage1 = pd.read_csv(f_usage1, sep=',')
df_usage2 = pd.read_csv(f_usage2, sep=',')
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows = item_rows[item_rows['authorized'] == 1].copy()
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
items = sorted(item_rows['item_ref'].unique())
item_cat = item_rows.set_index('item_ref')['category'].to_dict()
item_minlot = item_rows.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = item_rows.set_index('item_ref')['maximum_order'].to_dict()
for i in items:
    if i not in item_minlot or i not in item_maxorder or i not in item_cat:
        raise ValueError(f'Missing minlot/maxorder/category for item {i}')
df_category['category'] = df_category['category'].astype(str)
categories = sorted(df_category['category'].unique())
cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
for c in categories:
    if c not in cat_minqty or c not in cat_maxqty or c not in cat_fee:
        raise ValueError(f'Missing category bounds/fee for category {c}')
cat_items = {c: [] for c in categories}
for i in items:
    c = item_cat[i]
    if c in cat_items:
        cat_items[c].append(i)
    else:
        raise ValueError(f'Item {i} has unknown category {c}')
df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
benefit = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit = {i: benefit.get(i, 0) for i in items}
df_itemfee['item_ref'] = df_itemfee['item_ref'].astype(str)
item_fee = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee = {i: item_fee.get(i, 0) for i in items}
df_bundle['item_a'] = df_bundle['item_a'].astype(str)
df_bundle['item_b'] = df_bundle['item_b'].astype(str)
bundles = []
bundle_bonus = {}
for (idx, row) in df_bundle.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        bundles.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
df_incompat['item_a'] = df_incompat['item_a'].astype(str)
df_incompat['item_b'] = df_incompat['item_b'].astype(str)
incompatibles = []
for (idx, row) in df_incompat.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        incompatibles.append((a, b))
df_requires['item_ref'] = df_requires['item_ref'].astype(str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
prerequisites = []
for (idx, row) in df_requires.iterrows():
    (i, pre) = (row['item_ref'], row['prerequisite_ref'])
    if i in items and pre in items:
        prerequisites.append((i, pre))
df_usage1['item_ref'] = df_usage1['item_ref'].astype(str)
df_usage2['item_ref'] = df_usage2['item_ref'].astype(str)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['resource'] = df_usage['resource'].astype(str)
resources = sorted(df_capacity['resource'].unique())
usage = {(i, r): 0 for i in items for r in resources}
for (idx, row) in df_usage.iterrows():
    (i, r) = (row['item_ref'], row['resource'])
    if i in items and r in resources:
        usage[i, r] += row['amount']
df_capacity['resource'] = df_capacity['resource'].astype(str)
capacity = df_capacity.groupby('resource')['amount'].sum().to_dict()
capacity = {r: capacity.get(r, 0) for r in resources}

def solve_problem():
    m = gp.Model('NY_DevModule_Portfolio')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        minlot = item_minlot[i]
        maxorder = item_maxorder[i]
        m.addConstr(x[i] >= minlot * y[i], name='item_minlot_' + i)
        m.addConstr(x[i] <= maxorder * y[i], name='item_maxorder_' + i)
    for c in categories:
        for i in cat_items[c]:
            m.addConstr(x[i] <= item_maxorder[i] * z[c], name='cat_link_' + i)
    for c in categories:
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c], name='cat_min_' + c)
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c], name='cat_max_' + c)
    for r in resources:
        m.addConstr(gp.quicksum((usage[i, r] * x[i] for i in items)) <= capacity[r], name='res_' + r)
    for (i, j) in incompatibles:
        m.addConstr(y[i] + y[j] <= 1, name='incompat_' + i + '_' + j)
    for (i, pre) in prerequisites:
        m.addConstr(y[i] <= y[pre], name='prereq_' + i + '_' + pre)
    for (i, j) in bundles:
        m.addConstr(b[i, j] <= y[i], name='bundle1_' + i + '_' + j)
        m.addConstr(b[i, j] <= y[j], name='bundle2_' + i + '_' + j)
        m.addConstr(b[i, j] >= y[i] + y[j] - 1, name='bundle3_' + i + '_' + j)
    obj = gp.quicksum((benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
    obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')