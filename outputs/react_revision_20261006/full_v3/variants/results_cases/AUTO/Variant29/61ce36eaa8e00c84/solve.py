import gurobipy as gp
import pandas as pd
import numpy as np
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
df_requires = pd.read_csv(f_requires, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_item = df_item[df_item['authorized'] == 1].copy()
authorized_items = set(df_item['item_ref'])
item_minlot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = df_item.set_index('item_ref')['maximum_order'].to_dict()
item_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
item_category = df_item.set_index('item_ref')['category'].to_dict()
categories = set(df_category['category'])
cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_usage = df_usage[df_usage['item_ref'].isin(authorized_items)].copy()
resources = set(df_usage['resource'])
unit_conv = {'liter': 1000, 'ml': 1, 'hour': 60, 'minute': 1, 'kwh': 1000, 'wh': 1}
usage = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    u = row['unit'].strip().casefold()
    if u not in unit_conv:
        raise ValueError(f'Unknown unit {u} for item {i}, resource {r}')
    usage[i, r] = amt * unit_conv[u]
df_capacity = df_capacity.copy()
resource_caps = {}
for r in df_capacity['resource'].unique():
    df_r = df_capacity[df_capacity['resource'] == r]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if u not in unit_conv:
            raise ValueError(f'Unknown unit {u} for resource {r}')
        total += amt * unit_conv[u]
    resource_caps[r] = total
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    j = row['prerequisite_ref']
    if i in authorized_items and j in authorized_items:
        prereq_pairs.append((i, j))
bundle_list = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in authorized_items and b in authorized_items:
        key = (a, b)
        bundle_list.append(key)
        bundle_bonus[key] = row['bonus_cents']
cat_items = {c: set(df_item[df_item['category'] == c]['item_ref']) for c in categories}
m = gp.Model('BakeryOrder')
m.Params.MIPGap = 0.0001
x = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'lot_lb_{i}')
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'lot_ub_{i}')
for c in categories:
    for i in cat_items[c]:
        m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
for c in categories:
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c] * z[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c] * z[c], name=f'cat_max_{c}')
for r in resources:
    expr = gp.LinExpr()
    for i in authorized_items:
        if (i, r) in usage:
            expr += usage[i, r] * x[i]
    m.addConstr(expr <= resource_caps[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in prereq_pairs:
    m.addConstr(y[i] <= y[j], name=f'prereq_{i}_{j}')
for (a, b) in bundle_list:
    m.addConstr(b[a, b] <= y[a], name=f'bundle1_{a}_{b}')
    m.addConstr(b[a, b] <= y[b], name=f'bundle2_{a}_{b}')
    m.addConstr(b[a, b] >= y[a] + y[b] - 1, name=f'bundle3_{a}_{b}')
obj = gp.LinExpr()
obj += gp.quicksum((item_benefit[i] * x[i] for i in authorized_items))
obj -= gp.quicksum((item_fee[i] * y[i] for i in authorized_items))
obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
obj += gp.quicksum((bundle_bonus[a, b] * b[a, b] for (a, b) in bundle_list))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')