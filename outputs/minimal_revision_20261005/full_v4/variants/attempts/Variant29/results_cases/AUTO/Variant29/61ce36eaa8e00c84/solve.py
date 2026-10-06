import gurobipy as gp
import pandas as pd
import numpy as np
import re
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
df_requires = pd.read_csv(f_requires, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_item['authorized'] = df_item['authorized'].astype(int)
authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].astype(str).tolist()
authorized_items_set = set(authorized_items)
categories = df_category['category'].astype(str).tolist()
categories_set = set(categories)
item_to_category = df_item.set_index('item_ref')['category'].astype(str).to_dict()
minimum_lot = df_item.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = df_item.set_index('item_ref')['maximum_order'].astype(int).to_dict()
unit_benefit_cents = df_item.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = df_item.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
minimum_quantity = df_category.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = df_category.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = df_category.set_index('category')['activation_fee_cents'].astype(int).to_dict()
df_usage['item_ref'] = df_usage['item_ref'].astype(str)
df_usage = df_usage[df_usage['item_ref'].isin(authorized_items)]
resources = df_usage['resource'].unique().tolist()
resources_set = set(resources)
usage = {}
for (_, row) in df_usage.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    amt = float(row['amount'])
    unit = str(row['unit']).strip().casefold()
    if r == 'space':
        if unit == 'liter':
            amt = amt * 1000.0
        elif unit == 'ml':
            pass
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif r == 'labor':
        if unit == 'hour':
            amt = amt * 60.0
        elif unit == 'minute':
            pass
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    elif r == 'power':
        if unit == 'kwh':
            amt = amt * 1000.0
        elif unit == 'wh':
            pass
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    else:
        raise ValueError(f'Unknown resource: {r}')
    usage[i, r] = amt
capacity = {}
for r in resources:
    df_r = df_capacity[df_capacity['resource'].str.casefold() == r.casefold()]
    total = 0.0
    for (_, row) in df_r.iterrows():
        amt = float(row['amount'])
        unit = str(row['unit']).strip().casefold()
        if r == 'space':
            if unit == 'liter':
                amt = amt * 1000.0
            elif unit == 'ml':
                pass
            else:
                raise ValueError(f'Unknown unit for space: {unit}')
        elif r == 'labor':
            if unit == 'hour':
                amt = amt * 60.0
            elif unit == 'minute':
                pass
            else:
                raise ValueError(f'Unknown unit for labor: {unit}')
        elif r == 'power':
            if unit == 'kwh':
                amt = amt * 1000.0
            elif unit == 'wh':
                pass
            else:
                raise ValueError(f'Unknown unit for power: {unit}')
        else:
            raise ValueError(f'Unknown resource: {r}')
        total += amt
    capacity[r] = total
incompat_pairs = []
for (_, row) in df_incompatible.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in authorized_items_set and b in authorized_items_set:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = str(row['item_ref'])
    prereq = str(row['prerequisite_ref'])
    if i in authorized_items_set and prereq in authorized_items_set:
        prereq_pairs.append((i, prereq))
bundle_tuples = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in authorized_items_set and b in authorized_items_set:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
cat_to_items = {c: [] for c in categories}
for i in authorized_items:
    c = item_to_category[i]
    cat_to_items[c].append(i)
m = gp.Model('bakery_order')
m.Params.MIPGap = 0.0001
x = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(x[i] >= y[i] * minimum_lot[i], name=f'lot_lb_{i}')
    m.addConstr(x[i] <= y[i] * maximum_order[i], name=f'lot_ub_{i}')
for c in categories:
    items_in_c = cat_to_items[c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= minimum_quantity[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= maximum_quantity[c], name=f'cat_max_{c}')
    M_c = sum((maximum_order[i] for i in items_in_c))
    for i in items_in_c:
        m.addConstr(x[i] <= M_c * z[c], name=f'cat_link_{c}_{i}')
for r in resources:
    expr = gp.LinExpr()
    for i in authorized_items:
        if (i, r) in usage:
            expr += usage[i, r] * x[i]
    m.addConstr(expr <= capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in prereq_pairs:
    m.addConstr(y[i] <= y[prereq], name=f'prereq_{i}_{prereq}')
for (a, b) in bundle_tuples:
    m.addConstr(b[a, b] <= y[a], name=f'bundle1_{a}_{b}')
    m.addConstr(b[a, b] <= y[b], name=f'bundle2_{a}_{b}')
    m.addConstr(b[a, b] >= y[a] + y[b] - 1, name=f'bundle3_{a}_{b}')
obj = gp.LinExpr()
obj += gp.quicksum((unit_benefit_cents[i] * x[i] for i in authorized_items))
obj -= gp.quicksum((item_fee_cents[i] * y[i] for i in authorized_items))
obj -= gp.quicksum((activation_fee_cents[c] * z[c] for c in categories))
if bundle_tuples:
    obj += gp.quicksum((bundle_bonus[a, b] * b[a, b] for (a, b) in bundle_tuples))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in authorized_items:
        print(f'x[{i}] {x[i].VarName} {x[i].X}')
        print(f'y[{i}] {y[i].VarName} {y[i].X}')
    for c in categories:
        print(f'z[{c}] {z[c].VarName} {z[c].X}')
    for (a, b) in bundle_tuples:
        print(f'b[{a},{b}] {b[a, b].VarName} {b[a, b].X}')
else:
    print(f'Solver status: {m.status}')