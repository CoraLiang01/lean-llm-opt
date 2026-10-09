import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv', sep=',')
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv', sep=',')
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv', sep=',')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv', sep=',')
df_item_auth = df_item[df_item['authorized'] == 1].copy()
item_ids = df_item_auth['item_ref'].astype(str).tolist()
item_set = set(item_ids)
category_ids = df_category['category'].astype(str).tolist()
category_set = set(category_ids)
resource_ids = sorted(set(df_usage['resource'].astype(str)).union(df_capacity['resource'].astype(str)))
resource_set = set(resource_ids)
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_set and b in item_set:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = str(row['item_ref'])
    p = str(row['prerequisite_ref'])
    if i in item_set and p in item_set:
        prereq_pairs.append((i, p))
bundle_tuples = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_set and b in item_set:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
item_minlot = df_item_auth.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_maxorder = df_item_auth.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_unit_benefit = df_item_auth.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee = df_item_auth.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
item_category = df_item_auth.set_index('item_ref')['category'].astype(str).to_dict()
cat_minqty = df_category.set_index('category')['minimum_quantity'].astype(int).to_dict()
cat_maxqty = df_category.set_index('category')['maximum_quantity'].astype(int).to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].astype(int).to_dict()
usage = {i: {} for i in item_ids}
for (_, row) in df_usage.iterrows():
    i = str(row['item_ref'])
    if i not in item_set:
        continue
    r = str(row['resource'])
    amt = int(row['amount'])
    unit = str(row['unit']).strip().lower()
    if r == 'space':
        if unit == 'liter':
            amt = amt * 1000
        elif unit == 'ml':
            amt = amt
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif r == 'labor':
        if unit == 'hour':
            amt = amt * 60
        elif unit == 'minute':
            amt = amt
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    elif r == 'power':
        if unit == 'kwh':
            amt = amt * 1000
        elif unit == 'wh':
            amt = amt
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    else:
        raise ValueError(f'Unknown resource: {r}')
    usage[i][r] = amt
for i in item_ids:
    for r in resource_ids:
        if r not in usage[i]:
            usage[i][r] = 0
capacity = {}
for r in resource_ids:
    df_r = df_capacity[df_capacity['resource'] == r]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = int(row['amount'])
        unit = str(row['unit']).strip().lower()
        if r == 'space':
            if unit == 'liter':
                amt = amt * 1000
            elif unit == 'ml':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for space: {unit}')
        elif r == 'labor':
            if unit == 'hour':
                amt = amt * 60
            elif unit == 'minute':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for labor: {unit}')
        elif r == 'power':
            if unit == 'kwh':
                amt = amt * 1000
            elif unit == 'wh':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for power: {unit}')
        else:
            raise ValueError(f'Unknown resource: {r}')
        total += amt
    capacity[r] = total
m = gp.Model('BakeryOrderNetBenefit')
x = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'maxorder_{i}')
for c in category_ids:
    items_in_c = [i for i in item_ids if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(z[c] >= y[i], name=f'catact_{c}_{i}')
for c in category_ids:
    items_in_c = [i for i in item_ids if item_category[i] == c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_minqty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_maxqty[c], name=f'catmax_{c}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage[i][r] * x[i] for i in item_ids)) <= capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, p) in prereq_pairs:
    m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
for (i, j) in bundle_tuples:
    m.addConstr(b[i, j] <= y[i], name=f'bundle1_{i}_{j}')
    m.addConstr(b[i, j] <= y[j], name=f'bundle2_{i}_{j}')
    m.addConstr(b[i, j] >= y[i] + y[j] - 1, name=f'bundle3_{i}_{j}')
item_benefit = gp.quicksum((item_unit_benefit[i] * x[i] - item_fee[i] * y[i] for i in item_ids))
cat_fee_term = gp.quicksum((-cat_fee[c] * z[c] for c in category_ids))
bundle_term = gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundle_tuples))
m.setObjective(item_benefit + cat_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} USD cents')
    print('--- Order Plan ---')
    for i in item_ids:
        qty = x[i].X
        if qty > 1e-06:
            print(f'  Item {i}: {int(round(qty))} units (category {item_category[i]})')
    print('--- Category Activations ---')
    for c in category_ids:
        if z[c].X > 0.5:
            print(f'  Category {c}: ACTIVATED (fee {cat_fee[c]} cents)')
    print('--- Bundle Bonuses ---')
    for (i, j) in bundle_tuples:
        if b[i, j].X > 0.5:
            print(f'  Bundle ({i}, {j}): bonus {bundle_bonus[i, j]} cents')
else:
    print(f'No optimal solution found. Status: {m.status}')