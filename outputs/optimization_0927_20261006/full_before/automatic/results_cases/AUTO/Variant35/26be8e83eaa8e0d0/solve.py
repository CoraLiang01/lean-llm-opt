import gurobipy as gp
import pandas as pd
import numpy as np
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', sep=',')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', sep=',')
df_fx = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', sep=',')
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', sep=',')
df_itemfee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', sep=',')
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv', sep=',')
item_refs = df_item['item_ref'].astype(str).tolist()
item_set = set(item_refs)
platforms = sorted(df_item['location_id'].unique())
platforms_set = set(platforms)
categories = df_category['category'].astype(str).tolist()
category_set = set(categories)
bundle_tuples = [(str(row['item_a']), str(row['item_b'])) for (_, row) in df_bundle.iterrows()]
incompat_pairs = [(str(row['item_a']), str(row['item_b'])) for (_, row) in df_incompat.iterrows()]
prereq_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for (_, row) in df_requires.iterrows()]
fx_dict = {}
for (_, row) in df_fx.iterrows():
    fx_dict[str(row['currency'])] = (int(row['usd_cents_numerator']), int(row['denominator']))
benefit_per_unit = {}
for item in item_refs:
    df_ben = df_benefit[df_benefit['item_ref'].astype(str) == item]
    total = 0
    for (_, row) in df_ben.iterrows():
        amt = int(row['amount'])
        curr = str(row['currency'])
        if curr not in fx_dict:
            raise ValueError(f'Missing FX rate for currency {curr}')
        (num, denom) = fx_dict[curr]
        val = amt * num / denom
        total += val
    benefit_per_unit[item] = int(round(total))
item_fee = {}
for (_, row) in df_itemfee.iterrows():
    item_fee[str(row['item_ref'])] = int(row['activation_fee_cents'])
category_fee = {}
for (_, row) in df_category.iterrows():
    category_fee[str(row['category'])] = int(row['activation_fee_cents'])
bundle_bonus = {}
for (idx, (a, b)) in enumerate(bundle_tuples):
    bonus = int(df_bundle.iloc[idx]['bonus_cents'])
    bundle_bonus[a, b] = bonus
usage_per_item_platform = {}
for (_, row) in df_usage.iterrows():
    item = str(row['item_ref'])
    platform = str(row['resource'])
    amt = int(row['amount'])
    unit = str(row['unit']).strip().upper()
    if unit == 'GB':
        amt_mb = amt * 1000
    elif unit == 'MB':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown memory unit: {unit}')
    usage_per_item_platform[item, platform] = amt_mb
platform_capacity = {}
for platform in platforms:
    df_cap = df_capacity[df_capacity['resource'].astype(str) == platform]
    total = 0
    for (_, row) in df_cap.iterrows():
        amt = int(row['amount'])
        unit = str(row['unit']).strip().upper()
        if unit != 'MB':
            raise ValueError(f'Unknown capacity unit: {unit}')
        total += amt
    platform_capacity[platform] = total
category_min = {}
category_max = {}
for (_, row) in df_category.iterrows():
    cat = str(row['category'])
    category_min[cat] = int(row['minimum_quantity'])
    category_max[cat] = int(row['maximum_quantity'])
item_authorized = {}
item_minlot = {}
item_maxorder = {}
item_category = {}
item_platform = {}
for (_, row) in df_item.iterrows():
    item = str(row['item_ref'])
    item_authorized[item] = int(row['authorized'])
    item_minlot[item] = int(row['minimum_lot'])
    item_maxorder[item] = int(row['maximum_order'])
    item_category[item] = str(row['category'])
    item_platform[item] = str(row['location_id'])
m = gp.Model('GameEditionAllocation')
x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if item_authorized[i] == 0:
        m.addConstr(x[i] == 0, name=f'unauth_{i}')
        m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'maxorder_{i}')
for p in platforms:
    items_on_p = [i for i in item_refs if item_platform[i] == p]
    m.addConstr(gp.quicksum((usage_per_item_platform.get((i, p), 0) * x[i] for i in items_on_p)) <= platform_capacity[p], name=f'memcap_{p}')
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= category_min[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= category_max[c], name=f'catmax_{c}')
    m.addConstr(gp.quicksum((y[i] for i in items_in_c)) <= len(items_in_c) * z[c], name=f'catz1_{c}')
    m.addConstr(gp.quicksum((y[i] for i in items_in_c)) >= z[c], name=f'catz2_{c}')
for i in item_refs:
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'link1_{i}')
for (i, j) in incompat_pairs:
    if i in item_set and j in item_set:
        m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in prereq_pairs:
    if i in item_set and prereq in item_set:
        m.addConstr(y[i] <= y[prereq], name=f'prereq_{i}_{prereq}')
for (a, bnd) in bundle_tuples:
    if a in item_set and bnd in item_set:
        m.addConstr(b[a, bnd] <= y[a], name=f'bundle1_{a}_{bnd}')
        m.addConstr(b[a, bnd] <= y[bnd], name=f'bundle2_{a}_{bnd}')
        m.addConstr(b[a, bnd] >= y[a] + y[bnd] - 1, name=f'bundle3_{a}_{bnd}')
    else:
        m.addConstr(b[a, bnd] == 0, name=f'bundle0_{a}_{bnd}')
obj_benefit = gp.quicksum((benefit_per_unit[i] * x[i] for i in item_refs))
obj_itemfee = gp.quicksum((item_fee.get(i, 0) * y[i] for i in item_refs))
obj_catfee = gp.quicksum((category_fee[c] * z[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus[a, b] * b[a, b] for (a, b) in bundle_tuples))
m.setObjective(obj_benefit - obj_itemfee - obj_catfee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net benefit (USD cents): {int(round(m.objVal))}')
    print('\nSelected items (item_ref, quantity, platform, category):')
    for i in item_refs:
        if x[i].X > 0.5:
            print(f'  {i}: {int(round(x[i].X))} units on {item_platform[i]}, category {item_category[i]}')
    print('\nActivated categories (category):')
    for c in categories:
        if z[c].X > 0.5:
            print(f'  {c}')
    print('\nActivated bundles (item_a, item_b):')
    for (a, bnd) in bundle_tuples:
        if b[a, bnd].X > 0.5:
            print(f'  ({a}, {bnd})')
else:
    print(f'No optimal solution found. Status: {m.status}')