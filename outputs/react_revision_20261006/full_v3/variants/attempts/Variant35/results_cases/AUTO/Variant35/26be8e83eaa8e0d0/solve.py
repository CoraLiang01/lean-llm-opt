import gurobipy as gp
import pandas as pd
import numpy as np
import re
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv'
f_fx = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv'
df_benefit = pd.read_csv(f_benefit, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_fx = pd.read_csv(f_fx, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_item_fee = pd.read_csv(f_item_fee, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
item_rows = df_item[df_item['table'].str.casefold() == 'item']
item_refs = list(item_rows['item_ref'])
platforms = sorted(set(item_rows['location_id'].unique()))
category_rows = df_category[df_category['table'].str.casefold() == 'category']
categories = list(category_rows['category'])
bundle_rows = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_rows.iterrows()]
incompat_rows = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.casefold() == 'requires']
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_rows.iterrows()]
fx_rows = df_fx[df_fx['table'].str.casefold() == 'fx']
fx_map = {}
for (_, row) in fx_rows.iterrows():
    fx_map[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
benefit_rows = df_benefit[df_benefit['table'].str.casefold() == 'benefit']
benefit_per_unit = {}
for i in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'] == i]
    total = 0
    for (_, row) in rows.iterrows():
        currency = row['currency']
        amount = row['amount']
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_map[currency]
        val = amount * num / denom
        total += val
    benefit_per_unit[i] = total
item_fee_rows = df_item_fee[df_item_fee['table'].str.casefold() == 'item_fee']
item_fee = {}
for (_, row) in item_fee_rows.iterrows():
    item_fee[row['item_ref']] = row['activation_fee_cents']
cat_fee = {}
for (_, row) in category_rows.iterrows():
    cat_fee[row['category']] = row['activation_fee_cents']
bundle_bonus = {}
for (_, row) in bundle_rows.iterrows():
    bundle_bonus[row['item_a'], row['item_b']] = row['bonus_cents']
usage_rows = df_usage[df_usage['table'].str.casefold() == 'usage']
usage_per_unit = {}
for (_, row) in usage_rows.iterrows():
    item = row['item_ref']
    platform = row['resource']
    amount = row['amount']
    unit = row['unit']
    if unit.casefold() == 'gb':
        mb = amount * 1000
    elif unit.casefold() == 'mb':
        mb = amount
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_per_unit[item, platform] = mb
cap_rows = df_capacity_ledger[df_capacity_ledger['table'].str.casefold() == 'capacity_ledger']
platform_capacity = {}
for platform in platforms:
    rows = cap_rows[cap_rows['resource'].str.casefold() == platform.casefold()]
    total = rows['amount'].sum()
    platform_capacity[platform] = total
item_min_lot = {}
item_max_order = {}
item_authorized = {}
item_category = {}
item_platform = {}
for (_, row) in item_rows.iterrows():
    i = row['item_ref']
    item_min_lot[i] = row['minimum_lot']
    item_max_order[i] = row['maximum_order']
    item_authorized[i] = row['authorized']
    item_category[i] = row['category']
    item_platform[i] = row['location_id']
cat_min_qty = {}
cat_max_qty = {}
for (_, row) in category_rows.iterrows():
    c = row['category']
    cat_min_qty[c] = row['minimum_quantity']
    cat_max_qty[c] = row['maximum_quantity']
m = gp.Model('GameEditionAllocation')
x = m.addVars(item_refs, lb=0, ub={i: item_max_order[i] for i in item_refs}, vtype=gp.GRB.INTEGER, name='')
z = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if item_authorized[i] == 0:
        m.addConstr(x[i] == 0, name=f'auth_{i}')
        m.addConstr(z[i] == 0, name=f'authz_{i}')
    else:
        m.addConstr(x[i] >= item_min_lot[i] * z[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= item_max_order[i] * z[i], name=f'maxorder_{i}')
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_min_qty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_max_qty[c], name=f'catmax_{c}')
    for i in items_in_c:
        m.addConstr(z[i] <= w[c], name=f'catw_{c}_{i}')
for p in platforms:
    items_on_p = [i for i in item_refs if item_platform[i] == p]
    expr = gp.LinExpr()
    for i in items_on_p:
        key = (i, p)
        if key not in usage_per_unit:
            raise ValueError(f'Missing usage for item {i} on platform {p}')
        expr += x[i] * usage_per_unit[key]
    m.addConstr(expr <= platform_capacity[p], name=f'cap_{p}')
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in prereq_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(z[i] <= z[prereq], name=f'prereq_{i}_{prereq}')
for (i, j) in bundle_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(b[i, j] <= z[i], name=f'bundle1_{i}_{j}')
        m.addConstr(b[i, j] <= z[j], name=f'bundle2_{i}_{j}')
        m.addConstr(b[i, j] >= z[i] + z[j] - 1, name=f'bundle3_{i}_{j}')
    else:
        m.addConstr(b[i, j] == 0, name=f'bundle0_{i}_{j}')
obj = gp.LinExpr()
for i in item_refs:
    obj += benefit_per_unit[i] * x[i]
    if i in item_fee:
        obj -= item_fee[i] * z[i]
    else:
        obj -= 0 * z[i]
for c in categories:
    obj -= cat_fee[c] * w[c]
for pair in bundle_pairs:
    obj += bundle_bonus[pair] * b[pair]
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')