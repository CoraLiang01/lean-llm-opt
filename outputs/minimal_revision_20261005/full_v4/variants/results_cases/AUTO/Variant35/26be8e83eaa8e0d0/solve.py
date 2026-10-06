import gurobipy as gp
import pandas as pd
import numpy as np
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
item_refs = list(item_rows['item_ref'].astype(str).unique())
platforms = sorted(set(item_rows['location_id'].astype(str).unique()))
resources = sorted(set(df_capacity_ledger['resource'].astype(str).unique()))
assert set(platforms) == set(resources), 'Mismatch between platforms in item and resources in capacity_ledger'
category_rows = df_category[df_category['table'].str.casefold() == 'category']
categories = list(category_rows['category'].astype(str).unique())
bundle_rows = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
bundle_keys = list(bundle_rows.index)
fx_rows = df_fx[df_fx['table'].str.casefold() == 'fx']
fx_map = {}
for (_, row) in fx_rows.iterrows():
    fx_map[row['currency'].strip()] = (int(row['usd_cents_numerator']), int(row['denominator']))
benefit_rows = df_benefit[df_benefit['table'].str.casefold() == 'benefit']
benefit_per_unit = {}
for i in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'].astype(str) == i]
    total = 0
    for (_, r) in rows.iterrows():
        currency = r['currency'].strip()
        amt = int(r['amount'])
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_map[currency]
        total += amt * num // denom
    benefit_per_unit[i] = total
item_fee_rows = df_item_fee[df_item_fee['table'].str.casefold() == 'item_fee']
item_fee = {}
for i in item_refs:
    rows = item_fee_rows[item_fee_rows['item_ref'].astype(str) == i]
    if not rows.empty:
        item_fee[i] = int(rows.iloc[0]['activation_fee_cents'])
    else:
        item_fee[i] = 0
category_fee = {}
for (_, row) in category_rows.iterrows():
    c = str(row['category'])
    category_fee[c] = int(row['activation_fee_cents'])
bundle_bonus = {}
bundle_item_a = {}
bundle_item_b = {}
for (idx, row) in bundle_rows.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    bundle_bonus[idx] = int(row['bonus_cents'])
    bundle_item_a[idx] = a
    bundle_item_b[idx] = b
usage_rows = df_usage[df_usage['table'].str.casefold() == 'usage']
usage_per_unit = {}
for (_, row) in usage_rows.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit == 'gb':
        amt_mb = amt * 1000
    elif unit == 'mb':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_per_unit[i, r] = amt_mb
capacity_rows = df_capacity_ledger[df_capacity_ledger['table'].str.casefold() == 'capacity_ledger']
platform_capacity = {}
for r in resources:
    rows = capacity_rows[capacity_rows['resource'].astype(str) == r]
    total = rows['amount'].sum()
    platform_capacity[r] = int(total)
category_min = {}
category_max = {}
for (_, row) in category_rows.iterrows():
    c = str(row['category'])
    category_min[c] = int(row['minimum_quantity'])
    category_max[c] = int(row['maximum_quantity'])
authorized = {}
minimum_lot = {}
maximum_order = {}
item_category = {}
item_platform = {}
for (_, row) in item_rows.iterrows():
    i = str(row['item_ref'])
    authorized[i] = int(row['authorized'])
    minimum_lot[i] = int(row['minimum_lot'])
    maximum_order[i] = int(row['maximum_order'])
    item_category[i] = str(row['category'])
    item_platform[i] = str(row['location_id'])
incompat_rows = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
incompat_pairs = []
for (_, row) in incompat_rows.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompat_pairs.append((a, b))
requires_rows = df_requires[df_requires['table'].str.casefold() == 'requires']
prereq_pairs = []
for (_, row) in requires_rows.iterrows():
    i = str(row['item_ref'])
    pre = str(row['prerequisite_ref'])
    prereq_pairs.append((i, pre))
items_by_category = {c: [] for c in categories}
for i in item_refs:
    c = item_category[i]
    items_by_category[c].append(i)
items_by_platform = {r: [] for r in resources}
for i in item_refs:
    r = item_platform[i]
    items_by_platform[r].append(i)
m = gp.Model('GameEditionAllocation')
m.Params.MIPGap = 0.0001
x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if authorized[i] == 0:
        m.addConstr(x[i] == 0, name=f'auth0_{i}')
        m.addConstr(z[i] == 0, name=f'auth0z_{i}')
    else:
        m.addConstr(x[i] <= maximum_order[i] * z[i], name=f'ub_{i}')
        m.addConstr(x[i] >= minimum_lot[i] * z[i], name=f'lb_{i}')
        m.addConstr(x[i] <= maximum_order[i], name=f'maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
for r in resources:
    expr = gp.LinExpr()
    for i in items_by_platform[r]:
        usage = usage_per_unit.get((i, r), 0)
        expr += usage * x[i]
    m.addConstr(expr <= platform_capacity[r], name=f'cap_{r}')
for c in categories:
    expr = gp.LinExpr()
    for i in items_by_category[c]:
        expr += x[i]
    m.addConstr(expr >= category_min[c] * w[c], name=f'catmin_{c}')
    m.addConstr(expr <= category_max[c] * w[c], name=f'catmax_{c}')
    for i in items_by_category[c]:
        m.addConstr(z[i] <= w[c], name=f'catw_{c}_{i}')
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
for (i, pre) in prereq_pairs:
    if i in item_refs and pre in item_refs:
        m.addConstr(z[i] <= z[pre], name=f'prereq_{i}_{pre}')
for idx in bundle_keys:
    a = bundle_item_a[idx]
    b_ = bundle_item_b[idx]
    if a in item_refs and b_ in item_refs:
        m.addConstr(b[idx] <= z[a], name=f'bundle_a_{idx}')
        m.addConstr(b[idx] <= z[b_], name=f'bundle_b_{idx}')
        m.addConstr(b[idx] >= z[a] + z[b_] - 1, name=f'bundle_and_{idx}')
    else:
        m.addConstr(b[idx] == 0, name=f'bundle_invalid_{idx}')
obj_benefit = gp.quicksum((benefit_per_unit[i] * x[i] for i in item_refs))
obj_item_fee = gp.quicksum((item_fee[i] * z[i] for i in item_refs))
obj_cat_fee = gp.quicksum((category_fee[c] * w[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus[idx] * b[idx] for idx in bundle_keys))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')