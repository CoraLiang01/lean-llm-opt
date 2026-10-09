import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
df_benefit = dfs[0]
df_bundle = dfs[1]
df_capacity_ledger = dfs[2]
df_category = dfs[3]
df_fx = dfs[4]
df_identity = dfs[5]
df_incompatible = dfs[6]
df_item = dfs[7]
df_item_fee = dfs[8]
df_market = dfs[9]
df_requires = dfs[10]
df_usage = dfs[11]
item_rows = df_item[df_item['table'].str.casefold() == 'item']
item_refs = list(item_rows['item_ref'])
platforms = sorted(set(item_rows['location_id']) | set(df_capacity_ledger['resource']) | set(df_usage['resource']))
category_rows = df_category[df_category['table'].str.casefold() == 'category']
categories = list(category_rows['category'])
bundle_rows = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_rows.iterrows()]
incompat_rows = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.casefold() == 'requires']
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_rows.iterrows()]
item_param = {}
for (_, row) in item_rows.iterrows():
    i = row['item_ref']
    item_param[i] = {'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id']}
fx_rows = df_fx[df_fx['table'].str.casefold() == 'fx']
fx_map = {}
for (_, row) in fx_rows.iterrows():
    fx_map[row['currency']] = (int(row['usd_cents_numerator']), int(row['denominator']))
benefit_rows = df_benefit[df_benefit['table'].str.casefold() == 'benefit']
benefit_per_unit = {}
for i in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'] == i]
    total = 0.0
    for (_, r) in rows.iterrows():
        amt = int(r['amount'])
        currency = r['currency']
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_map[currency]
        total += amt * num / denom
    benefit_per_unit[i] = total
item_fee_rows = df_item_fee[df_item_fee['table'].str.casefold() == 'item_fee']
item_fee_map = {}
for (_, row) in item_fee_rows.iterrows():
    i = row['item_ref']
    item_fee_map[i] = int(row['activation_fee_cents'])
category_param = {}
for (_, row) in category_rows.iterrows():
    c = row['category']
    category_param[c] = {'activation_fee_cents': int(row['activation_fee_cents']), 'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity'])}
bundle_bonus = {}
for (_, row) in bundle_rows.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    bundle_bonus[a, b] = int(row['bonus_cents'])
usage_rows = df_usage[df_usage['table'].str.casefold() == 'usage']
usage_per_item_platform = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    p = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().upper()
    if unit == 'GB':
        amt_mb = amt * 1000
    elif unit == 'MB':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown memory unit: {unit}')
    usage_per_item_platform[i, p] = amt_mb
cap_rows = df_capacity_ledger[df_capacity_ledger['table'].str.casefold() == 'capacity_ledger']
platform_capacity = {}
for p in platforms:
    rows = cap_rows[cap_rows['resource'] == p]
    total = rows['amount'].astype(int).sum()
    platform_capacity[p] = total
items_by_category = {c: [] for c in categories}
items_by_platform = {p: [] for p in platforms}
for i in item_refs:
    c = item_param[i]['category']
    p = item_param[i]['location_id']
    if c in items_by_category:
        items_by_category[c].append(i)
    if p in items_by_platform:
        items_by_platform[p].append(i)
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_param[i]['authorized']
    min_lot = item_param[i]['minimum_lot']
    max_order = item_param[i]['maximum_order']
    if auth == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] <= max_order * z_vars[i])
        m.addConstr(x_vars[i] >= min_lot * z_vars[i])
        m.addConstr(x_vars[i] <= max_order)
        m.addConstr(x_vars[i] >= 0)
for p in platforms:
    items_p = [i for i in items_by_platform.get(p, []) if (i, p) in usage_per_item_platform]
    m.addConstr(gp.quicksum((usage_per_item_platform[i, p] * x_vars[i] for i in items_p)) <= platform_capacity[p])
for c in categories:
    items_c = items_by_category[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_c)) >= category_param[c]['minimum_quantity'])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_c)) <= category_param[c]['maximum_quantity'])
for c in categories:
    for i in items_by_category[c]:
        m.addConstr(z_vars[i] <= w_vars[c])
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, prereq) in requires_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(z_vars[i] <= z_vars[prereq])
for (a, b) in bundle_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(b_vars[a, b] <= z_vars[a])
        m.addConstr(b_vars[a, b] <= z_vars[b])
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
benefit_expr = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in item_refs))
item_fee_expr = gp.quicksum((item_fee_map.get(i, 0) * z_vars[i] for i in item_refs))
cat_fee_expr = gp.quicksum((category_param[c]['activation_fee_cents'] * w_vars[c] for c in categories))
bundle_expr = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_expr, gp.GRB.MAXIMIZE)
m.optimize()