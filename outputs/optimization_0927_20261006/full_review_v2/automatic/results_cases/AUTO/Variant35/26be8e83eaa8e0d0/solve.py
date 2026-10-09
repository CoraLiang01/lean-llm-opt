import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_capacity_ledger = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_fx = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_incompatible = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_item_fee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv', dtype=str, keep_default_na=False)
item_rows = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_refs = item_rows['item_ref'].tolist()
item_category = dict(zip(item_rows['item_ref'], item_rows['category']))
item_platform = dict(zip(item_rows['item_ref'], item_rows['location_id']))
item_authorized = {row['item_ref']: int(row['authorized']) for (_, row) in item_rows.iterrows()}
item_min_lot = {row['item_ref']: int(row['minimum_lot']) for (_, row) in item_rows.iterrows()}
item_max_order = {row['item_ref']: int(row['maximum_order']) for (_, row) in item_rows.iterrows()}
platforms = sorted(set((item_platform[i] for i in item_refs)))
category_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = category_rows['category'].tolist()
category_min_qty = {row['category']: int(row['minimum_quantity']) for (_, row) in category_rows.iterrows()}
category_max_qty = {row['category']: int(row['maximum_quantity']) for (_, row) in category_rows.iterrows()}
category_activation_fee = {row['category']: int(row['activation_fee_cents']) for (_, row) in category_rows.iterrows()}
fx_rows = df_fx[df_fx['table'].str.strip().str.casefold() == 'fx']
fx_map = {row['currency']: (int(row['usd_cents_numerator']), int(row['denominator'])) for (_, row) in fx_rows.iterrows()}
benefit_rows = df_benefit[df_benefit['table'].str.strip().str.casefold() == 'benefit']
item_benefit = {i: 0 for i in item_refs}
for i in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'] == i]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        currency = row['currency']
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_map[currency]
        val = amt * num / denom
        total += val
    item_benefit[i] = int(round(total))
item_fee_rows = df_item_fee[df_item_fee['table'].str.strip().str.casefold() == 'item_fee']
item_activation_fee = {}
for i in item_refs:
    row = item_fee_rows[item_fee_rows['item_ref'] == i]
    if not row.empty:
        item_activation_fee[i] = int(row.iloc[0]['activation_fee_cents'])
    else:
        item_activation_fee[i] = 0
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = int(row['bonus_cents'])
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
item_usage = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    resource = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit == 'gb':
        amt_mb = amt * 1000
    elif unit == 'mb':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    item_usage[i, resource] = amt_mb
cap_rows = df_capacity_ledger[df_capacity_ledger['table'].str.strip().str.casefold() == 'capacity_ledger']
platform_capacity = {}
for resource in platforms:
    rows = cap_rows[cap_rows['resource'].apply(norm_str) == norm_str(resource)]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'mb':
            amt_mb = amt
        elif unit == 'gb':
            amt_mb = amt * 1000
        else:
            raise ValueError(f'Unknown unit {unit} for capacity_ledger')
        total += amt_mb
    platform_capacity[resource] = total
incompat_rows = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = set()
for (_, row) in incompat_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    incompat_pairs.add((a, b))
    incompat_pairs.add((b, a))
req_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = []
for (_, row) in req_rows.iterrows():
    item = row['item_ref']
    prereq = row['prerequisite_ref']
    prereq_pairs.append((item, prereq))
category_items = {c: [] for c in categories}
for i in item_refs:
    c = item_category[i]
    if c in category_items:
        category_items[c].append(i)
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth == 0:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauthz_{i}')
    else:
        m.addConstr(x_vars[i] <= max_order * z_vars[i], name=f'xmax_{i}')
        m.addConstr(x_vars[i] >= min_lot * z_vars[i], name=f'xmin_{i}')
        m.addConstr(x_vars[i] <= max_order, name=f'xmax2_{i}')
for resource in platforms:
    items_on_platform = [i for i in item_refs if norm_str(item_platform[i]) == norm_str(resource)]
    m.addConstr(gp.quicksum((item_usage.get((i, resource), 0) * x_vars[i] for i in items_on_platform)) <= platform_capacity[resource], name=f'memcap_{resource}')
for c in categories:
    items = category_items[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) >= category_min_qty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) <= category_max_qty[c], name=f'catmax_{c}')
for c in categories:
    items = category_items[c]
    m.addConstr(gp.quicksum((z_vars[i] for i in items)) >= w_vars[c], name=f'catw_lb_{c}')
    m.addConstr(w_vars[c] <= 1, name=f'catw_ub_{c}')
for (a, b) in incompat_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incompat_{a}_{b}')
for (item, prereq) in prereq_pairs:
    if item in item_refs and prereq in item_refs:
        m.addConstr(z_vars[item] <= z_vars[prereq], name=f'prereq_{item}_{prereq}')
for (a, b) in bundle_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(b_vars[a, b] <= z_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(b_vars[a, b] <= z_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle0_{a}_{b}')
benefit_expr = gp.quicksum((item_benefit[i] * x_vars[i] for i in item_refs))
item_fee_expr = gp.quicksum((item_activation_fee[i] * z_vars[i] for i in item_refs))
cat_fee_expr = gp.quicksum((category_activation_fee[c] * w_vars[c] for c in categories))
bundle_expr = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_expr, gp.GRB.MAXIMIZE)
m.optimize()