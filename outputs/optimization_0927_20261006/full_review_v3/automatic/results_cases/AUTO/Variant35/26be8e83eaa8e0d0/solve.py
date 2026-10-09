import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
(df_benefit, df_bundle, df_capacity_ledger, df_category, df_fx, df_identity, df_incompatible, df_item, df_item_fee, df_market, df_requires, df_usage) = dfs
platforms = ['PC', 'CONSOLE', 'MOBILE']
item_rows = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_refs = item_rows['item_ref'].tolist()
category_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = category_rows['category'].tolist()
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundles = [(row['item_a'], row['item_b']) for (_, row) in bundle_rows.iterrows()]
incompat_rows = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_rows.iterrows()]
fx_rows = df_fx[df_fx['table'].str.strip().str.casefold() == 'fx']
fx_map = {}
for (_, row) in fx_rows.iterrows():
    currency = row['currency']
    usd_cents_numerator = int(row['usd_cents_numerator'])
    denominator = int(row['denominator'])
    fx_map[currency] = (usd_cents_numerator, denominator)
benefit_rows = df_benefit[df_benefit['table'].str.strip().str.casefold() == 'benefit']
benefit_per_unit = {}
for item in item_refs:
    item_benefits = benefit_rows[benefit_rows['item_ref'] == item]
    total = 0
    for (_, row) in item_benefits.iterrows():
        amount = int(row['amount'])
        currency = row['currency']
        if currency not in fx_map:
            raise ValueError(f'Currency {currency} not found in fx table for item {item}')
        (usd_cents_numerator, denominator) = fx_map[currency]
        value = amount * usd_cents_numerator / denominator
        total += value
    benefit_per_unit[item] = int(round(total))
item_fee_rows = df_item_fee[df_item_fee['table'].str.strip().str.casefold() == 'item_fee']
item_fee_dict = {}
for (_, row) in item_fee_rows.iterrows():
    item_fee_dict[row['item_ref']] = int(row['activation_fee_cents'])
bundle_bonus_dict = {}
for (_, row) in bundle_rows.iterrows():
    bundle_bonus_dict[row['item_a'], row['item_b']] = int(row['bonus_cents'])
category_min_qty = {}
category_max_qty = {}
category_activation_fee = {}
for (_, row) in category_rows.iterrows():
    c = row['category']
    category_min_qty[c] = int(row['minimum_quantity'])
    category_max_qty[c] = int(row['maximum_quantity'])
    category_activation_fee[c] = int(row['activation_fee_cents'])
item_authorized = {}
item_min_lot = {}
item_max_order = {}
item_category = {}
item_platform = {}
for (_, row) in item_rows.iterrows():
    i = row['item_ref']
    item_authorized[i] = int(row['authorized'])
    item_min_lot[i] = int(row['minimum_lot'])
    item_max_order[i] = int(row['maximum_order'])
    item_category[i] = row['category']
    item_platform[i] = row['location_id']
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
item_usage = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    resource = row['resource']
    amount_gb = int(row['amount'])
    unit = row['unit']
    if unit.strip().upper() == 'GB':
        amount_mb = amount_gb * 1000
    elif unit.strip().upper() == 'MB':
        amount_mb = amount_gb
    else:
        raise ValueError(f'Unknown unit {unit} for item {i}')
    item_usage[i, resource] = amount_mb
cap_rows = df_capacity_ledger[df_capacity_ledger['table'].str.strip().str.casefold() == 'capacity_ledger']
platform_capacity = {}
for p in platforms:
    rows = cap_rows[cap_rows['resource'] == p]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit']
        if unit.strip().upper() == 'MB':
            total += amt
        elif unit.strip().upper() == 'GB':
            total += amt * 1000
        else:
            raise ValueError(f'Unknown unit {unit} in capacity_ledger for {p}')
    platform_capacity[p] = total
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if item_authorized[i] == 0:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= item_min_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= item_max_order[i] * y_vars[i], name=f'maxorder_{i}')
for i in item_refs:
    m.addConstr(x_vars[i] <= item_max_order[i] * y_vars[i], name=f'link1_{i}')
    m.addConstr(x_vars[i] >= y_vars[i], name=f'link2_{i}')
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    if items_in_c:
        m.addConstr(gp.quicksum((y_vars[i] for i in items_in_c)) <= len(items_in_c) * z_vars[c], name=f'cat_act_up_{c}')
        m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)), name=f'cat_act_lo_{c}')
    else:
        m.addConstr(z_vars[c] == 0, name=f'cat_empty_{c}')
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    if items_in_c:
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_min_qty[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_max_qty[c], name=f'cat_max_{c}')
for p in platforms:
    items_on_p = [i for i in item_refs if item_platform[i] == p]
    m.addConstr(gp.quicksum((item_usage.get((i, p), 0) * x_vars[i] for i in items_on_p)) <= platform_capacity[p], name=f'cap_{p}')
for (a, b) in incompat_pairs:
    if a in y_vars and b in y_vars:
        m.addConstr(y_vars[a] + y_vars[b] <= 1, name=f'incompat_{a}_{b}')
for (i, pre) in requires_pairs:
    if i in y_vars and pre in y_vars:
        m.addConstr(y_vars[i] <= y_vars[pre], name=f'req_{i}_{pre}')
for (a, b) in bundles:
    if a in y_vars and b in y_vars:
        m.addConstr(w_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(w_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(w_vars[a, b] == 0, name=f'bundle_zero_{a}_{b}')
obj_benefit = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in item_refs))
obj_item_fee = gp.quicksum((item_fee_dict.get(i, 0) * y_vars[i] for i in item_refs))
obj_cat_fee = gp.quicksum((category_activation_fee[c] * z_vars[c] for c in categories))
obj_bundle_bonus = gp.quicksum((bundle_bonus_dict.get((a, b), 0) * w_vars[a, b] for (a, b) in bundles))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle_bonus, gp.GRB.MAXIMIZE)
m.optimize()