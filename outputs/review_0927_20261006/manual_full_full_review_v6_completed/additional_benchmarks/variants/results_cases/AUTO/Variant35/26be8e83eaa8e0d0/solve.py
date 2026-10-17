import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
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
platforms = sorted(set(item_rows['location_id'].unique()) | set(df_capacity['resource'].unique()) | set(df_usage['resource'].unique()))
category_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = category_rows['category'].tolist()
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for _, row in bundle_rows.iterrows()]
incompat_rows = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for _, row in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for _, row in requires_rows.iterrows()]
fx_rows = df_fx[df_fx['table'].str.strip().str.casefold() == 'fx']
fx_dict = {}
for _, row in fx_rows.iterrows():
    currency = row['currency']
    num = int(row['usd_cents_numerator'])
    denom = int(row['denominator'])
    fx_dict[currency] = (num, denom)
benefit_rows = df_benefit[df_benefit['table'].str.strip().str.casefold() == 'benefit']
benefit_per_unit = {}
for item in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'] == item]
    total = 0.0
    for _, r in rows.iterrows():
        amt = int(r['amount'])
        curr = r['currency']
        if curr not in fx_dict:
            raise ValueError(f'Missing FX rate for currency {curr}')
        num, denom = fx_dict[curr]
        total += amt * num / denom
    benefit_per_unit[item] = int(round(total))
item_fee_rows = df_item_fee[df_item_fee['table'].str.strip().str.casefold() == 'item_fee']
item_fee_dict = {}
for _, row in item_fee_rows.iterrows():
    item_fee_dict[row['item_ref']] = int(row['activation_fee_cents'])
cat_fee_dict = {}
for _, row in category_rows.iterrows():
    cat_fee_dict[row['category']] = int(row['activation_fee_cents'])
bundle_bonus_dict = {}
for _, row in bundle_rows.iterrows():
    bundle_bonus_dict[row['item_a'], row['item_b']] = int(row['bonus_cents'])
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
usage_per_unit = {}
for _, row in usage_rows.iterrows():
    item = row['item_ref']
    plat = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().upper()
    if unit == 'GB':
        amt_mb = amt * 1000
    elif unit == 'MB':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_per_unit[item, plat] = amt_mb
cap_rows = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
platform_capacity = {}
for plat in platforms:
    plat_rows = cap_rows[cap_rows['resource'] == plat]
    total = 0
    for _, row in plat_rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().upper()
        if unit == 'GB':
            amt_mb = amt * 1000
        elif unit == 'MB':
            amt_mb = amt
        else:
            raise ValueError(f'Unknown unit {unit} for capacity')
        total += amt_mb
    platform_capacity[plat] = total
item_auth = {}
item_min_lot = {}
item_max_order = {}
item_category = {}
item_platform = {}
for _, row in item_rows.iterrows():
    item = row['item_ref']
    item_auth[item] = int(row['authorized'])
    item_min_lot[item] = int(row['minimum_lot'])
    item_max_order[item] = int(row['maximum_order'])
    item_category[item] = row['category']
    item_platform[item] = row['location_id']
cat_min_qty = {}
cat_max_qty = {}
for _, row in category_rows.iterrows():
    cat = row['category']
    cat_min_qty[cat] = int(row['minimum_quantity'])
    cat_max_qty[cat] = int(row['maximum_quantity'])
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if item_auth[i] == 1:
        m.addConstr(x_vars[i] >= y_vars[i] * item_min_lot[i])
        m.addConstr(x_vars[i] <= y_vars[i] * item_max_order[i])
        m.addConstr(x_vars[i] >= y_vars[i])
        m.addConstr(x_vars[i] <= item_max_order[i] * y_vars[i])
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= cat_min_qty[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= cat_max_qty[c])
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for plat in platforms:
    items_on_p = [i for i in item_refs if item_platform[i] == plat]
    m.addConstr(gp.quicksum((usage_per_unit.get((i, plat), 0) * x_vars[i] for i in items_on_p)) <= platform_capacity.get(plat, 0))
for i, j in incompat_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for i, prereq in prereq_pairs:
    if i in y_vars and prereq in y_vars:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for i, j in bundle_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(b_vars[i, j] <= y_vars[i])
        m.addConstr(b_vars[i, j] <= y_vars[j])
        m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(b_vars[i, j] == 0)
obj_benefit = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in item_refs))
obj_item_fee = gp.quicksum((item_fee_dict.get(i, 0) * y_vars[i] for i in item_refs))
obj_cat_fee = gp.quicksum((cat_fee_dict[c] * z_vars[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus_dict[i, j] * b_vars[i, j] for i, j in bundle_pairs))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()