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
platforms = sorted(item_rows['location_id'].unique())
resources = sorted(df_capacity['resource'].unique())
category_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = category_rows['category'].tolist()
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_rows.iterrows()]
incompat_rows = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_rows.iterrows()]
fx_rows = df_fx[df_fx['table'].str.strip().str.casefold() == 'fx']
fx_dict = {}
for (_, row) in fx_rows.iterrows():
    currency = row['currency']
    fx_dict[currency] = (int(row['usd_cents_numerator']), int(row['denominator']))
benefit_rows = df_benefit[df_benefit['table'].str.strip().str.casefold() == 'benefit']
benefit_per_unit = {}
for item in item_refs:
    rows = benefit_rows[benefit_rows['item_ref'] == item]
    total = 0
    for (_, row) in rows.iterrows():
        amount = int(row['amount'])
        currency = row['currency']
        if currency not in fx_dict:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_dict[currency]
        total += amount * num // denom
    benefit_per_unit[item] = total
item_fee_rows = df_item_fee[df_item_fee['table'].str.strip().str.casefold() == 'item_fee']
item_activation_fee = {}
for (_, row) in item_fee_rows.iterrows():
    item_activation_fee[row['item_ref']] = int(row['activation_fee_cents'])
category_activation_fee = {}
category_min = {}
category_max = {}
for (_, row) in category_rows.iterrows():
    cat = row['category']
    category_activation_fee[cat] = int(row['activation_fee_cents'])
    category_min[cat] = int(row['minimum_quantity'])
    category_max[cat] = int(row['maximum_quantity'])
bundle_bonus = {}
for (_, row) in bundle_rows.iterrows():
    bundle_bonus[row['item_a'], row['item_b']] = int(row['bonus_cents'])
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
usage_per_unit = {}
for (_, row) in usage_rows.iterrows():
    item = row['item_ref']
    resource = row['resource']
    amount = int(row['amount'])
    unit = row['unit'].strip().upper()
    if unit == 'GB':
        amount_mb = amount * 1000
    elif unit == 'MB':
        amount_mb = amount
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_per_unit[item, resource] = amount_mb
capacity_rows = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_per_resource = {}
for resource in resources:
    rows = capacity_rows[capacity_rows['resource'] == resource]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().upper()
        if unit == 'GB':
            amt_mb = amt * 1000
        elif unit == 'MB':
            amt_mb = amt
        else:
            raise ValueError(f'Unknown unit {unit} for capacity')
        total += amt_mb
    capacity_per_resource[resource] = total
item_authorized = {}
item_min_lot = {}
item_max_order = {}
item_category = {}
item_location = {}
for (_, row) in item_rows.iterrows():
    item = row['item_ref']
    item_authorized[item] = int(row['authorized'])
    item_min_lot[item] = int(row['minimum_lot'])
    item_max_order[item] = int(row['maximum_order'])
    item_category[item] = row['category']
    item_location[item] = row['location_id']
category_items = {cat: [] for cat in categories}
for item in item_refs:
    cat = item_category[item]
    if cat in category_items:
        category_items[cat].append(item)
    else:
        category_items[cat] = [item]
resource_items = {res: [] for res in resources}
for item in item_refs:
    loc = item_location[item]
    if loc in resource_items:
        resource_items[loc].append(item)
    else:
        resource_items[loc] = [item]
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for item in item_refs:
    auth = item_authorized[item]
    min_lot = item_min_lot[item]
    max_order = item_max_order[item]
    if auth == 0:
        m.addConstr(x_vars[item] == 0)
        m.addConstr(y_vars[item] == 0)
    else:
        m.addConstr(x_vars[item] >= min_lot * y_vars[item])
        m.addConstr(x_vars[item] <= max_order * y_vars[item])
        m.addConstr(x_vars[item] >= y_vars[item])
        m.addConstr(x_vars[item] <= max_order * y_vars[item])
for resource in resources:
    items = resource_items.get(resource, [])
    m.addConstr(gp.quicksum((usage_per_unit.get((item, resource), 0) * x_vars[item] for item in items)) <= capacity_per_resource[resource])
for cat in categories:
    items = category_items.get(cat, [])
    m.addConstr(gp.quicksum((x_vars[item] for item in items)) >= category_min[cat])
    m.addConstr(gp.quicksum((x_vars[item] for item in items)) <= category_max[cat])
    for item in items:
        m.addConstr(y_vars[item] <= z_vars[cat])
    m.addConstr(z_vars[cat] <= gp.quicksum((y_vars[item] for item in items)))
for (a, b) in incompat_pairs:
    if a in y_vars and b in y_vars:
        m.addConstr(y_vars[a] + y_vars[b] <= 1)
for (i, prereq) in requires_pairs:
    if i in y_vars and prereq in y_vars:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for (a, b) in bundle_pairs:
    if a in y_vars and b in y_vars:
        m.addConstr(b_vars[a, b] <= y_vars[a])
        m.addConstr(b_vars[a, b] <= y_vars[b])
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
benefit_expr = gp.quicksum((benefit_per_unit[item] * x_vars[item] for item in item_refs))
item_fee_expr = gp.quicksum((item_activation_fee.get(item, 0) * y_vars[item] for item in item_refs))
cat_fee_expr = gp.quicksum((category_activation_fee[cat] * z_vars[cat] for cat in categories))
bundle_expr = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_expr, gp.GRB.MAXIMIZE)
m.optimize()