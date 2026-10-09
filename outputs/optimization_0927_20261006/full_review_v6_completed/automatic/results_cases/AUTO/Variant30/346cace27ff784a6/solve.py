import gurobipy as gp
import pandas as pd
import numpy as np
import re

def filter_latest(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    df['revision'] = df['revision'].astype(int)
    df = df.sort_values(['tenant', 'table', 'record_id', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=['tenant', 'table', 'record_id'] + [c for c in df.columns if c not in ['revision']], keep='first')
    df = df.drop_duplicates(subset=['tenant', 'table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_latest_no_tenant(df, asof_date):
    df = df[df['effective_date'] <= asof_date]
    df['revision'] = df['revision'].astype(int)
    df = df.sort_values(['table', 'record_id', 'revision'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['table', 'record_id'] + [c for c in df.columns if c not in ['revision']], keep='first')
    df = df.drop_duplicates(subset=['table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df
asof_date = '2026-03-12'
tenant = 'NORTH'
df_requires_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_requires_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv', dtype=str, keep_default_na=False)
df_requires = pd.concat([df_requires_1, df_requires_2], ignore_index=True)
df_requires = filter_latest(df_requires, asof_date, tenant)
requires_pairs = set()
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if i and k:
        requires_pairs.add((i, k))
df_identity_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_identity_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', dtype=str, keep_default_na=False)
df_identity = pd.concat([df_identity_1, df_identity_2], ignore_index=True)
df_identity = filter_latest(df_identity, asof_date, tenant)
identity_map = {}
for (_, row) in df_identity.iterrows():
    if row.get('kind', '').strip().casefold() == 'item':
        identity_map[row['ref']] = row['entity_id']
df_item_fee_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_item_fee_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', dtype=str, keep_default_na=False)
df_item_fee = pd.concat([df_item_fee_1, df_item_fee_2], ignore_index=True)
df_item_fee = filter_latest(df_item_fee, asof_date, tenant)
item_fee_map = {}
for (_, row) in df_item_fee.iterrows():
    if row['item_ref']:
        item_fee_map[row['item_ref']] = float(row['activation_fee_cents'])
df_bundle_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_bundle_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
df_bundle = pd.concat([df_bundle_1, df_bundle_2], ignore_index=True)
df_bundle = filter_latest(df_bundle, asof_date, tenant)
bundle_list = []
bundle_bonus = {}
for (idx, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a and b:
        key = (a, b)
        bundle_list.append(key)
        bundle_bonus[key] = float(row['bonus_cents'])
df_benefit_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_benefit_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', dtype=str, keep_default_na=False)
df_benefit_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', dtype=str, keep_default_na=False)
df_benefit = pd.concat([df_benefit_1, df_benefit_2, df_benefit_3], ignore_index=True)
df_benefit = filter_latest(df_benefit, asof_date, tenant)
benefit_map = {}
for (item_ref, group) in df_benefit.groupby('item_ref'):
    benefit_map[item_ref] = group['amount_cents'].astype(float).sum()
df_usage_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_usage_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', dtype=str, keep_default_na=False)
df_usage_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', dtype=str, keep_default_na=False)
df_usage = pd.concat([df_usage_1, df_usage_2, df_usage_3], ignore_index=True)
df_usage = filter_latest(df_usage, asof_date, tenant)
usage_per_unit = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    if not i or not r:
        continue
    amt = float(row['amount'])
    unit = row['unit'].strip().casefold()
    if i not in usage_per_unit:
        usage_per_unit[i] = {}
    if unit == 'liter':
        amt = amt * 1000
        unit = 'ml'
    elif unit == 'hour':
        amt = amt * 60
        unit = 'minute'
    elif unit == 'kwh':
        amt = amt * 1000
        unit = 'wh'
    usage_per_unit[i][r] = (amt, unit)
df_capacity_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_capacity_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', dtype=str, keep_default_na=False)
df_capacity = pd.concat([df_capacity_1, df_capacity_2], ignore_index=True)
df_capacity = filter_latest(df_capacity, asof_date, tenant)
total_capacity = {}
for (_, row) in df_capacity.iterrows():
    r = row['resource']
    if not r:
        continue
    amt = float(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit == 'liter':
        amt = amt * 1000
        unit = 'ml'
    elif unit == 'hour':
        amt = amt * 60
        unit = 'minute'
    elif unit == 'kwh':
        amt = amt * 1000
        unit = 'wh'
    elif unit == 'ml' or unit == 'minute' or unit == 'wh':
        pass
    else:
        raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
    if r not in total_capacity:
        total_capacity[r] = 0.0
    total_capacity[r] += amt
df_item_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
df_item_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', dtype=str, keep_default_na=False)
df_item_3 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', dtype=str, keep_default_na=False)
df_item = pd.concat([df_item_1, df_item_2], ignore_index=True)
df_item = filter_latest(df_item, asof_date, tenant)
df_item = df_item[df_item['item_ref'].notnull() & (df_item['item_ref'] != '')]
item_set = set(df_item['item_ref'])
item_authorized = {}
item_min_lot = {}
item_max_order = {}
item_category = {}
for (_, row) in df_item.iterrows():
    i = row['item_ref']
    item_authorized[i] = int(float(row['authorized']))
    item_min_lot[i] = int(float(row['minimum_lot']))
    item_max_order[i] = int(float(row['maximum_order']))
    item_category[i] = row['category']
df_cat_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', dtype=str, keep_default_na=False)
df_cat_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', dtype=str, keep_default_na=False)
df_cat = pd.concat([df_cat_1, df_cat_2], ignore_index=True)
df_cat = filter_latest(df_cat, asof_date, tenant)
category_set = set(df_cat['category'])
cat_min_qty = {}
cat_max_qty = {}
cat_fee = {}
for (_, row) in df_cat.iterrows():
    g = row['category']
    cat_min_qty[g] = int(float(row['minimum_quantity']))
    cat_max_qty[g] = int(float(row['maximum_quantity']))
    cat_fee[g] = float(row['activation_fee_cents'])
df_incomp_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_incomp_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', dtype=str, keep_default_na=False)
df_incomp = pd.concat([df_incomp_1, df_incomp_2], ignore_index=True)
df_incomp = filter_latest(df_incomp, asof_date, tenant)
incomp_pairs = set()
for (_, row) in df_incomp.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a and b:
        incomp_pairs.add(tuple(sorted((a, b))))
bundle_list = [b for b in bundle_list if b[0] in item_set and b[1] in item_set]
requires_pairs = set([(i, k) for (i, k) in requires_pairs if i in item_set and k in item_set])
incomp_pairs = set([(a, b) for (a, b) in incomp_pairs if a in item_set and b in item_set])
resource_set = set(total_capacity.keys())
for i in list(usage_per_unit.keys()):
    usage_per_unit[i] = {r: usage_per_unit[i][r] for r in usage_per_unit[i] if r in resource_set}
    if not usage_per_unit[i]:
        del usage_per_unit[i]
for i in item_set:
    if i not in benefit_map:
        benefit_map[i] = 0.0
    if i not in item_fee_map:
        item_fee_map[i] = 0.0
for g in category_set:
    if g not in cat_min_qty:
        cat_min_qty[g] = 0
    if g not in cat_max_qty:
        cat_max_qty[g] = 999999
    if g not in cat_fee:
        cat_fee[g] = 0.0
for i in item_set:
    if i not in item_category:
        raise ValueError(f'Item {i} missing category')
for i in item_set:
    if i not in item_min_lot or i not in item_max_order:
        raise ValueError(f'Item {i} missing min_lot or max_order')
for i in item_set:
    if i not in item_authorized:
        raise ValueError(f'Item {i} missing authorized')
for r in resource_set:
    if r not in total_capacity:
        raise ValueError(f'Resource {r} missing total_capacity')
for i in item_set:
    if i not in usage_per_unit:
        usage_per_unit[i] = {}
    for r in resource_set:
        if r not in usage_per_unit[i]:
            usage_per_unit[i][r] = (0.0, None)
for b in bundle_list:
    if b not in bundle_bonus:
        bundle_bonus[b] = 0.0
cat_items = {g: set() for g in category_set}
for i in item_set:
    g = item_category[i]
    cat_items[g].add(i)
items = sorted(item_set)
categories = sorted(category_set)
resources = sorted(resource_set)
bundles = bundle_list
incompat = sorted(incomp_pairs)
requires = sorted(requires_pairs)
m = gp.Model('vehicle_dealer_replenishment')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
obj = gp.LinExpr()
obj += gp.quicksum((benefit_map[i] * q_vars[i] for i in items))
obj -= gp.quicksum((item_fee_map[i] * z_vars[i] for i in items))
obj -= gp.quicksum((cat_fee[g] * y_vars[g] for g in categories))
obj += gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in items:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth > 0:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i])
        m.addConstr(q_vars[i] <= max_order * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for g in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_items[g])) >= cat_min_qty[g])
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_items[g])) <= cat_max_qty[g])
    for i in cat_items[g]:
        m.addConstr(z_vars[i] <= y_vars[g])
    m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in cat_items[g])))
for r in resources:
    m.addConstr(gp.quicksum((usage_per_unit[i][r][0] * q_vars[i] for i in items)) <= total_capacity[r])
for (a, b) in incompat:
    m.addConstr(z_vars[a] + z_vars[b] <= 1)
for (i, k) in requires:
    m.addConstr(z_vars[i] <= z_vars[k])
for b in bundles:
    (i, j) = b
    m.addConstr(w_vars[b] <= z_vars[i])
    m.addConstr(w_vars[b] <= z_vars[j])
    m.addConstr(w_vars[b] >= z_vars[i] + z_vars[j] - 1)
m.optimize()