import gurobipy as gp
import pandas as pd
import numpy as np
import re

def parse_date(s):
    return s

def select_latest_valid(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'record_id', 'revision'], ascending=[True, True, False])
    df = df.drop_duplicates(['table', 'record_id'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def select_latest_valid_category(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'category', 'revision'], ascending=[True, True, False])
    df = df.drop_duplicates(['table', 'category'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def select_latest_valid_itemref(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'item_ref', 'revision'], ascending=[True, True, False])
    df = df.drop_duplicates(['table', 'item_ref'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def select_latest_valid_pair(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'item_a', 'item_b', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(['table', 'item_a', 'item_b'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def select_latest_valid_requires(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'item_ref', 'prerequisite_ref', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(['table', 'item_ref', 'prerequisite_ref'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def select_latest_valid_capacity(df, as_of_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'].apply(parse_date) <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['table', 'resource', 'entry', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(['table', 'resource', 'entry'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df
as_of_date = '2026-03-12'
tenant = 'NORTH'
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = {}
for path in paths:
    key = re.sub('.*/', '', path)
    dfs[key] = pd.read_csv(path, dtype=str, keep_default_na=False)
table_map = {}
for df in dfs.values():
    if 'table' in df.columns:
        for t in df['table'].unique():
            if t not in table_map:
                table_map[t] = []
            table_map[t].append(df)
item_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('item').any():
        item_tables.append(df)
item_df = pd.concat(item_tables, ignore_index=True)
item_df = select_latest_valid(item_df, as_of_date, tenant)
item_df = item_df[item_df['item_ref'].str.strip() != '']
item_df = item_df[item_df['category'].str.strip() != '']
items = sorted(item_df['item_ref'].unique())
item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
item_authorized = dict(zip(item_df['item_ref'], item_df['authorized'].astype(float)))
item_min_lot = dict(zip(item_df['item_ref'], item_df['minimum_lot'].astype(float)))
item_max_order = dict(zip(item_df['item_ref'], item_df['maximum_order'].astype(float)))
category_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('category').any():
        category_tables.append(df)
if category_tables:
    category_df = pd.concat(category_tables, ignore_index=True)
    category_df = select_latest_valid_category(category_df, as_of_date, tenant)
    category_df = category_df[category_df['category'].str.strip() != '']
    categories = sorted(category_df['category'].unique())
    category_min_qty = dict(zip(category_df['category'], category_df['minimum_quantity'].astype(float)))
    category_max_qty = dict(zip(category_df['category'], category_df['maximum_quantity'].astype(float)))
    category_activation_fee = dict(zip(category_df['category'], category_df['activation_fee_cents'].astype(float)))
else:
    categories = []
    category_min_qty = {}
    category_max_qty = {}
    category_activation_fee = {}
benefit_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('benefit').any():
        benefit_tables.append(df)
if benefit_tables:
    benefit_df = pd.concat(benefit_tables, ignore_index=True)
    benefit_df = select_latest_valid_itemref(benefit_df, as_of_date, tenant)
    benefit_df = benefit_df[benefit_df['item_ref'].str.strip() != '']
    benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='raise')
    per_unit_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
else:
    per_unit_benefit = {}
item_fee_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('item_fee').any():
        item_fee_tables.append(df)
if item_fee_tables:
    item_fee_df = pd.concat(item_fee_tables, ignore_index=True)
    item_fee_df = select_latest_valid_itemref(item_fee_df, as_of_date, tenant)
    item_fee_df = item_fee_df[item_fee_df['item_ref'].str.strip() != '']
    item_activation_fee = dict(zip(item_fee_df['item_ref'], pd.to_numeric(item_fee_df['activation_fee_cents'], errors='coerce').fillna(0)))
else:
    item_activation_fee = {}
bundle_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('bundle').any():
        bundle_tables.append(df)
if bundle_tables:
    bundle_df = pd.concat(bundle_tables, ignore_index=True)
    bundle_df = select_latest_valid_pair(bundle_df, as_of_date, tenant)
    bundle_df = bundle_df[(bundle_df['item_a'].str.strip() != '') & (bundle_df['item_b'].str.strip() != '')]
    bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'], errors='coerce').fillna(0)
    bundles = []
    bundle_bonus = {}
    for (_, row) in bundle_df.iterrows():
        a = row['item_a']
        b = row['item_b']
        bundles.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
else:
    bundles = []
    bundle_bonus = {}
usage_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('usage').any():
        usage_tables.append(df)
if usage_tables:
    usage_df = pd.concat(usage_tables, ignore_index=True)
    usage_df = select_latest_valid_itemref(usage_df, as_of_date, tenant)
    usage_df = usage_df[(usage_df['item_ref'].str.strip() != '') & (usage_df['resource'].str.strip() != '')]
    usage_df['amount'] = pd.to_numeric(usage_df['amount'], errors='raise')
    usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()

    def convert_usage(row):
        amt = row['amount']
        unit = row['unit']
        if unit == 'liter':
            return amt * 1000
        elif unit == 'ml':
            return amt
        elif unit == 'hour':
            return amt * 60
        elif unit == 'minute':
            return amt
        elif unit == 'kwh':
            return amt * 1000
        elif unit == 'wh':
            return amt
        else:
            raise ValueError(f'Unknown unit in usage: {unit}')
    usage_df['amount_base'] = usage_df.apply(convert_usage, axis=1)
    resource_usage = {}
    for (_, row) in usage_df.iterrows():
        resource_usage[row['item_ref'], row['resource']] = row['amount_base']
    resources = sorted(usage_df['resource'].unique())
else:
    resource_usage = {}
    resources = []
capacity_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('capacity_ledger').any():
        capacity_tables.append(df)
if capacity_tables:
    capacity_df = pd.concat(capacity_tables, ignore_index=True)
    capacity_df = select_latest_valid_capacity(capacity_df, as_of_date, tenant)
    capacity_df = capacity_df[(capacity_df['resource'].str.strip() != '') & (capacity_df['amount'].str.strip() != '')]
    capacity_df['amount'] = pd.to_numeric(capacity_df['amount'], errors='raise')
    capacity_df['unit'] = capacity_df['unit'].str.strip().str.casefold()

    def convert_capacity(row):
        amt = row['amount']
        unit = row['unit']
        if unit == 'liter':
            return amt * 1000
        elif unit == 'ml':
            return amt
        elif unit == 'hour':
            return amt * 60
        elif unit == 'minute':
            return amt
        elif unit == 'kwh':
            return amt * 1000
        elif unit == 'wh':
            return amt
        else:
            raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
    capacity_df['amount_base'] = capacity_df.apply(convert_capacity, axis=1)
    resource_capacity = capacity_df.groupby('resource')['amount_base'].sum().to_dict()
else:
    resource_capacity = {}
incompatible_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('incompatible').any():
        incompatible_tables.append(df)
if incompatible_tables:
    incompatible_df = pd.concat(incompatible_tables, ignore_index=True)
    incompatible_df = select_latest_valid_pair(incompatible_df, as_of_date, tenant)
    incompatible_df = incompatible_df[(incompatible_df['item_a'].str.strip() != '') & (incompatible_df['item_b'].str.strip() != '')]
    incompatible_pairs = set()
    for (_, row) in incompatible_df.iterrows():
        a = row['item_a']
        b = row['item_b']
        incompatible_pairs.add((a, b))
        incompatible_pairs.add((b, a))
else:
    incompatible_pairs = set()
requires_tables = []
for fname in dfs:
    df = dfs[fname]
    if 'table' in df.columns and df['table'].str.strip().str.casefold().eq('requires').any():
        requires_tables.append(df)
if requires_tables:
    requires_df = pd.concat(requires_tables, ignore_index=True)
    requires_df = select_latest_valid_requires(requires_df, as_of_date, tenant)
    requires_df = requires_df[(requires_df['item_ref'].str.strip() != '') & (requires_df['prerequisite_ref'].str.strip() != '')]
    requires_pairs = set()
    for (_, row) in requires_df.iterrows():
        i = row['item_ref']
        j = row['prerequisite_ref']
        requires_pairs.add((i, j))
else:
    requires_pairs = set()
category_items = {}
for i in items:
    g = item_to_category[i]
    if g not in category_items:
        category_items[g] = []
    category_items[g].append(i)
for i in items:
    if i not in per_unit_benefit:
        per_unit_benefit[i] = 0
    if i not in item_activation_fee:
        item_activation_fee[i] = 0
for g in categories:
    if g not in category_min_qty:
        category_min_qty[g] = 0
    if g not in category_max_qty:
        category_max_qty[g] = float('inf')
    if g not in category_activation_fee:
        category_activation_fee[g] = 0
for r in resources:
    if r not in resource_capacity:
        resource_capacity[r] = float('inf')
m = gp.Model('InventoryReplenishment')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(list(range(len(bundles))), vtype=gp.GRB.BINARY, name='')
for i in items:
    auth = item_authorized.get(i, 0)
    min_lot = item_min_lot.get(i, 0)
    max_order = item_max_order.get(i, 0)
    if auth == 0:
        m.addConstr(q_vars[i] == 0, name=f'auth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'z0_{i}')
    else:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= max_order * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(q_vars[i] <= max_order, name=f'maxorder2_{i}')
        m.addConstr(q_vars[i] >= 0, name=f'nonneg_{i}')
for g in categories:
    items_in_g = category_items.get(g, [])
    if items_in_g:
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= category_min_qty[g], name=f'cat_min_{g}')
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= category_max_qty[g], name=f'cat_max_{g}')
    for i in items_in_g:
        m.addConstr(z_vars[i] <= w_vars[g], name=f'catact_{g}_{i}')
for r in resources:
    usage_terms = []
    for i in items:
        amt = resource_usage.get((i, r), 0)
        usage_terms.append(q_vars[i] * amt)
    m.addConstr(gp.quicksum(usage_terms) <= resource_capacity[r], name=f'res_cap_{r}')
for (i, j) in incompatible_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] <= z_vars[j], name=f'req_{i}_{j}')
for (bidx, (a, b)) in enumerate(bundles):
    if a in items and b in items:
        m.addConstr(s_vars[bidx] <= z_vars[a], name=f'bundle1_{bidx}')
        m.addConstr(s_vars[bidx] <= z_vars[b], name=f'bundle2_{bidx}')
        m.addConstr(s_vars[bidx] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{bidx}')
    else:
        m.addConstr(s_vars[bidx] == 0, name=f'bundle0_{bidx}')
obj = gp.quicksum((per_unit_benefit[i] * q_vars[i] for i in items))
obj -= gp.quicksum((item_activation_fee[i] * z_vars[i] for i in items))
obj -= gp.quicksum((category_activation_fee[g] * w_vars[g] for g in categories))
obj += gp.quicksum((bundle_bonus[bundles[bidx]] * s_vars[bidx] for bidx in range(len(bundles))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()