import gurobipy as gp
import pandas as pd
import numpy as np
import re

def filter_latest_revision(df, asof_date, key_cols):
    """
    For a DataFrame with revisioned records, filter to:
    - Only rows with effective_date <= asof_date
    - For each (key_cols), keep the row with the highest revision
    - Drop rows where action == DELETE
    - Deduplicate retransmissions (identical (key_cols, revision))
    """
    df = df[df['effective_date'] <= asof_date].copy()
    df['revision'] = df['revision'].astype(int)
    df = df.drop_duplicates(subset=key_cols + ['revision'])
    df = df.sort_values(key_cols + ['revision'], ascending=True)
    idx = df.groupby(key_cols)['revision'].idxmax()
    df = df.loc[idx]
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def safe_float(series):
    return pd.to_numeric(series, errors='coerce')

def safe_int(series):
    return pd.to_numeric(series, errors='coerce').astype('Int64')
asof_date = '2026-03-12'
tenant = 'NORTH'
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = {}
for path in paths:
    fname = path.split('/')[-1]
    dfs[fname] = pd.read_csv(path, dtype=str, keep_default_na=False)
requires_df = pd.concat([dfs['export_01.csv'], dfs['export_24.csv']], ignore_index=True)
identity_df = pd.concat([dfs['export_02.csv'], dfs['export_22.csv']], ignore_index=True)
item_fee_df = pd.concat([dfs['export_03.csv'], dfs['export_17.csv']], ignore_index=True)
bundle_df = pd.concat([dfs['export_04.csv'], dfs['export_11.csv']], ignore_index=True)
benefit_df = pd.concat([dfs['export_05.csv'], dfs['export_12.csv'], dfs['export_15.csv']], ignore_index=True)
usage_df = pd.concat([dfs['export_06.csv'], dfs['export_19.csv'], dfs['export_23.csv']], ignore_index=True)
incompatible_df = pd.concat([dfs['export_07.csv'], dfs['export_14.csv']], ignore_index=True)
capacity_ledger_df = pd.concat([dfs['export_08.csv'], dfs['export_20.csv']], ignore_index=True)
item_df = pd.concat([dfs['export_10.csv'], dfs['export_13.csv']], ignore_index=True)
market_df = pd.concat([dfs['export_09.csv'], dfs['export_21.csv']], ignore_index=True)
category_df = pd.concat([dfs['export_16.csv'], dfs['export_18.csv']], ignore_index=True)
requires_df = requires_df[requires_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
requires_df = filter_latest_revision(requires_df, asof_date, ['table', 'tenant', 'record_id'])
identity_df = identity_df[identity_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
identity_df = filter_latest_revision(identity_df, asof_date, ['table', 'tenant', 'record_id'])
item_fee_df = item_fee_df[item_fee_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
item_fee_df = filter_latest_revision(item_fee_df, asof_date, ['table', 'tenant', 'record_id'])
bundle_df = bundle_df[bundle_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
bundle_df = filter_latest_revision(bundle_df, asof_date, ['table', 'tenant', 'record_id'])
benefit_df = benefit_df[benefit_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
benefit_df = filter_latest_revision(benefit_df, asof_date, ['table', 'tenant', 'record_id'])
usage_df = usage_df[usage_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
usage_df = filter_latest_revision(usage_df, asof_date, ['table', 'tenant', 'record_id'])
incompatible_df = incompatible_df[incompatible_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
incompatible_df = filter_latest_revision(incompatible_df, asof_date, ['table', 'tenant', 'record_id'])
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
capacity_ledger_df = filter_latest_revision(capacity_ledger_df, asof_date, ['table', 'tenant', 'record_id'])
item_df = item_df[item_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
item_df = filter_latest_revision(item_df, asof_date, ['table', 'tenant', 'record_id'])
category_df = category_df[category_df['tenant'].str.strip().str.casefold() == tenant.casefold()]
category_df = filter_latest_revision(category_df, asof_date, ['table', 'tenant', 'record_id'])
item_df = item_df[item_df['item_ref'].notna() & (item_df['item_ref'] != '')]
items = sorted(item_df['item_ref'].unique())
category_df = category_df[category_df['category'].notna() & (category_df['category'] != '')]
categories = sorted(category_df['category'].unique())
item_to_category = item_df.set_index('item_ref')['category'].to_dict()
category_to_items = {g: [] for g in categories}
for i in items:
    g = item_to_category.get(i)
    if g in categories:
        category_to_items[g].append(i)
benefit_df = benefit_df[benefit_df['item_ref'].notna() & (benefit_df['item_ref'] != '')]
benefit_df['amount_cents'] = safe_float(benefit_df['amount_cents'])
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in item_benefit:
        item_benefit[i] = 0.0
item_fee_df = item_fee_df[item_fee_df['item_ref'].notna() & (item_fee_df['item_ref'] != '')]
item_fee_df['activation_fee_cents'] = safe_float(item_fee_df['activation_fee_cents'])
item_activation_fee = item_fee_df.drop_duplicates('item_ref').set_index('item_ref')['activation_fee_cents'].to_dict()
for i in items:
    if i not in item_activation_fee:
        item_activation_fee[i] = 0.0
category_df['activation_fee_cents'] = safe_float(category_df['activation_fee_cents'])
category_activation_fee = category_df.drop_duplicates('category').set_index('category')['activation_fee_cents'].to_dict()
for g in categories:
    if g not in category_activation_fee:
        category_activation_fee[g] = 0.0
bundle_df = bundle_df[bundle_df['item_a'].notna() & (bundle_df['item_a'] != '') & bundle_df['item_b'].notna() & (bundle_df['item_b'] != '')]
bundle_df['bonus_cents'] = safe_float(bundle_df['bonus_cents'])
bundle_pairs = []
bundle_bonus = {}
for idx, row in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        key = tuple(sorted([a, b]))
        bundle_pairs.append(key)
        bundle_bonus[key] = float(row['bonus_cents'])
bundle_pairs = sorted(set(bundle_pairs))
item_df['minimum_lot'] = safe_float(item_df['minimum_lot'])
item_df['maximum_order'] = safe_float(item_df['maximum_order'])
item_df['authorized'] = safe_float(item_df['authorized'])
item_minimum_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_maximum_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
for i in items:
    if i not in item_minimum_lot:
        item_minimum_lot[i] = 0.0
    if i not in item_maximum_order:
        item_maximum_order[i] = 0.0
    if i not in item_authorized:
        item_authorized[i] = 0.0
category_df['minimum_quantity'] = safe_float(category_df['minimum_quantity'])
category_df['maximum_quantity'] = safe_float(category_df['maximum_quantity'])
category_minimum_quantity = category_df.set_index('category')['minimum_quantity'].to_dict()
category_maximum_quantity = category_df.set_index('category')['maximum_quantity'].to_dict()
for g in categories:
    if g not in category_minimum_quantity:
        category_minimum_quantity[g] = 0.0
    if g not in category_maximum_quantity:
        category_maximum_quantity[g] = float('inf')
usage_df = usage_df[usage_df['item_ref'].notna() & (usage_df['item_ref'] != '') & usage_df['resource'].notna() & (usage_df['resource'] != '')]
usage_df['amount'] = safe_float(usage_df['amount'])
usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()

def usage_unit_factor(resource, unit):
    if resource == 'space':
        if unit == 'ml':
            return 1.0
        elif unit == 'liter':
            return 1000.0
    elif resource == 'labor':
        if unit == 'minute':
            return 1.0
        elif unit == 'hour':
            return 60.0
    elif resource == 'power':
        if unit == 'wh':
            return 1.0
        elif unit == 'kwh':
            return 1000.0
    return 1.0
item_resource_usage = {}
resources = set()
for idx, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = float(row['amount'])
    unit = row['unit']
    factor = usage_unit_factor(r, unit)
    key = (i, r)
    resources.add(r)
    item_resource_usage[key] = item_resource_usage.get(key, 0.0) + amt * factor
resources = sorted(resources)
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['resource'].notna() & (capacity_ledger_df['resource'] != '')]
capacity_ledger_df['amount'] = safe_float(capacity_ledger_df['amount'])
capacity_ledger_df['unit'] = capacity_ledger_df['unit'].str.strip().str.casefold()
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['effective_date'] <= asof_date]

def ledger_unit_factor(resource, unit):
    if resource == 'space':
        if unit == 'ml':
            return 1.0
        elif unit == 'liter':
            return 1000.0
    elif resource == 'labor':
        if unit == 'minute':
            return 1.0
        elif unit == 'hour':
            return 60.0
    elif resource == 'power':
        if unit == 'wh':
            return 1.0
        elif unit == 'kwh':
            return 1000.0
    return 1.0
resource_capacity = {}
for r in resources:
    rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
    total = 0.0
    for idx, row in rows.iterrows():
        amt = float(row['amount'])
        unit = row['unit']
        factor = ledger_unit_factor(r, unit)
        total += amt * factor
    resource_capacity[r] = total
incompatible_df = incompatible_df[incompatible_df['item_a'].notna() & (incompatible_df['item_a'] != '') & incompatible_df['item_b'].notna() & (incompatible_df['item_b'] != '')]
incompatible_pairs = set()
for idx, row in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in items and b in items:
        key = tuple(sorted([a, b]))
        incompatible_pairs.add(key)
incompatible_pairs = sorted(incompatible_pairs)
requires_df = requires_df[requires_df['item_ref'].notna() & (requires_df['item_ref'] != '') & requires_df['prerequisite_ref'].notna() & (requires_df['prerequisite_ref'] != '')]
requires_pairs = []
for idx, row in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if i in items and k in items:
        requires_pairs.append((i, k))
requires_pairs = sorted(set(requires_pairs))
m = gp.Model('InventoryReplenishment')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in items:
    min_lot = item_minimum_lot[i]
    max_order = item_maximum_order[i]
    authorized = item_authorized[i]
    if authorized > 0:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i])
        m.addConstr(q_vars[i] <= max_order * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for g in categories:
    for i in category_to_items[g]:
        m.addConstr(z_vars[i] <= y_vars[g])
    m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in category_to_items[g])))
for g in categories:
    minq = category_minimum_quantity[g]
    maxq = category_maximum_quantity[g]
    m.addConstr(gp.quicksum((q_vars[i] for i in category_to_items[g])) >= minq)
    m.addConstr(gp.quicksum((q_vars[i] for i in category_to_items[g])) <= maxq)
for r in resources:
    usage_sum = gp.quicksum((item_resource_usage.get((i, r), 0.0) * q_vars[i] for i in items))
    m.addConstr(usage_sum <= resource_capacity.get(r, 0.0))
for i, j in incompatible_pairs:
    m.addConstr(z_vars[i] + z_vars[j] <= 1)
for i, k in requires_pairs:
    m.addConstr(z_vars[i] <= z_vars[k])
for i, j in bundle_pairs:
    m.addConstr(w_vars[i, j] <= z_vars[i])
    m.addConstr(w_vars[i, j] <= z_vars[j])
    m.addConstr(w_vars[i, j] >= z_vars[i] + z_vars[j] - 1)
item_benefit_term = gp.quicksum((item_benefit[i] * q_vars[i] for i in items))
item_fee_term = gp.quicksum((item_activation_fee[i] * z_vars[i] for i in items))
category_fee_term = gp.quicksum((category_activation_fee[g] * y_vars[g] for g in categories))
bundle_bonus_term = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_pairs))
m.setObjective(item_benefit_term - item_fee_term - category_fee_term + bundle_bonus_term, gp.GRB.MAXIMIZE)
m.optimize()