import gurobipy as gp
import pandas as pd
import numpy as np
import re

def select_latest_valid(df, key_cols, asof_date):
    """
    For a DataFrame with revisioned records, select the latest revision <= asof_date for each key,
    and drop if that revision is DELETE. Identical retransmissions count once.
    """
    df = df[df['effective_date'] <= asof_date].copy()
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(key_cols + ['revision'], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(subset=key_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def canonical_resource_unit(resource, unit):
    """
    Convert resource/unit to canonical units:
    - space: ml
    - power: wh
    - labor: minute
    """
    resource = resource.strip().casefold()
    unit = unit.strip().casefold()
    if resource == 'space':
        if unit == 'ml':
            return 1.0
        elif unit == 'liter':
            return 1000.0
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return 1.0
        elif unit == 'kwh':
            return 1000.0
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return 1.0
        elif unit == 'hour':
            return 60.0
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
asof_date = '2026-05-07'
dealership_id = 'OSLO_NEW_CARS'

def collect_table(table_name):
    result = []
    for df in dfs:
        if 'table' in df.columns and 'dealership_id' in df.columns:
            mask = (df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold())
            if mask.any():
                result.append(df[mask].copy())
    if result:
        return pd.concat(result, ignore_index=True)
    else:
        return pd.DataFrame()
benefit_df = collect_table('benefit')
if not benefit_df.empty:
    benefit_df = select_latest_valid(benefit_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    benefit_df = pd.DataFrame()
fx_df = collect_table('fx')
if not fx_df.empty:
    fx_df = select_latest_valid(fx_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    fx_df = pd.DataFrame()
item_fee_df = collect_table('item_fee')
if not item_fee_df.empty:
    item_fee_df = select_latest_valid(item_fee_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    item_fee_df = pd.DataFrame()
category_df = collect_table('category')
if not category_df.empty:
    category_df = select_latest_valid(category_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    category_df = pd.DataFrame()
item_df = collect_table('item')
if not item_df.empty:
    item_df = select_latest_valid(item_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    item_df = pd.DataFrame()
usage_df = collect_table('usage')
if not usage_df.empty:
    usage_df = select_latest_valid(usage_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    usage_df = pd.DataFrame()
capacity_ledger_df = collect_table('capacity_ledger')
if not capacity_ledger_df.empty:
    capacity_ledger_df = select_latest_valid(capacity_ledger_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    capacity_ledger_df = pd.DataFrame()
incompatible_df = collect_table('incompatible')
if not incompatible_df.empty:
    incompatible_df = select_latest_valid(incompatible_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    incompatible_df = pd.DataFrame()
requires_df = collect_table('requires')
if not requires_df.empty:
    requires_df = select_latest_valid(requires_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    requires_df = pd.DataFrame()
bundle_df = collect_table('bundle')
if not bundle_df.empty:
    bundle_df = select_latest_valid(bundle_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    bundle_df = pd.DataFrame()
identity_df = collect_table('identity')
if not identity_df.empty:
    identity_df = select_latest_valid(identity_df, ['table', 'dealership_id', 'record_id'], asof_date)
else:
    identity_df = pd.DataFrame()
item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='raise')
items = item_df[item_df['authorized'] > 0]['item_ref'].tolist()
if not items:
    raise ValueError('No authorized items found for OSLO_NEW_CARS as of 2026-05-07.')
item_to_category = item_df.set_index('item_ref')['category'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
categories = category_df['category'].dropna().unique().tolist()
category_min_qty = category_df.set_index('category')['minimum_quantity'].astype(int).to_dict()
category_max_qty = category_df.set_index('category')['maximum_quantity'].astype(int).to_dict()
category_activation_fee = category_df.set_index('category')['activation_fee_cents'].astype(int).to_dict()
category_to_items = {g: [] for g in categories}
for i in items:
    g = item_to_category[i]
    if g in category_to_items:
        category_to_items[g].append(i)
item_fee_df = item_fee_df[item_fee_df['item_ref'].notna()]
item_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].astype(float).to_dict()
fx_df = fx_df[fx_df['currency'].notna()]
fx_df['usd_cents_numerator'] = pd.to_numeric(fx_df['usd_cents_numerator'], errors='raise')
fx_df['denominator'] = pd.to_numeric(fx_df['denominator'], errors='raise')
fx_map = {}
for (_, row) in fx_df.iterrows():
    c = row['currency'].strip()
    fx_map[c] = (row['usd_cents_numerator'], row['denominator'])
benefit_per_item = {}
for i in items:
    df = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for (_, row) in df.iterrows():
        amt = float(row['amount'])
        curr = row['currency'].strip()
        if curr not in fx_map:
            raise ValueError(f'Missing FX rate for currency {curr}')
        (num, denom) = fx_map[curr]
        total += amt * num / denom
    benefit_per_item[i] = total
usage_per_item_resource = {}
resources = set()
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource'].strip().casefold()
    amt = float(row['amount'])
    unit = row['unit']
    factor = canonical_resource_unit(r, unit)
    usage_per_item_resource[i, r] = amt * factor
    resources.add(r)
resources = sorted(resources)
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['resource'].notna()]
resource_capacity = {}
for r in resources:
    total = 0.0
    for (_, row) in capacity_ledger_df.iterrows():
        if row['resource'].strip().casefold() == r:
            amt = float(row['amount'])
            unit = row['unit']
            factor = canonical_resource_unit(r, unit)
            total += amt * factor
    resource_capacity[r] = total
incompat_pairs = set()
for (_, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if pd.notna(a) and pd.notna(b):
        if a in items and b in items:
            incompat_pairs.add(tuple(sorted((a, b))))
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if pd.notna(i) and pd.notna(k):
        if i in items and k in items:
            requires_pairs.add((i, k))
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if pd.notna(a) and pd.notna(b):
        if a in items and b in items:
            key = tuple(sorted((a, b)))
            bundle_pairs.append(key)
            bundle_bonus[key] = float(row['bonus_cents'])
bundle_pairs = list(dict.fromkeys(bundle_pairs))
m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in items:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(q_vars[i] >= min_lot * z_vars[i])
    m.addConstr(q_vars[i] <= max_order * z_vars[i])
for g in categories:
    items_in_g = category_to_items[g]
    if items_in_g:
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= category_min_qty[g])
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= category_max_qty[g])
    else:
        m.addConstr(y_vars[g] == 0)
for g in categories:
    items_in_g = category_to_items[g]
    if items_in_g:
        for i in items_in_g:
            m.addConstr(z_vars[i] <= y_vars[g])
        m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in items_in_g)))
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource.get((i, r), 0.0) * q_vars[i] for i in items)) <= resource_capacity[r])
for (i, j) in incompat_pairs:
    m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, k) in requires_pairs:
    m.addConstr(q_vars[i] <= item_max_order[i] * gp.quicksum(q_vars[k] >= 1))
for b in bundle_pairs:
    (i, j) = b
    m.addConstr(w_vars[b] <= z_vars[i])
    m.addConstr(w_vars[b] <= z_vars[j])
    m.addConstr(w_vars[b] >= z_vars[i] + z_vars[j] - 1)
item_benefit_expr = gp.quicksum((benefit_per_item[i] * q_vars[i] for i in items))
item_fee_expr = gp.quicksum((item_fee.get(i, 0.0) * z_vars[i] for i in items))
category_fee_expr = gp.quicksum((category_activation_fee.get(g, 0.0) * y_vars[g] for g in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_pairs))
m.setObjective(item_benefit_expr - item_fee_expr - category_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.optimize()