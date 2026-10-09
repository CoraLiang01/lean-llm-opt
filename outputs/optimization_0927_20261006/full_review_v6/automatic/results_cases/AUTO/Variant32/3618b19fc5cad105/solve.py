import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_str(s):
    return s.strip().casefold() if isinstance(s, str) else s

def as_of_revision(df, key_cols, as_of_date):
    df = df[df['effective_date'] <= as_of_date].copy()
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(key_cols + ['revision'], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(subset=key_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def get_oslo_table(df, table_name, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == table_name]
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_table_no_tablecol(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    key_cols = ['dealership_id', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_identity(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'identity']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_item(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'item']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_category(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'category']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_item_fee(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'item_fee']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_bundle(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'bundle']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_incompatible(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'incompatible']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_requires(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'requires']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_usage(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'usage']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_capacity_ledger(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'capacity_ledger']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_fx(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'fx']
    key_cols = ['dealership_id', 'table', 'record_id']
    return as_of_revision(df, key_cols, as_of_date)

def get_oslo_benefit(df, as_of_date):
    df = df[df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars']
    df = df[df['table'].str.strip().str.casefold() == 'benefit']
    key_cols = ['dealership_id', 'table', 'record_id']
    df = as_of_revision(df, key_cols, as_of_date)
    return df
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = []
for path in csv_paths:
    dfs.append(pd.read_csv(path, dtype=str, keep_default_na=False))
benefit_df = pd.concat([dfs[0], dfs[1], dfs[2]], ignore_index=True)
bundle_df = pd.concat([dfs[3], dfs[4]], ignore_index=True)
capacity_ledger_df = pd.concat([dfs[5], dfs[6]], ignore_index=True)
category_df = pd.concat([dfs[7], dfs[8]], ignore_index=True)
fx_df = dfs[9]
identity_df = pd.concat([dfs[10], dfs[11]], ignore_index=True)
incompatible_df = pd.concat([dfs[12], dfs[13]], ignore_index=True)
item_df = pd.concat([dfs[14], dfs[15]], ignore_index=True)
item_fee_df = pd.concat([dfs[16], dfs[17]], ignore_index=True)
requires_df = pd.concat([dfs[20], dfs[21]], ignore_index=True)
usage_df = pd.concat([dfs[22], dfs[23], dfs[24]], ignore_index=True)
as_of_date = '2026-05-07'
benefit_df = get_oslo_benefit(benefit_df, as_of_date)
bundle_df = get_oslo_bundle(bundle_df, as_of_date)
capacity_ledger_df = get_oslo_capacity_ledger(capacity_ledger_df, as_of_date)
category_df = get_oslo_category(category_df, as_of_date)
fx_df = get_oslo_fx(fx_df, as_of_date)
identity_df = get_oslo_identity(identity_df, as_of_date)
incompatible_df = get_oslo_incompatible(incompatible_df, as_of_date)
item_df = get_oslo_item(item_df, as_of_date)
item_fee_df = get_oslo_item_fee(item_fee_df, as_of_date)
requires_df = get_oslo_requires(requires_df, as_of_date)
usage_df = get_oslo_usage(usage_df, as_of_date)
item_df = item_df.reset_index(drop=True)
items = list(item_df['item_ref'].unique())
category_df = category_df.reset_index(drop=True)
categories = list(category_df['category'].unique())
usage_resources = set(usage_df['resource'].unique())
capacity_resources = set(capacity_ledger_df['resource'].unique())
resources = sorted(list(usage_resources | capacity_resources))
bundle_df = bundle_df.reset_index(drop=True)
bundles = []
bundle_pairs = []
for (idx, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if pd.isna(a) or pd.isna(b) or a == '' or (b == ''):
        continue
    bundles.append(idx)
    bundle_pairs.append((a, b))
incompatible_df = incompatible_df.reset_index(drop=True)
incompatible_pairs = []
for (idx, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if pd.isna(a) or pd.isna(b) or a == '' or (b == ''):
        continue
    incompatible_pairs.append((a, b))
requires_df = requires_df.reset_index(drop=True)
requires_pairs = []
for (idx, row) in requires_df.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if pd.isna(i) or pd.isna(p) or i == '' or (p == ''):
        continue
    requires_pairs.append((i, p))
fx_df = fx_df.reset_index(drop=True)
fx_map = {}
for (idx, row) in fx_df.iterrows():
    currency = row['currency']
    if pd.isna(currency) or currency == '':
        continue
    currency = currency.strip()
    usd_cents_numerator = float(row['usd_cents_numerator'])
    denominator = float(row['denominator'])
    fx_map[currency] = (usd_cents_numerator, denominator)
benefit_per_item = {}
for i in items:
    rows = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for (idx, row) in rows.iterrows():
        amount = float(row['amount'])
        currency = row['currency'].strip()
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency} as of {as_of_date}')
        (num, denom) = fx_map[currency]
        usd_cents = amount * num / denom
        total += usd_cents
    benefit_per_item[i] = total
item_fee_df = item_fee_df.reset_index(drop=True)
item_fee_map = {}
for i in items:
    rows = item_fee_df[item_fee_df['item_ref'] == i]
    if len(rows) == 0:
        continue
    fee = float(rows.iloc[0]['activation_fee_cents'])
    item_fee_map[i] = fee
category_param_map = {}
for (idx, row) in category_df.iterrows():
    c = row['category']
    minq = float(row['minimum_quantity'])
    maxq = float(row['maximum_quantity'])
    fee = float(row['activation_fee_cents'])
    category_param_map[c] = {'min': minq, 'max': maxq, 'fee': fee}
item_param_map = {}
for (idx, row) in item_df.iterrows():
    i = row['item_ref']
    c = row['category']
    authorized = int(float(row['authorized']))
    min_lot = int(float(row['minimum_lot']))
    max_order = int(float(row['maximum_order']))
    item_param_map[i] = {'category': c, 'authorized': authorized, 'min_lot': min_lot, 'max_order': max_order}
usage_map = {}
for (idx, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    if pd.isna(i) or pd.isna(r) or i == '' or (r == ''):
        continue
    amount = float(row['amount'])
    unit = row['unit'].strip().casefold()
    if r == 'space':
        if unit == 'liter':
            amount = amount * 1000.0
        elif unit == 'ml':
            pass
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif r == 'labor':
        if unit == 'hour':
            amount = amount * 60.0
        elif unit == 'minute':
            pass
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    elif r == 'power':
        if unit == 'kwh':
            amount = amount * 1000.0
        elif unit == 'wh':
            pass
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    else:
        raise ValueError(f'Unknown resource: {r}')
    usage_map[i, r] = amount
capacity_map = {}
for r in resources:
    rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
    total = 0.0
    for (idx, row) in rows.iterrows():
        amount = float(row['amount'])
        unit = row['unit'].strip().casefold()
        if r == 'space':
            if unit == 'liter':
                amount = amount * 1000.0
            elif unit == 'ml':
                pass
            else:
                raise ValueError(f'Unknown unit for space: {unit}')
        elif r == 'labor':
            if unit == 'hour':
                amount = amount * 60.0
            elif unit == 'minute':
                pass
            else:
                raise ValueError(f'Unknown unit for labor: {unit}')
        elif r == 'power':
            if unit == 'kwh':
                amount = amount * 1000.0
            elif unit == 'wh':
                pass
            else:
                raise ValueError(f'Unknown unit for power: {unit}')
        else:
            raise ValueError(f'Unknown resource: {r}')
        total += amount
    capacity_map[r] = total
bundle_bonus_map = {}
for (idx, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if pd.isna(a) or pd.isna(b) or a == '' or (b == ''):
        continue
    bonus = float(row['bonus_cents'])
    bundle_bonus_map[a, b] = bonus
category_items_map = {c: [] for c in categories}
for i in items:
    c = item_param_map[i]['category']
    if c in category_items_map:
        category_items_map[c].append(i)
    else:
        category_items_map[c] = [i]
m = gp.Model('OsloVehicleOrderNetBenefit')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(range(len(bundle_pairs)), vtype=gp.GRB.BINARY, name='')
for i in items:
    params = item_param_map[i]
    authorized = params['authorized']
    min_lot = params['min_lot']
    max_order = params['max_order']
    if authorized > 0:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i])
        m.addConstr(q_vars[i] <= max_order * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for c in categories:
    minq = category_param_map[c]['min']
    maxq = category_param_map[c]['max']
    items_in_c = category_items_map[c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= minq)
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= maxq)
for c in categories:
    items_in_c = category_items_map[c]
    for i in items_in_c:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)))
for r in resources:
    usage_expr = gp.quicksum((usage_map.get((i, r), 0.0) * q_vars[i] for i in items))
    m.addConstr(usage_expr <= capacity_map[r])
for (i, j) in incompatible_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, p) in requires_pairs:
    if i in items and p in items:
        m.addConstr(q_vars[p] >= z_vars[i])
for (b, (i, j)) in enumerate(bundle_pairs):
    if i in items and j in items:
        m.addConstr(w_vars[b] <= z_vars[i])
        m.addConstr(w_vars[b] <= z_vars[j])
        m.addConstr(w_vars[b] >= z_vars[i] + z_vars[j] - 1)
    else:
        m.addConstr(w_vars[b] == 0)
benefit_term = gp.quicksum((benefit_per_item.get(i, 0.0) * q_vars[i] for i in items))
item_fee_term = gp.quicksum((item_fee_map.get(i, 0.0) * z_vars[i] for i in items))
category_fee_term = gp.quicksum((category_param_map[c]['fee'] * y_vars[c] for c in categories))
bundle_bonus_term = gp.quicksum((bundle_bonus_map.get(bundle_pairs[b], 0.0) * w_vars[b] for b in range(len(bundle_pairs))))
m.setObjective(benefit_term - item_fee_term - category_fee_term + bundle_bonus_term, gp.GRB.MAXIMIZE)
m.optimize()