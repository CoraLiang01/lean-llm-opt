import gurobipy as gp
import pandas as pd
import numpy as np
import re

def select_latest_valid(df, key_cols, date_col, revision_col, action_col, as_of_date):
    df = df[df[date_col] <= as_of_date]
    df[revision_col] = df[revision_col].astype(int)
    df = df.sort_values(key_cols + [revision_col], ascending=True)
    idx = df.groupby(key_cols)[revision_col].idxmax()
    df_latest = df.loc[idx].copy()
    df_latest = df_latest[df_latest[action_col].str.strip().str.casefold() != 'delete']
    return df_latest

def select_oslo_table(df, table_name):
    return df[(df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars')].copy()

def select_oslo_table_valid(df, table_name, as_of_date):
    df = df[(df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars')].copy()
    return select_latest_valid(df, ['dealership_id', 'table', 'record_id'], 'effective_date', 'revision', 'action', as_of_date)

def select_oslo_table_all(df, table_name):
    return df[(df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars')].copy()

def select_oslo_table_valid_noid(df, table_name, as_of_date):
    df = df[(df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars')].copy()
    if 'record_id' in df.columns:
        return select_latest_valid(df, ['dealership_id', 'table', 'record_id'], 'effective_date', 'revision', 'action', as_of_date)
    else:
        df = df[df['effective_date'] <= as_of_date]
        if 'action' in df.columns:
            df = df[df['action'].str.strip().str.casefold() != 'delete']
        return df

def convert_to_base_unit(amount, unit):
    unit = unit.strip().casefold()
    if unit == 'liter':
        return float(amount) * 1000.0
    elif unit == 'ml':
        return float(amount)
    elif unit == 'kwh':
        return float(amount) * 1000.0
    elif unit == 'wh':
        return float(amount)
    elif unit == 'hour':
        return float(amount) * 60.0
    elif unit == 'minute':
        return float(amount)
    else:
        raise ValueError(f'Unknown unit: {unit}')

def convert_to_usd_cents(amount, currency, fx_dict):
    currency = currency.strip().casefold()
    if currency not in fx_dict:
        raise KeyError(f'Missing FX rate for currency {currency}')
    (num, denom) = fx_dict[currency]
    return float(amount) * float(num) / float(denom)
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
as_of_date = '2026-05-07'
benefit_df = pd.concat([dfs[0], dfs[1], dfs[2]], ignore_index=True)
benefit_df = select_oslo_table_valid(benefit_df, 'benefit', as_of_date)
fx_df = dfs[9]
fx_df = select_oslo_table_valid(fx_df, 'fx', as_of_date)
fx_df['currency_norm'] = fx_df['currency'].str.strip().str.casefold()
fx_dict = {}
for (_, row) in fx_df.iterrows():
    fx_dict[row['currency_norm']] = (float(row['usd_cents_numerator']), float(row['denominator']))
item_df = pd.concat([dfs[14], dfs[15]], ignore_index=True)
item_df = select_oslo_table_valid(item_df, 'item', as_of_date)
item_df['authorized'] = item_df['authorized'].astype(float)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(float)
item_df['maximum_order'] = item_df['maximum_order'].astype(float)
item_df['category'] = item_df['category'].str.strip()
category_df = pd.concat([dfs[7], dfs[8], dfs[9]], ignore_index=True)
category_df = select_oslo_table_valid(category_df, 'category', as_of_date)
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(float)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(float)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(float)
category_df['category'] = category_df['category'].str.strip()
item_fee_df = pd.concat([dfs[16], dfs[17]], ignore_index=True)
item_fee_df = select_oslo_table_valid(item_fee_df, 'item_fee', as_of_date)
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(float)
bundle_df = pd.concat([dfs[3], dfs[4]], ignore_index=True)
bundle_df = select_oslo_table_valid(bundle_df, 'bundle', as_of_date)
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(float)
bundle_df['item_a'] = bundle_df['item_a'].str.strip()
bundle_df['item_b'] = bundle_df['item_b'].str.strip()
incomp_df = pd.concat([dfs[12], dfs[13]], ignore_index=True)
incomp_df = select_oslo_table_valid(incomp_df, 'incompatible', as_of_date)
incomp_df['item_a'] = incomp_df['item_a'].str.strip()
incomp_df['item_b'] = incomp_df['item_b'].str.strip()
requires_df = pd.concat([dfs[20], dfs[21]], ignore_index=True)
requires_df = select_oslo_table_valid(requires_df, 'requires', as_of_date)
requires_df['item_ref'] = requires_df['item_ref'].str.strip()
requires_df['prerequisite_ref'] = requires_df['prerequisite_ref'].str.strip()
usage_df = pd.concat([dfs[22], dfs[23], dfs[24]], ignore_index=True)
usage_df = select_oslo_table_all(usage_df, 'usage')
usage_df['amount'] = usage_df['amount'].astype(float)
usage_df['item_ref'] = usage_df['item_ref'].str.strip()
usage_df['resource'] = usage_df['resource'].str.strip()
usage_df['unit'] = usage_df['unit'].str.strip()
cap_df = pd.concat([dfs[5], dfs[6]], ignore_index=True)
cap_df = select_oslo_table_valid_noid(cap_df, 'capacity_ledger', as_of_date)
cap_df['amount'] = pd.to_numeric(cap_df['amount'], errors='coerce')
cap_df['resource'] = cap_df['resource'].str.strip()
cap_df['unit'] = cap_df['unit'].str.strip()
identity_df = pd.concat([dfs[10], dfs[11]], ignore_index=True)
identity_df = select_oslo_table_valid(identity_df, 'identity', as_of_date)
identity_df['ref'] = identity_df['ref'].str.strip()
identity_df['entity_id'] = identity_df['entity_id'].str.strip()
identity_df['display_name'] = identity_df['display_name'].str.strip()
options = sorted(item_df['item_ref'].unique())
item_category = item_df.set_index('item_ref')['category'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
categories = sorted(category_df['category'].unique())
cat_min_qty = category_df.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = category_df.set_index('category')['maximum_quantity'].to_dict()
cat_activation_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
usage_resources = usage_df['resource'].unique()
resources = sorted([r for r in usage_resources if r != ''])
bundles = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    key = tuple(sorted([a, b]))
    if key not in bundle_bonus or row['bonus_cents'] > bundle_bonus[key]:
        bundle_bonus[key] = row['bonus_cents']
    if key not in bundles:
        bundles.append(key)
incompat_pairs = set()
for (_, row) in incomp_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    key = tuple(sorted([a, b]))
    incompat_pairs.add(key)
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    requires_pairs.add((i, k))
item_fee_map = {}
for (_, row) in item_fee_df.iterrows():
    item_fee_map[row['item_ref']] = row['activation_fee_cents']
usage_per_unit = {i: {} for i in options}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    if i not in options or r == '':
        continue
    amt = convert_to_base_unit(row['amount'], row['unit'])
    usage_per_unit.setdefault(i, {})
    usage_per_unit[i][r] = amt
cap_df = cap_df[cap_df['resource'] != '']
resource_capacity = {}
for r in resources:
    cap_r = cap_df[cap_df['resource'] == r]
    total = 0.0
    for (_, row) in cap_r.iterrows():
        amt = row['amount']
        if pd.isnull(amt) or amt == '':
            continue
        amt = convert_to_base_unit(amt, row['unit'])
        total += amt
    resource_capacity[r] = total
benefit_per_unit = {i: 0.0 for i in options}
for i in options:
    rows = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for (_, row) in rows.iterrows():
        amt = float(row['amount'])
        curr = row['currency']
        curr_norm = curr.strip().casefold()
        usd_amt = convert_to_usd_cents(amt, curr, fx_dict)
        total += usd_amt
    benefit_per_unit[i] = total
item_fee = {i: item_fee_map.get(i, 0.0) for i in options}
option_category = {i: item_category[i] for i in options}
category_fee = {g: cat_activation_fee.get(g, 0.0) for g in categories}
authorized = {i: int(item_authorized[i]) for i in options}
min_lot = {i: int(item_min_lot[i]) for i in options}
max_order = {i: int(item_max_order[i]) for i in options}
cat_min = {g: int(cat_min_qty[g]) for g in categories}
cat_max = {g: int(cat_max_qty[g]) for g in categories}
bundle_bonus_val = {b: bundle_bonus[b] for b in bundles}
usage = {}
for i in options:
    usage[i] = {}
    for r in resources:
        usage[i][r] = usage_per_unit.get(i, {}).get(r, 0.0)
m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(options, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in options:
    if authorized[i] > 0:
        m.addConstr(x_vars[i] >= min_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= max_order[i] * y_vars[i], name=f'maxorder_{i}')
    else:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    m.addConstr((x_vars[i] >= 1) >> (y_vars[i] == 1), name=f'y_link1_{i}')
    m.addConstr((y_vars[i] == 0) >> (x_vars[i] == 0), name=f'y_link2_{i}')
for g in categories:
    items_in_g = [i for i in options if option_category[i] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_min[g] * z_vars[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_max[g] * z_vars[g], name=f'cat_max_{g}')
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'cat_z_{g}_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage[i][r] * x_vars[i] for i in options)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    if i in options and j in options:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, k) in requires_pairs:
    if i in options and k in options:
        m.addConstr(y_vars[i] <= y_vars[k], name=f'requires_{i}_{k}')
for b in bundles:
    (i, j) = b
    if i in options and j in options:
        m.addConstr(w_vars[b] <= y_vars[i], name=f'bundle1_{i}_{j}')
        m.addConstr(w_vars[b] <= y_vars[j], name=f'bundle2_{i}_{j}')
        m.addConstr(w_vars[b] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{i}_{j}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle0_{i}_{j}')
obj = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in options))
obj -= gp.quicksum((item_fee[i] * y_vars[i] for i in options))
obj -= gp.quicksum((category_fee[g] * z_vars[g] for g in categories))
obj += gp.quicksum((bundle_bonus_val[b] * w_vars[b] for b in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()