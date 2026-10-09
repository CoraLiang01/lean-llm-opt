import gurobipy as gp
import pandas as pd
import numpy as np
import re

def latest_valid_records(df, key_cols, date_col, revision_col, action_col, cutoff_date):
    """
    For a DataFrame with revisioned records, return only the latest valid (not future, not deleted) record
    for each key (tuple of key_cols), as of cutoff_date (inclusive).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = pd.to_datetime(cutoff_date)
    df = df[df[date_col] <= cutoff]
    df = df.sort_values(key_cols + [revision_col], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(subset=key_cols, keep='first')
    if action_col in df.columns:
        df = df[df[action_col].str.casefold() != 'delete']
    return df

def get_fx_dict(fx_df):
    """
    Returns a dict: currency -> (usd_cents_numerator, denominator)
    """
    fx_dict = {}
    for (_, row) in fx_df.iterrows():
        c = row['currency']
        fx_dict[c] = (row['usd_cents_numerator'], row['denominator'])
    return fx_dict

def convert_to_usd_cents(amount, currency, fx_dict):
    """
    Convert amount in given currency to USD cents using fx_dict.
    """
    if currency not in fx_dict:
        raise ValueError(f'Missing FX rate for currency {currency}')
    (num, denom) = fx_dict[currency]
    return amount * num / denom

def unit_to_base(resource, amount, unit):
    """
    Convert amount/unit to base units:
    - space: ml (from liter)
    - power: wh (from kwh)
    - labor: minute (from hour)
    """
    if resource == 'space':
        if unit == 'liter':
            return amount * 1000
        elif unit == 'ml':
            return amount
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'kwh':
            return amount * 1000
        elif unit == 'wh':
            return amount
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'hour':
            return amount * 60
        elif unit == 'minute':
            return amount
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, sep=',') for p in paths]
table_map = {'benefit': [0, 1, 2], 'bundle': [3, 4], 'capacity_ledger': [5, 6], 'category': [7, 8], 'fx': [9], 'identity': [10, 11], 'incompatible': [12, 13], 'item': [14, 15], 'item_fee': [16, 17], 'market': [18, 19], 'requires': [20, 21], 'usage': [22, 23, 24]}
cutoff_date = '2026-05-07'
dealership_id = 'OSLO_NEW_CARS'

def select_table(table_name, key_cols, date_col='effective_date', revision_col='revision', action_col='action'):
    idxs = table_map[table_name]
    df = pd.concat([dfs[i] for i in idxs], ignore_index=True)
    df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
    df = df[df['table'].str.casefold() == table_name.casefold()]
    df = latest_valid_records(df, key_cols, date_col, revision_col, action_col, cutoff_date)
    return df
benefit_df = select_table('benefit', ['record_id'])
bundle_df = select_table('bundle', ['record_id'])
capacity_df = select_table('capacity_ledger', ['record_id'])
category_df = select_table('category', ['record_id'])
fx_df = select_table('fx', ['record_id'])
identity_df = select_table('identity', ['record_id'])
incompatible_df = select_table('incompatible', ['record_id'])
item_df = select_table('item', ['record_id'])
item_fee_df = select_table('item_fee', ['record_id'])
requires_df = select_table('requires', ['record_id'])
usage_df = select_table('usage', ['record_id'])
fx_df = fx_df.dropna(subset=['currency'])
fx_dict = {}
for (_, row) in fx_df.iterrows():
    fx_dict[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
item_df = item_df.dropna(subset=['item_ref'])
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['category'] = item_df['category'].astype(str)
options = sorted(item_df['item_ref'].unique())
option_set = set(options)
authorized_dict = dict(zip(item_df['item_ref'], item_df['authorized']))
min_lot_dict = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
max_order_dict = dict(zip(item_df['item_ref'], item_df['maximum_order']))
option_category = dict(zip(item_df['item_ref'], item_df['category']))
category_df = category_df.dropna(subset=['category'])
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = sorted(category_df['category'].unique())
cat_min_qty = dict(zip(category_df['category'], category_df['minimum_quantity']))
cat_max_qty = dict(zip(category_df['category'], category_df['maximum_quantity']))
cat_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
item_fee_df = item_fee_df.dropna(subset=['item_ref'])
item_fee_map = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
benefit_df = benefit_df.dropna(subset=['item_ref', 'amount', 'currency'])
benefit_per_unit = {}
for i in options:
    df_i = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for (_, row) in df_i.iterrows():
        amt = float(row['amount'])
        curr = row['currency']
        usd_cents = convert_to_usd_cents(amt, curr, fx_dict)
        total += usd_cents
    benefit_per_unit[i] = total
usage_df = usage_df.dropna(subset=['item_ref', 'resource', 'amount', 'unit'])
resource_types = sorted(usage_df['resource'].unique())
usage_per_unit = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = float(row['amount'])
    unit = row['unit']
    amt_base = unit_to_base(r, amt, unit)
    usage_per_unit[i, r] = amt_base
capacity_df = capacity_df.dropna(subset=['resource', 'amount', 'unit'])
resource_capacity = {}
for r in resource_types:
    df_r = capacity_df[capacity_df['resource'] == r]
    total = 0.0
    for (_, row) in df_r.iterrows():
        amt = float(row['amount'])
        unit = row['unit']
        amt_base = unit_to_base(r, amt, unit)
        total += amt_base
    resource_capacity[r] = total
incompatible_df = incompatible_df.dropna(subset=['item_a', 'item_b'])
incompat_pairs = set()
for (_, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in option_set and b in option_set:
        incompat_pairs.add(tuple(sorted((a, b))))
requires_df = requires_df.dropna(subset=['item_ref', 'prerequisite_ref'])
requires_pairs = []
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if i in option_set and k in option_set:
        requires_pairs.append((i, k))
bundle_df = bundle_df.dropna(subset=['item_a', 'item_b', 'bonus_cents'])
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in option_set and b in option_set:
        key = tuple(sorted((a, b)))
        bundle_pairs.append(key)
        bundle_bonus[key] = float(row['bonus_cents'])
cat_options = {g: set() for g in categories}
for i in options:
    g = option_category[i]
    if g in categories:
        cat_options[g].add(i)
m = gp.Model('oslo_vehicle_order')
q = m.addVars(options, vtype=gp.GRB.INTEGER, name='')
z = m.addVars(options, vtype=gp.GRB.BINARY, name='')
y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in options:
    if authorized_dict[i] == 0:
        m.addConstr(q[i] == 0)
        m.addConstr(z[i] == 0)
    else:
        m.addConstr(q[i] >= min_lot_dict[i] * z[i])
        m.addConstr(q[i] <= max_order_dict[i] * z[i])
        m.addConstr(z[i] <= 1)
        m.addConstr(q[i] >= 0)
for g in categories:
    opts = list(cat_options[g])
    if len(opts) == 0:
        m.addConstr(y[g] == 0)
        continue
    m.addConstr(gp.quicksum((q[i] for i in opts)) >= cat_min_qty[g] * y[g])
    m.addConstr(gp.quicksum((q[i] for i in opts)) <= cat_max_qty[g] * y[g])
    for i in opts:
        m.addConstr(z[i] <= y[g])
for r in resource_types:
    m.addConstr(gp.quicksum((usage_per_unit.get((i, r), 0.0) * q[i] for i in options)) <= resource_capacity[r])
for (a, b) in incompat_pairs:
    m.addConstr(z[a] + z[b] <= 1)
for (i, k) in requires_pairs:
    m.addConstr(q[i] <= max_order_dict[i] * z[k])
for key in bundle_pairs:
    (a, b) = key
    m.addConstr(w[key] <= z[a])
    m.addConstr(w[key] <= z[b])
    m.addConstr(w[key] >= z[a] + z[b] - 1)
obj = gp.quicksum((benefit_per_unit.get(i, 0.0) * q[i] - item_fee_map.get(i, 0.0) * z[i] for i in options))
obj += gp.quicksum((bundle_bonus[key] * w[key] for key in bundle_pairs))
obj -= gp.quicksum((cat_fee[g] * y[g] for g in categories))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net benefit in USD cents: {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')