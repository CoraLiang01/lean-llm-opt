import gurobipy as gp
import pandas as pd
import numpy as np
import re

def select_latest_valid(df, as_of_date, key_cols):
    df = df[df['effective_date'] <= as_of_date].copy()
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(key_cols + ['revision'], ascending=True)
    idx = df.groupby(key_cols)['revision'].idxmax()
    df = df.loc[idx]
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def dropna_strict(df, required_cols):
    return df.dropna(subset=required_cols)

def to_base_unit(amount, unit):
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
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
as_of_date = '2026-05-07'
dealership_id = 'OSLO_NEW_CARS'
benefit_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'benefit').any()], ignore_index=True)
benefit_df = benefit_df[benefit_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
benefit_df = select_latest_valid(benefit_df, as_of_date, ['dealership_id', 'table', 'record_id'])
benefit_df = dropna_strict(benefit_df, ['item_ref', 'component', 'amount', 'currency'])
fx_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'fx').any()], ignore_index=True)
fx_df = fx_df[fx_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
fx_df = select_latest_valid(fx_df, as_of_date, ['dealership_id', 'table', 'currency'])
fx_df = dropna_strict(fx_df, ['currency', 'usd_cents_numerator', 'denominator'])
fx_df['usd_cents_numerator'] = fx_df['usd_cents_numerator'].astype(float)
fx_df['denominator'] = fx_df['denominator'].astype(float)
fx_map = {}
for (_, row) in fx_df.iterrows():
    fx_map[row['currency'].strip()] = (row['usd_cents_numerator'], row['denominator'])
item_fee_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'item_fee').any()], ignore_index=True)
item_fee_df = item_fee_df[item_fee_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item_fee_df = select_latest_valid(item_fee_df, as_of_date, ['dealership_id', 'table', 'record_id'])
item_fee_df = dropna_strict(item_fee_df, ['item_ref', 'activation_fee_cents'])
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(float)
item_fee_map = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
category_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'category').any()], ignore_index=True)
category_df = category_df[category_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
category_df = select_latest_valid(category_df, as_of_date, ['dealership_id', 'table', 'record_id'])
category_df = dropna_strict(category_df, ['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'])
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(float)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(float)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(float)
category_map = {}
for (_, row) in category_df.iterrows():
    category_map[row['category']] = {'min': row['minimum_quantity'], 'max': row['maximum_quantity'], 'fee': row['activation_fee_cents']}
item_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'item').any()], ignore_index=True)
item_df = item_df[item_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item_df = select_latest_valid(item_df, as_of_date, ['dealership_id', 'table', 'record_id'])
item_df = dropna_strict(item_df, ['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order'])
item_df['authorized'] = item_df['authorized'].astype(float)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(float)
item_df['maximum_order'] = item_df['maximum_order'].astype(float)
item_category_map = dict(zip(item_df['item_ref'], item_df['category']))
item_authorized_map = dict(zip(item_df['item_ref'], item_df['authorized']))
item_minlot_map = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_maxorder_map = dict(zip(item_df['item_ref'], item_df['maximum_order']))
usage_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'usage').any()], ignore_index=True)
usage_df = usage_df[usage_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
usage_df = select_latest_valid(usage_df, as_of_date, ['dealership_id', 'table', 'record_id'])
usage_df = dropna_strict(usage_df, ['item_ref', 'resource', 'amount', 'unit'])
usage_df['amount'] = usage_df['amount'].astype(float)
usage_map = {}
for (_, row) in usage_df.iterrows():
    key = (row['item_ref'], row['resource'])
    usage_map[key] = to_base_unit(row['amount'], row['unit'])
capacity_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'capacity_ledger').any()], ignore_index=True)
capacity_df = capacity_df[capacity_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
capacity_df = select_latest_valid(capacity_df, as_of_date, ['dealership_id', 'table', 'record_id'])
capacity_df = dropna_strict(capacity_df, ['resource', 'amount', 'unit'])
capacity_df['amount'] = capacity_df['amount'].astype(float)
capacity_resource_map = {}
for (_, group) in capacity_df.groupby('resource'):
    resource = group['resource'].iloc[0]
    total = 0.0
    for (_, row) in group.iterrows():
        total += to_base_unit(row['amount'], row['unit'])
    capacity_resource_map[resource] = total
incompat_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'incompatible').any()], ignore_index=True)
incompat_df = incompat_df[incompat_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
incompat_df = select_latest_valid(incompat_df, as_of_date, ['dealership_id', 'table', 'record_id'])
incompat_df = dropna_strict(incompat_df, ['item_a', 'item_b'])
incompat_pairs = set()
for (_, row) in incompat_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a and b:
        incompat_pairs.add(tuple(sorted((a, b))))
requires_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'requires').any()], ignore_index=True)
requires_df = requires_df[requires_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
requires_df = select_latest_valid(requires_df, as_of_date, ['dealership_id', 'table', 'record_id'])
requires_df = dropna_strict(requires_df, ['item_ref', 'prerequisite_ref'])
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    (i, j) = (row['item_ref'], row['prerequisite_ref'])
    if i and j:
        requires_pairs.add((i, j))
bundle_df = pd.concat([df for df in dfs if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'bundle').any()], ignore_index=True)
bundle_df = bundle_df[bundle_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
bundle_df = select_latest_valid(bundle_df, as_of_date, ['dealership_id', 'table', 'record_id'])
bundle_df = dropna_strict(bundle_df, ['item_a', 'item_b', 'bonus_cents'])
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(float)
bundle_list = []
bundle_bonus_map = {}
for (idx, row) in bundle_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a and b:
        key = tuple(sorted((a, b)))
        bundle_list.append(key)
        bundle_bonus_map[key] = row['bonus_cents']
items = sorted(set(item_df['item_ref']))
categories = sorted(set(category_df['category']))
resources = sorted(set(capacity_resource_map.keys()))
bundles = sorted(bundle_list)
incompat_pairs = sorted(incompat_pairs)
requires_pairs = sorted(requires_pairs)
benefit_per_item = {i: 0.0 for i in items}
for (_, row) in benefit_df.iterrows():
    i = row['item_ref']
    if i not in benefit_per_item:
        continue
    amount = float(row['amount'])
    currency = row['currency'].strip()
    if currency not in fx_map:
        raise ValueError(f'Missing FX rate for currency {currency}')
    (num, denom) = fx_map[currency]
    benefit_usd_cents = amount * float(num) / float(denom)
    benefit_per_item[i] += benefit_usd_cents
usage_per_item_resource = {}
for i in items:
    for r in resources:
        usage_per_item_resource[i, r] = usage_map.get((i, r), 0.0)
item_fee = {i: item_fee_map.get(i, 0.0) for i in items}
item_category = {i: item_category_map[i] for i in items}
category_items = {c: [i for i in items if item_category[i] == c] for c in categories}
item_authorized = {i: int(item_authorized_map[i]) for i in items}
item_minlot = {i: int(item_minlot_map[i]) for i in items}
item_maxorder = {i: int(item_maxorder_map[i]) for i in items}
category_min = {c: int(category_map[c]['min']) for c in categories}
category_max = {c: int(category_map[c]['max']) for c in categories}
category_fee = {c: float(category_map[c]['fee']) for c in categories}
resource_capacity = {r: float(capacity_resource_map[r]) for r in resources}
bundle_bonus = {b: bundle_bonus_map[b] for b in bundles}
m = gp.Model('OsloVehicleOrderNetBenefit')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    if item_authorized[i]:
        m.addConstr(q_vars[i] >= item_minlot[i] * z_vars[i])
        m.addConstr(q_vars[i] <= item_maxorder[i] * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for c in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) >= category_min[c] * y_vars[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) <= category_max[c] * y_vars[c])
    for i in category_items[c]:
        m.addConstr(z_vars[i] <= y_vars[c])
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource[i, r] * q_vars[i] for i in items)) <= resource_capacity[r])
for (a, b) in incompat_pairs:
    if a in items and b in items:
        m.addConstr(z_vars[a] + z_vars[b] <= 1)
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(q_vars[i] <= item_maxorder[i] * z_vars[j])
        m.addConstr(q_vars[j] >= z_vars[i])
for b in bundles:
    (a, b_) = b
    if a in items and b_ in items:
        m.addConstr(w_vars[b] <= z_vars[a])
        m.addConstr(w_vars[b] <= z_vars[b_])
        m.addConstr(w_vars[b] >= z_vars[a] + z_vars[b_] - 1)
objective = gp.quicksum((benefit_per_item[i] * q_vars[i] for i in items)) - gp.quicksum((item_fee[i] * z_vars[i] for i in items)) - gp.quicksum((category_fee[c] * y_vars[c] for c in categories)) + gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundles))
m.setObjective(objective, gp.GRB.MAXIMIZE)
m.optimize()