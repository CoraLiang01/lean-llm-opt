import gurobipy as gp
import pandas as pd
import numpy as np
import re

def filter_latest(df, as_of_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= as_of_date]
    df['revision'] = df['revision'].astype(int)
    df = df.sort_values(['tenant', 'table', 'record_id', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=['tenant', 'table', 'record_id'], keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def safe_float(series):
    return pd.to_numeric(series.fillna(0), errors='raise')

def safe_int(series):
    return pd.to_numeric(series.fillna(0), errors='raise', downcast='integer')
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
table_map = {'requires': [0, 23], 'identity': [1, 21], 'item_fee': [2, 16], 'bundle': [3, 10], 'benefit': [4, 11, 14], 'usage': [5, 18, 22], 'incompatible': [6, 13], 'capacity_ledger': [7, 19], 'market': [8, 20], 'item': [9, 12], 'category': [15, 17]}
as_of_date = '2026-03-12'
tenant = 'NORTH'
item_dfs = [dfs[i] for i in table_map['item']]
item_df = pd.concat(item_dfs, ignore_index=True)
item_df = filter_latest(item_df, as_of_date, tenant=tenant)
item_df = item_df[item_df['item_ref'].str.strip() != '']
item_keys = list(item_df['item_ref'])
item_category = item_df.set_index('item_ref')['category'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].astype(float).to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].astype(float).to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].astype(float).to_dict()
cat_dfs = [dfs[i] for i in table_map['category']]
cat_df = pd.concat(cat_dfs, ignore_index=True)
cat_df = filter_latest(cat_df, as_of_date, tenant=tenant)
cat_df = cat_df[cat_df['category'].str.strip() != '']
category_keys = list(cat_df['category'].unique())
cat_min_qty = cat_df.set_index('category')['minimum_quantity'].astype(float).to_dict()
cat_max_qty = cat_df.set_index('category')['maximum_quantity'].astype(float).to_dict()
cat_activation_fee = cat_df.set_index('category')['activation_fee_cents'].astype(float).to_dict()
benefit_dfs = [dfs[i] for i in table_map['benefit']]
benefit_df = pd.concat(benefit_dfs, ignore_index=True)
benefit_df = filter_latest(benefit_df, as_of_date, tenant=tenant)
benefit_df = benefit_df[benefit_df['item_ref'].str.strip() != '']
benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='raise')
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in item_keys:
    if i not in item_benefit:
        item_benefit[i] = 0.0
item_fee_dfs = [dfs[i] for i in table_map['item_fee']]
item_fee_df = pd.concat(item_fee_dfs, ignore_index=True)
item_fee_df = filter_latest(item_fee_df, as_of_date, tenant=tenant)
item_fee_df = item_fee_df[item_fee_df['item_ref'].str.strip() != '']
item_activation_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].astype(float).to_dict()
for i in item_keys:
    if i not in item_activation_fee:
        item_activation_fee[i] = 0.0
bundle_dfs = [dfs[i] for i in table_map['bundle']]
bundle_df = pd.concat(bundle_dfs, ignore_index=True)
bundle_df = filter_latest(bundle_df, as_of_date, tenant=tenant)
bundle_df = bundle_df[(bundle_df['item_a'].str.strip() != '') & (bundle_df['item_b'].str.strip() != '')]
bundle_keys = list(bundle_df.index)
bundle_item_a = bundle_df['item_a'].to_dict()
bundle_item_b = bundle_df['item_b'].to_dict()
bundle_bonus = bundle_df['bonus_cents'].astype(float).to_dict()
incomp_dfs = [dfs[i] for i in table_map['incompatible']]
incomp_df = pd.concat(incomp_dfs, ignore_index=True)
incomp_df = filter_latest(incomp_df, as_of_date, tenant=tenant)
incomp_df = incomp_df[(incomp_df['item_a'].str.strip() != '') & (incomp_df['item_b'].str.strip() != '')]
incompatible_pairs = set()
for _, row in incomp_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a != '' and b != '':
        incompatible_pairs.add(frozenset([a, b]))
requires_dfs = [dfs[i] for i in table_map['requires']]
requires_df = pd.concat(requires_dfs, ignore_index=True)
requires_df = filter_latest(requires_df, as_of_date, tenant=tenant)
requires_df = requires_df[(requires_df['item_ref'].str.strip() != '') & (requires_df['prerequisite_ref'].str.strip() != '')]
requires_pairs = []
for _, row in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    requires_pairs.append((i, k))
usage_dfs = [dfs[i] for i in table_map['usage']]
usage_df = pd.concat(usage_dfs, ignore_index=True)
usage_df = filter_latest(usage_df, as_of_date, tenant=tenant)
usage_df = usage_df[(usage_df['item_ref'].str.strip() != '') & (usage_df['resource'].str.strip() != '')]
unit_map = {'ml': ('space', 1.0), 'liter': ('space', 1000.0), 'wh': ('power', 1.0), 'kwh': ('power', 1000.0), 'minute': ('labor', 1.0), 'hour': ('labor', 60.0)}
usage_per_unit = {}
for _, row in usage_df.iterrows():
    item = row['item_ref']
    resource = row['resource'].strip().casefold()
    amount = float(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_map:
        continue
    canonical_resource, factor = unit_map[unit]
    if item not in usage_per_unit:
        usage_per_unit[item] = {}
    usage_per_unit[item][canonical_resource] = usage_per_unit[item].get(canonical_resource, 0.0) + amount * factor
cap_dfs = [dfs[i] for i in table_map['capacity_ledger']]
cap_df = pd.concat(cap_dfs, ignore_index=True)
cap_df = filter_latest(cap_df, as_of_date, tenant=tenant)
cap_df = cap_df[(cap_df['resource'].str.strip() != '') & (cap_df['amount'].astype(str).str.strip() != '') & (cap_df['unit'].str.strip() != '')]
resource_capacity = {}
for _, row in cap_df.iterrows():
    resource = row['resource'].strip().casefold()
    amount = float(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_map:
        continue
    canonical_resource, factor = unit_map[unit]
    resource_capacity[canonical_resource] = resource_capacity.get(canonical_resource, 0.0) + amount * factor
items = sorted(item_keys)
categories = sorted(category_keys)
resources = sorted(resource_capacity.keys())
bundles = list(bundle_df.index)
bundle_valid = []
for b in bundles:
    a = bundle_item_a[b]
    bb = bundle_item_b[b]
    if a in items and bb in items:
        bundle_valid.append(b)
bundles = bundle_valid
m = gp.Model('vehicle_dealer_replenishment')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    min_lot = float(item_min_lot[i])
    max_order = float(item_max_order[i])
    authorized = float(item_authorized[i])
    if authorized <= 0:
        m.addConstr(q_vars[i] == 0, name=f'unauth_q_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= max_order * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(q_vars[i] <= max_order, name=f'maxorder2_{i}')
        m.addConstr(z_vars[i] <= 1, name=f'zbin_{i}')
for i in items:
    g = item_category[i]
    if g in categories:
        m.addConstr(z_vars[i] <= y_vars[g], name=f'catact_{i}')
for g in categories:
    members = [i for i in items if item_category[i] == g]
    if members:
        m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in members)), name=f'catflag_{g}')
for g in categories:
    members = [i for i in items if item_category[i] == g]
    if members:
        m.addConstr(gp.quicksum((q_vars[i] for i in members)) >= float(cat_min_qty[g]), name=f'catmin_{g}')
        m.addConstr(gp.quicksum((q_vars[i] for i in members)) <= float(cat_max_qty[g]), name=f'catmax_{g}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_unit.get(i, {}).get(r, 0.0) * q_vars[i] for i in items)) <= resource_capacity[r], name=f'rescap_{r}')
for pair in incompatible_pairs:
    pair = list(pair)
    if len(pair) == 2 and pair[0] in items and (pair[1] in items):
        m.addConstr(z_vars[pair[0]] + z_vars[pair[1]] <= 1, name=f'incomp_{pair[0]}_{pair[1]}')
for i, k in requires_pairs:
    if i in items and k in items:
        max_order_i = float(item_max_order[i])
        m.addConstr(q_vars[i] <= max_order_i * z_vars[k], name=f'req1_{i}_{k}')
        m.addConstr(z_vars[i] <= z_vars[k], name=f'req2_{i}_{k}')
for b in bundles:
    a = bundle_item_a[b]
    bb = bundle_item_b[b]
    if a in items and bb in items:
        m.addConstr(w_vars[b] <= z_vars[a], name=f'bundle1_{b}')
        m.addConstr(w_vars[b] <= z_vars[bb], name=f'bundle2_{b}')
        m.addConstr(w_vars[b] >= z_vars[a] + z_vars[bb] - 1, name=f'bundle3_{b}')
item_benefit_expr = gp.quicksum((item_benefit[i] * q_vars[i] for i in items))
item_fee_expr = gp.quicksum((item_activation_fee[i] * z_vars[i] for i in items))
cat_fee_expr = gp.quicksum((cat_activation_fee[g] * y_vars[g] for g in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundles))
m.setObjective(item_benefit_expr - item_fee_expr - cat_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.optimize()