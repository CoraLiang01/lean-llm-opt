import gurobipy as gp
import pandas as pd
import numpy as np
import re

def canonicalize_resource(resource, amount, unit):
    resource = resource.strip().casefold()
    unit = unit.strip().casefold()
    if resource == 'space':
        if unit == 'ml':
            return float(amount)
        elif unit == 'liter':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return float(amount)
        elif unit == 'hour':
            return float(amount) * 60.0
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return float(amount)
        elif unit == 'kwh':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')

def select_latest(df, asof_date):
    df = df[df['tenant'].str.strip().str.casefold() == 'north']
    df = df[df['effective_date'] <= asof_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(['tenant', 'table', 'record_id', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=['tenant', 'table', 'record_id', 'revision'], keep='first')
    df = df.drop_duplicates(subset=['tenant', 'table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
asof_date = '2026-03-12'
identity_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'identity').any():
        identity_dfs.append(df)
identity_df = pd.concat(identity_dfs, ignore_index=True) if identity_dfs else pd.DataFrame()
identity_df = select_latest(identity_df, asof_date)
identity_df = identity_df[identity_df['kind'].str.strip().str.casefold() == 'item']
itemref_to_entityid = dict(zip(identity_df['ref'], identity_df['entity_id']))
item_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'item').any():
        item_dfs.append(df)
item_df = pd.concat(item_dfs, ignore_index=True) if item_dfs else pd.DataFrame()
item_df = select_latest(item_df, asof_date)
item_df = item_df[item_df['item_ref'] != '']
item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='raise')
item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'], errors='raise')
item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'], errors='raise')
item_df['category'] = item_df['category'].str.strip()
item_df['item_ref'] = item_df['item_ref'].str.strip()
item_df = item_df.reset_index(drop=True)
category_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'category').any():
        category_dfs.append(df)
category_df = pd.concat(category_dfs, ignore_index=True) if category_dfs else pd.DataFrame()
category_df = select_latest(category_df, asof_date)
category_df['category'] = category_df['category'].str.strip()
category_df['minimum_quantity'] = pd.to_numeric(category_df['minimum_quantity'], errors='raise')
category_df['maximum_quantity'] = pd.to_numeric(category_df['maximum_quantity'], errors='raise')
category_df['activation_fee_cents'] = pd.to_numeric(category_df['activation_fee_cents'], errors='raise')
category_df = category_df.reset_index(drop=True)
benefit_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'benefit').any():
        benefit_dfs.append(df)
benefit_df = pd.concat(benefit_dfs, ignore_index=True) if benefit_dfs else pd.DataFrame()
benefit_df = select_latest(benefit_df, asof_date)
benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='raise')
benefit_df['item_ref'] = benefit_df['item_ref'].str.strip()
benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_fee_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'item_fee').any():
        item_fee_dfs.append(df)
item_fee_df = pd.concat(item_fee_dfs, ignore_index=True) if item_fee_dfs else pd.DataFrame()
item_fee_df = select_latest(item_fee_df, asof_date)
item_fee_df = item_fee_df[item_fee_df['item_ref'] != '']
item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'], errors='raise')
item_fee_df['item_ref'] = item_fee_df['item_ref'].str.strip()
item_fee_per_item = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
usage_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'usage').any():
        usage_dfs.append(df)
usage_df = pd.concat(usage_dfs, ignore_index=True) if usage_dfs else pd.DataFrame()
usage_df = select_latest(usage_df, asof_date)
usage_df = usage_df[usage_df['item_ref'] != '']
usage_df['amount'] = pd.to_numeric(usage_df['amount'], errors='raise')
usage_df['resource'] = usage_df['resource'].str.strip().str.casefold()
usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()
usage_df['item_ref'] = usage_df['item_ref'].str.strip()
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    key = (row['item_ref'], row['resource'])
    val = canonicalize_resource(row['resource'], row['amount'], row['unit'])
    usage_per_item_resource[key] = val
capacity_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'capacity_ledger').any():
        capacity_dfs.append(df)
capacity_df = pd.concat(capacity_dfs, ignore_index=True) if capacity_dfs else pd.DataFrame()
capacity_df = select_latest(capacity_df, asof_date)
capacity_df = capacity_df[capacity_df['resource'] != '']
capacity_df['amount'] = pd.to_numeric(capacity_df['amount'], errors='raise')
capacity_df['resource'] = capacity_df['resource'].str.strip().str.casefold()
capacity_df['unit'] = capacity_df['unit'].str.strip().str.casefold()
capacity_per_resource = {}
for ((resource, unit), group) in capacity_df.groupby(['resource', 'unit']):
    total = group['amount'].sum()
    canonical = canonicalize_resource(resource, total, unit)
    if resource in capacity_per_resource:
        capacity_per_resource[resource] += canonical
    else:
        capacity_per_resource[resource] = canonical
bundle_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'bundle').any():
        bundle_dfs.append(df)
bundle_df = pd.concat(bundle_dfs, ignore_index=True) if bundle_dfs else pd.DataFrame()
bundle_df = select_latest(bundle_df, asof_date)
bundle_df = bundle_df[(bundle_df['item_a'] != '') & (bundle_df['item_b'] != '')]
bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'], errors='raise')
bundle_df['item_a'] = bundle_df['item_a'].str.strip()
bundle_df['item_b'] = bundle_df['item_b'].str.strip()
bundle_tuples = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    key = (row['item_a'], row['item_b'])
    bundle_tuples.append(key)
    bundle_bonus[key] = row['bonus_cents']
incompatible_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'incompatible').any():
        incompatible_dfs.append(df)
incompatible_df = pd.concat(incompatible_dfs, ignore_index=True) if incompatible_dfs else pd.DataFrame()
incompatible_df = select_latest(incompatible_df, asof_date)
incompatible_df = incompatible_df[(incompatible_df['item_a'] != '') & (incompatible_df['item_b'] != '')]
incompatible_df['item_a'] = incompatible_df['item_a'].str.strip()
incompatible_df['item_b'] = incompatible_df['item_b'].str.strip()
incompatible_pairs = set()
for (_, row) in incompatible_df.iterrows():
    incompatible_pairs.add((row['item_a'], row['item_b']))
    incompatible_pairs.add((row['item_b'], row['item_a']))
requires_dfs = []
for df in dfs:
    if 'table' in df.columns and (df['table'].str.strip().str.casefold() == 'requires').any():
        requires_dfs.append(df)
requires_df = pd.concat(requires_dfs, ignore_index=True) if requires_dfs else pd.DataFrame()
requires_df = select_latest(requires_df, asof_date)
requires_df = requires_df[(requires_df['item_ref'] != '') & (requires_df['prerequisite_ref'] != '')]
requires_df['item_ref'] = requires_df['item_ref'].str.strip()
requires_df['prerequisite_ref'] = requires_df['prerequisite_ref'].str.strip()
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    requires_pairs.add((row['item_ref'], row['prerequisite_ref']))
item_refs = list(item_df['item_ref'])
item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
item_authorized = dict(zip(item_df['item_ref'], item_df['authorized']))
item_min_lot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_max_order = dict(zip(item_df['item_ref'], item_df['maximum_order']))
categories = list(category_df['category'])
cat_min_qty = dict(zip(category_df['category'], category_df['minimum_quantity']))
cat_max_qty = dict(zip(category_df['category'], category_df['maximum_quantity']))
cat_activation_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
cat_to_items = {g: [] for g in categories}
for i in item_refs:
    g = item_to_category[i]
    if g in cat_to_items:
        cat_to_items[g].append(i)
    else:
        cat_to_items[g] = [i]
resources = set()
for (i, r) in usage_per_item_resource:
    resources.add(r)
for r in capacity_per_resource:
    resources.add(r)
resources = list(resources)
m = gp.Model('vehicle_dealer_replenishment')
q_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth > 0:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i])
        m.addConstr(q_vars[i] <= max_order * z_vars[i])
        m.addConstr(q_vars[i] >= 0)
        m.addConstr(z_vars[i] <= 1)
        m.addConstr((q_vars[i] >= 1) >> (z_vars[i] == 1))
        m.addConstr((z_vars[i] == 0) >> (q_vars[i] == 0))
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for g in categories:
    items_in_g = cat_to_items.get(g, [])
    if not items_in_g:
        m.addConstr(y_vars[g] == 0)
        continue
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= cat_min_qty[g] * y_vars[g])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= cat_max_qty[g] * y_vars[g])
    for i in items_in_g:
        m.addConstr(z_vars[i] <= y_vars[g])
    m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in items_in_g)))
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource.get((i, r), 0.0) * q_vars[i] for i in item_refs)) <= capacity_per_resource.get(r, 0.0))
for (i, j) in incompatible_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, k) in requires_pairs:
    if i in item_refs and k in item_refs:
        m.addConstr(z_vars[i] <= z_vars[k])
for b in bundle_tuples:
    (i, j) = b
    if i in item_refs and j in item_refs:
        m.addConstr(w_vars[b] <= z_vars[i])
        m.addConstr(w_vars[b] <= z_vars[j])
        m.addConstr(w_vars[b] >= z_vars[i] + z_vars[j] - 1)
    else:
        m.addConstr(w_vars[b] == 0)
obj_benefit = gp.quicksum((benefit_per_item.get(i, 0.0) * q_vars[i] for i in item_refs))
obj_item_fee = gp.quicksum((item_fee_per_item.get(i, 0.0) * z_vars[i] for i in item_refs))
obj_cat_fee = gp.quicksum((cat_activation_fee.get(g, 0.0) * y_vars[g] for g in categories))
obj_bundle = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_tuples))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()