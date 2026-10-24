import gurobipy as gp
import pandas as pd
import numpy as np
import re

def filter_latest(df, as_of_date, tenant, table=None):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    if table is not None:
        df = df[df['table'].str.strip().str.casefold() == table.casefold()]
    df = df[df['effective_date'] <= as_of_date]
    df = df.copy()
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df = df.sort_values(['tenant', 'table', 'record_id', 'revision', 'effective_date'], ascending=[True, True, True, False, False])
    df = df.drop_duplicates(['tenant', 'table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_latest_no_record_id(df, as_of_date, tenant, table=None):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    if table is not None:
        df = df[df['table'].str.strip().str.casefold() == table.casefold()]
    df = df[df['effective_date'] <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    dedup_cols = [c for c in df.columns if c not in ['revision', 'effective_date']]
    df = df.sort_values(dedup_cols + ['revision', 'effective_date'], ascending=[True] * len(dedup_cols) + [False, False])
    df = df.drop_duplicates(dedup_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def convert_resource(amount, unit, resource):
    unit = unit.strip().casefold()
    resource = resource.strip().casefold()
    if resource == 'space':
        if unit == 'ml':
            return float(amount)
        elif unit == 'liter':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return float(amount)
        elif unit == 'kwh':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return float(amount)
        elif unit == 'hour':
            return float(amount) * 60.0
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
as_of_date = '2026-03-12'
tenant = 'NORTH'
paths = ['benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', 'benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', 'benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', 'benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', 'benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', 'benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', 'benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', 'benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', 'benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', 'benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', 'benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', 'benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', 'benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', 'benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', 'benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', 'benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = []
for path in paths:
    dfs.append(pd.read_csv(path, dtype=str, keep_default_na=False))

def concat_tables(dfs, table_name):
    return pd.concat([df for df in dfs if df['table'].str.strip().str.casefold().eq(table_name.casefold()).any()], ignore_index=True)
item_df = pd.concat([filter_latest(dfs[9], as_of_date, tenant, 'item'), filter_latest(dfs[12], as_of_date, tenant, 'item')], ignore_index=True)
item_df = item_df[item_df['item_ref'].str.strip() != '']
category_df = pd.concat([filter_latest(dfs[15], as_of_date, tenant, 'category'), filter_latest(dfs[17], as_of_date, tenant, 'category')], ignore_index=True)
category_df = category_df[category_df['category'].str.strip() != '']
benefit_df = pd.concat([filter_latest(dfs[4], as_of_date, tenant, 'benefit'), filter_latest(dfs[11], as_of_date, tenant, 'benefit'), filter_latest(dfs[14], as_of_date, tenant, 'benefit')], ignore_index=True)
benefit_df = benefit_df[benefit_df['item_ref'].str.strip() != '']
item_fee_df = pd.concat([filter_latest(dfs[2], as_of_date, tenant, 'item_fee'), filter_latest(dfs[16], as_of_date, tenant, 'item_fee')], ignore_index=True)
item_fee_df = item_fee_df[item_fee_df['item_ref'].str.strip() != '']
bundle_df = pd.concat([filter_latest_no_record_id(dfs[3], as_of_date, tenant, 'bundle'), filter_latest_no_record_id(dfs[10], as_of_date, tenant, 'bundle')], ignore_index=True)
bundle_df = bundle_df[(bundle_df['item_a'].str.strip() != '') & (bundle_df['item_b'].str.strip() != '')]
usage_df = pd.concat([filter_latest(dfs[5], as_of_date, tenant, 'usage'), filter_latest(dfs[18], as_of_date, tenant, 'usage'), filter_latest(dfs[22], as_of_date, tenant, 'usage')], ignore_index=True)
usage_df = usage_df[usage_df['item_ref'].str.strip() != '']
capacity_df = pd.concat([filter_latest(dfs[7], as_of_date, tenant, 'capacity_ledger'), filter_latest(dfs[19], as_of_date, tenant, 'capacity_ledger')], ignore_index=True)
capacity_df = capacity_df[capacity_df['resource'].str.strip() != '']
incompat_df = pd.concat([filter_latest_no_record_id(dfs[6], as_of_date, tenant, 'incompatible'), filter_latest_no_record_id(dfs[13], as_of_date, tenant, 'incompatible')], ignore_index=True)
incompat_df = incompat_df[(incompat_df['item_a'].str.strip() != '') & (incompat_df['item_b'].str.strip() != '')]
requires_df = pd.concat([filter_latest_no_record_id(dfs[0], as_of_date, tenant, 'requires'), filter_latest_no_record_id(dfs[23], as_of_date, tenant, 'requires')], ignore_index=True)
requires_df = requires_df[(requires_df['item_ref'].str.strip() != '') & (requires_df['prerequisite_ref'].str.strip() != '')]
identity_df = pd.concat([filter_latest(dfs[1], as_of_date, tenant, 'identity'), filter_latest(dfs[21], as_of_date, tenant, 'identity')], ignore_index=True)
identity_df = identity_df[identity_df['ref'].str.strip() != '']
items = sorted(item_df['item_ref'].unique())
categories = sorted(category_df['category'].unique())
resource_set = set()
for df in [usage_df, capacity_df]:
    resource_set.update(df['resource'].str.strip().str.casefold().unique())
resources = sorted(resource_set)
bundles_set = set()
for (_, row) in bundle_df.iterrows():
    key = (row['item_a'], row['item_b'])
    bundles_set.add(key)
bundles = sorted(bundles_set)
incompat_pairs = set()
for (_, row) in incompat_df.iterrows():
    incompat_pairs.add((row['item_a'], row['item_b']))
    incompat_pairs.add((row['item_b'], row['item_a']))
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    requires_pairs.add((row['item_ref'], row['prerequisite_ref']))
benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='raise')
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in item_benefit:
        item_benefit[i] = 0
item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'], errors='raise')
item_activation_fee = {}
for i in items:
    fee_rows = item_fee_df[item_fee_df['item_ref'] == i]
    if not fee_rows.empty:
        item_activation_fee[i] = float(fee_rows.iloc[0]['activation_fee_cents'])
    else:
        item_activation_fee[i] = 0.0
category_df['activation_fee_cents'] = pd.to_numeric(category_df['activation_fee_cents'], errors='raise')
category_activation_fee = {}
for g in categories:
    fee_rows = category_df[category_df['category'] == g]
    if not fee_rows.empty:
        category_activation_fee[g] = float(fee_rows.iloc[0]['activation_fee_cents'])
    else:
        category_activation_fee[g] = 0.0
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    key = (row['item_a'], row['item_b'])
    try:
        bundle_bonus[key] = float(row['bonus_cents'])
    except Exception:
        bundle_bonus[key] = 0.0
item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'], errors='raise')
item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'], errors='raise')
item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='raise')
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_category = item_df.set_index('item_ref')['category'].to_dict()
category_df['minimum_quantity'] = pd.to_numeric(category_df['minimum_quantity'], errors='raise')
category_df['maximum_quantity'] = pd.to_numeric(category_df['maximum_quantity'], errors='raise')
category_min_qty = category_df.set_index('category')['minimum_quantity'].to_dict()
category_max_qty = category_df.set_index('category')['maximum_quantity'].to_dict()
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource'].strip().casefold()
    try:
        amt = convert_resource(row['amount'], row['unit'], r)
    except Exception:
        continue
    usage_per_item_resource.setdefault(i, {})[r] = amt
capacity_df = capacity_df[capacity_df['amount'].str.strip() != '']
capacity_df['amount'] = pd.to_numeric(capacity_df['amount'], errors='raise')
resource_capacity = {}
for r in resources:
    total = 0.0
    for (_, row) in capacity_df.iterrows():
        if row['resource'].strip().casefold() == r:
            try:
                amt = convert_resource(row['amount'], row['unit'], r)
            except Exception:
                continue
            total += amt
    resource_capacity[r] = total
category_items = {g: [] for g in categories}
for i in items:
    g = item_category.get(i, None)
    if g is not None and g in categories:
        category_items[g].append(i)
m = gp.Model('vehicle_dealer_replenishment')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    auth = item_authorized.get(i, 0)
    min_lot = item_min_lot.get(i, 0)
    max_order = item_max_order.get(i, 0)
    if auth > 0:
        m.addConstr(q_vars[i] >= min_lot * y_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= max_order * y_vars[i], name=f'maxorder_{i}')
    else:
        m.addConstr(q_vars[i] == 0, name=f'unauth_q_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    m.addConstr(q_vars[i] >= y_vars[i], name=f'y_link_lb_{i}')
    m.addConstr(q_vars[i] <= max_order * y_vars[i], name=f'y_link_ub_{i}')
for g in categories:
    items_in_g = category_items[g]
    if items_in_g:
        m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)), name=f'cat_z_ub_{g}')
        for i in items_in_g:
            m.addConstr(y_vars[i] <= z_vars[g], name=f'cat_y_leq_z_{i}_{g}')
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= category_min_qty[g], name=f'cat_minqty_{g}')
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= category_max_qty[g], name=f'cat_maxqty_{g}')
    else:
        m.addConstr(z_vars[g] == 0, name=f'cat_z_zero_{g}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource.get(i, {}).get(r, 0.0) * q_vars[i] for i in items)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, k) in requires_pairs:
    if i in items and k in items:
        m.addConstr(y_vars[i] <= y_vars[k], name=f'requires_{i}_{k}')
for (i, j) in bundles:
    if i in items and j in items:
        m.addConstr(w_vars[i, j] <= y_vars[i], name=f'bundle_w_leq_yi_{i}_{j}')
        m.addConstr(w_vars[i, j] <= y_vars[j], name=f'bundle_w_leq_yj_{i}_{j}')
        m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1, name=f'bundle_w_geq_sum_{i}_{j}')
    else:
        m.addConstr(w_vars[i, j] == 0, name=f'bundle_w_zero_{i}_{j}')
obj_benefit = gp.quicksum((item_benefit.get(i, 0.0) * q_vars[i] for i in items))
obj_item_fee = gp.quicksum((item_activation_fee.get(i, 0.0) * y_vars[i] for i in items))
obj_cat_fee = gp.quicksum((category_activation_fee.get(g, 0.0) * z_vars[g] for g in categories))
obj_bundle = gp.quicksum((bundle_bonus.get((i, j), 0.0) * w_vars[i, j] for (i, j) in bundles))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()