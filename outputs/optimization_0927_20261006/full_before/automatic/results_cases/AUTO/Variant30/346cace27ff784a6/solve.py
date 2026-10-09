import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = [pd.read_csv(p, sep=',') for p in paths]

def select_latest(df, key_cols, date_col, cutoff, action_col='action', revision_col='revision'):
    df = df[df[date_col] <= cutoff]
    df = df.sort_values(key_cols + [revision_col], ascending=True)
    df = df.drop_duplicates(subset=key_cols + [revision_col], keep='last')
    idx = df.groupby(key_cols)[revision_col].transform('max') == df[revision_col]
    df = df[idx]
    if action_col in df.columns:
        df = df[df[action_col].str.casefold() != 'delete']
    return df
table_map = {}
for df in dfs:
    if 'table' in df.columns:
        for t in df['table'].unique():
            if pd.notnull(t):
                table_map.setdefault(t.strip().casefold(), []).append(df)

def get_table(table_name):
    dfs_ = table_map.get(table_name.strip().casefold(), [])
    if not dfs_:
        return pd.DataFrame()
    return pd.concat(dfs_, ignore_index=True)
CUTOFF_DATE = '2026-03-12'
TENANT = 'NORTH'
item_tables = []
for t in ['item']:
    df = get_table(t)
    if not df.empty:
        item_tables.append(df)
items_df = pd.concat(item_tables, ignore_index=True) if item_tables else pd.DataFrame()
items_df = items_df[items_df['tenant'].str.casefold() == TENANT.casefold()]
items_df = select_latest(items_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
for col in ['authorized', 'minimum_lot', 'maximum_order']:
    if col in items_df.columns:
        items_df[col] = pd.to_numeric(items_df[col], errors='coerce')
items_df = items_df[items_df['item_ref'].notnull()]
item_ids = items_df['item_ref'].unique().tolist()
cat_tables = []
for t in ['category']:
    df = get_table(t)
    if not df.empty:
        cat_tables.append(df)
categories_df = pd.concat(cat_tables, ignore_index=True) if cat_tables else pd.DataFrame()
categories_df = categories_df[categories_df['tenant'].str.casefold() == TENANT.casefold()]
categories_df = select_latest(categories_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
categories_df = categories_df[categories_df['category'].notnull()]
category_ids = categories_df['category'].unique().tolist()
item_fee_tables = []
for t in ['item_fee']:
    df = get_table(t)
    if not df.empty:
        item_fee_tables.append(df)
item_fee_df = pd.concat(item_fee_tables, ignore_index=True) if item_fee_tables else pd.DataFrame()
item_fee_df = item_fee_df[item_fee_df['tenant'].str.casefold() == TENANT.casefold()]
item_fee_df = select_latest(item_fee_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
item_fee_df = item_fee_df[item_fee_df['item_ref'].notnull()]
benefit_tables = []
for t in ['benefit']:
    df = get_table(t)
    if not df.empty:
        benefit_tables.append(df)
benefit_df = pd.concat(benefit_tables, ignore_index=True) if benefit_tables else pd.DataFrame()
benefit_df = benefit_df[benefit_df['tenant'].str.casefold() == TENANT.casefold()]
benefit_df = select_latest(benefit_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
benefit_df = benefit_df[benefit_df['item_ref'].notnull()]
usage_tables = []
for t in ['usage']:
    df = get_table(t)
    if not df.empty:
        usage_tables.append(df)
usage_df = pd.concat(usage_tables, ignore_index=True) if usage_tables else pd.DataFrame()
usage_df = usage_df[usage_df['tenant'].str.casefold() == TENANT.casefold()]
usage_df = select_latest(usage_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
usage_df = usage_df[usage_df['item_ref'].notnull() & usage_df['resource'].notnull() & usage_df['unit'].notnull()]
cap_tables = []
for t in ['capacity_ledger']:
    df = get_table(t)
    if not df.empty:
        cap_tables.append(df)
capacity_df = pd.concat(cap_tables, ignore_index=True) if cap_tables else pd.DataFrame()
capacity_df = capacity_df[capacity_df['tenant'].str.casefold() == TENANT.casefold()]
capacity_df = select_latest(capacity_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
capacity_df = capacity_df[capacity_df['resource'].notnull() & capacity_df['unit'].notnull() & capacity_df['amount'].notnull()]
incomp_tables = []
for t in ['incompatible']:
    df = get_table(t)
    if not df.empty:
        incomp_tables.append(df)
incomp_df = pd.concat(incomp_tables, ignore_index=True) if incomp_tables else pd.DataFrame()
incomp_df = incomp_df[incomp_df['tenant'].str.casefold() == TENANT.casefold()]
incomp_df = select_latest(incomp_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
incomp_df = incomp_df[incomp_df['item_a'].notnull() & incomp_df['item_b'].notnull()]
requires_tables = []
for t in ['requires']:
    df = get_table(t)
    if not df.empty:
        requires_tables.append(df)
requires_df = pd.concat(requires_tables, ignore_index=True) if requires_tables else pd.DataFrame()
requires_df = requires_df[requires_df['tenant'].str.casefold() == TENANT.casefold()]
requires_df = select_latest(requires_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
requires_df = requires_df[requires_df['item_ref'].notnull() & requires_df['prerequisite_ref'].notnull()]
bundle_tables = []
for t in ['bundle']:
    df = get_table(t)
    if not df.empty:
        bundle_tables.append(df)
bundle_df = pd.concat(bundle_tables, ignore_index=True) if bundle_tables else pd.DataFrame()
bundle_df = bundle_df[bundle_df['tenant'].str.casefold() == TENANT.casefold()]
bundle_df = select_latest(bundle_df, ['tenant', 'table', 'record_id'], 'effective_date', CUTOFF_DATE)
bundle_df = bundle_df[bundle_df['item_a'].notnull() & bundle_df['item_b'].notnull()]
benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_fee_per_item = {}
for (_, row) in item_fee_df.iterrows():
    item_fee_per_item[str(row['item_ref'])] = float(row['activation_fee_cents'])
cat_fee_per_cat = {}
for (_, row) in categories_df.iterrows():
    cat_fee_per_cat[str(row['category'])] = float(row['activation_fee_cents'])
item_bounds = {}
item_category = {}
item_authorized = {}
for (_, row) in items_df.iterrows():
    i = str(row['item_ref'])
    item_bounds[i] = (int(row['minimum_lot']), int(row['maximum_order']))
    item_category[i] = str(row['category'])
    item_authorized[i] = int(row['authorized'])
cat_qty_limits = {}
for (_, row) in categories_df.iterrows():
    g = str(row['category'])
    cat_qty_limits[g] = (int(row['minimum_quantity']), int(row['maximum_quantity']))
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    amt = float(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        amt = amt * 1000
        unit = 'ml'
    elif unit == 'kwh':
        amt = amt * 1000
        unit = 'wh'
    elif unit == 'hour':
        amt = amt * 60
        unit = 'minute'
    usage_per_item_resource[i, r] = (amt, unit)
resource_capacity = {}
for (_, group) in capacity_df.groupby(['resource', 'unit']):
    r = str(group['resource'].iloc[0])
    unit = str(group['unit'].iloc[0]).strip().lower()
    total = group['amount'].astype(float).sum()
    if unit == 'liter':
        total = total * 1000
        unit = 'ml'
    elif unit == 'kwh':
        total = total * 1000
        unit = 'wh'
    elif unit == 'hour':
        total = total * 60
        unit = 'minute'
    resource_capacity[r, unit] = total
resource_base_unit = {}
for (r, unit) in resource_capacity:
    resource_base_unit[r] = unit
incompat_pairs = []
for (_, row) in incomp_df.iterrows():
    i = str(row['item_a'])
    j = str(row['item_b'])
    if i in item_ids and j in item_ids:
        incompat_pairs.append((i, j))
requires_pairs = []
for (_, row) in requires_df.iterrows():
    i = str(row['item_ref'])
    k = str(row['prerequisite_ref'])
    if i in item_ids and k in item_ids:
        requires_pairs.append((i, k))
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_ids and b in item_ids:
        bundle_bonus[a, b] = float(row['bonus_cents'])
cat_items = {}
for i in item_ids:
    g = item_category[i]
    cat_items.setdefault(g, []).append(i)
item_to_cat = {i: item_category[i] for i in item_ids}
m = gp.Model('inventory_replenishment')
x = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
w = m.addVars(bundle_bonus.keys(), vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    (min_lot, max_order) = item_bounds[i]
    if item_authorized[i] == 0:
        m.addConstr(x[i] == 0)
        m.addConstr(y[i] == 0)
    else:
        m.addConstr(x[i] >= min_lot * y[i])
        m.addConstr(x[i] <= max_order * y[i])
        m.addConstr(y[i] <= 1)
        m.addConstr(x[i] >= 0)
for g in category_ids:
    items_in_g = cat_items.get(g, [])
    if not items_in_g:
        m.addConstr(z[g] == 0)
        continue
    (min_qty, max_qty) = cat_qty_limits[g]
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= min_qty * z[g])
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= max_qty * z[g])
    for i in items_in_g:
        m.addConstr(y[i] <= z[g])
for r in resource_base_unit:
    unit = resource_base_unit[r]
    usage_terms = []
    for i in item_ids:
        key = (i, r)
        if key in usage_per_item_resource:
            (amt, u) = usage_per_item_resource[key]
            if u != unit:
                if u == 'liter' and unit == 'ml':
                    amt = amt * 1000
                elif u == 'ml' and unit == 'liter':
                    amt = amt / 1000
                elif u == 'kwh' and unit == 'wh':
                    amt = amt * 1000
                elif u == 'wh' and unit == 'kwh':
                    amt = amt / 1000
                elif u == 'hour' and unit == 'minute':
                    amt = amt * 60
                elif u == 'minute' and unit == 'hour':
                    amt = amt / 60
                else:
                    raise ValueError(f'Unknown unit conversion from {u} to {unit}')
            usage_terms.append(amt * x[i])
    if usage_terms:
        m.addConstr(gp.quicksum(usage_terms) <= resource_capacity[r, unit])
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1)
for (i, k) in requires_pairs:
    m.addConstr(y[i] <= y[k])
    m.addConstr(x[i] <= sum(item_bounds.values(), (0, 0))[1] * y[k])
for (a, b) in bundle_bonus:
    m.addConstr(w[a, b] <= y[a])
    m.addConstr(w[a, b] <= y[b])
    m.addConstr(w[a, b] >= y[a] + y[b] - 1)
obj_benefit = gp.quicksum((benefit_per_item.get(i, 0.0) * x[i] for i in item_ids))
obj_item_fee = gp.quicksum((item_fee_per_item.get(i, 0.0) * y[i] for i in item_ids))
obj_cat_fee = gp.quicksum((cat_fee_per_cat.get(g, 0.0) * z[g] for g in category_ids))
obj_bundle = gp.quicksum((bundle_bonus[a, b] * w[a, b] for (a, b) in bundle_bonus))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')