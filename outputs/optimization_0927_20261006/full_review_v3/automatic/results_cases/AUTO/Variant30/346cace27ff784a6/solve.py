import gurobipy as gp
import pandas as pd
import numpy as np
import re

def latest_valid(df, cutoff_date, tenant):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= cutoff_date]
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values(['tenant', 'table', 'record_id', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(['tenant', 'table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def dropna_keys(df, keys):
    return df.dropna(subset=keys)

def convert_unit(amount, from_unit, to_unit):
    u_from = from_unit.strip().casefold()
    u_to = to_unit.strip().casefold()
    if u_from == u_to:
        return amount
    if u_from == 'ml' and u_to == 'liter':
        return amount / 1000.0
    if u_from == 'liter' and u_to == 'ml':
        return amount * 1000.0
    if u_from == 'minute' and u_to == 'hour':
        return amount / 60.0
    if u_from == 'hour' and u_to == 'minute':
        return amount * 60.0
    if u_from == 'wh' and u_to == 'kwh':
        return amount / 1000.0
    if u_from == 'kwh' and u_to == 'wh':
        return amount * 1000.0
    raise ValueError(f'Cannot convert from {from_unit} to {to_unit}')
cutoff_date = '2026-03-12'
tenant = 'NORTH'
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
table_dfs = {}
for df in dfs:
    if 'table' in df.columns:
        for t in df['table'].unique():
            if t not in table_dfs:
                table_dfs[t] = []
            table_dfs[t].append(df[df['table'] == t])

def get_latest_table(table, cutoff_date, tenant):
    if table not in table_dfs:
        return pd.DataFrame()
    df = pd.concat(table_dfs[table], ignore_index=True)
    return latest_valid(df, cutoff_date, tenant)
item_df = pd.concat([get_latest_table('item', cutoff_date, tenant), get_latest_table('item', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
item_df = dropna_keys(item_df, ['item_ref', 'category'])
items = sorted(item_df['item_ref'].unique())
categories = sorted(item_df['category'].unique())
item_to_category = item_df.set_index('item_ref')['category'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].astype(float).to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].astype(float).to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].astype(float).to_dict()
cat_df = pd.concat([get_latest_table('category', cutoff_date, tenant), get_latest_table('category', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
cat_df = dropna_keys(cat_df, ['category'])
cat_min_qty = cat_df.set_index('category')['minimum_quantity'].astype(float).to_dict()
cat_max_qty = cat_df.set_index('category')['maximum_quantity'].astype(float).to_dict()
cat_activation_fee = cat_df.set_index('category')['activation_fee_cents'].astype(float).to_dict()
item_fee_df = pd.concat([get_latest_table('item_fee', cutoff_date, tenant), get_latest_table('item_fee', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
item_fee_df = dropna_keys(item_fee_df, ['item_ref'])
item_activation_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].astype(float).to_dict()
benefit_df = pd.concat([get_latest_table('benefit', cutoff_date, tenant), get_latest_table('benefit', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
benefit_df = dropna_keys(benefit_df, ['item_ref', 'amount_cents'])
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(float)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in item_benefit:
        item_benefit[i] = 0.0
bundle_df = pd.concat([get_latest_table('bundle', cutoff_date, tenant), get_latest_table('bundle', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
bundle_df = dropna_keys(bundle_df, ['item_a', 'item_b', 'bonus_cents'])
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(float)
bundles = []
bundle_bonus = {}
for (idx, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    key = (a, b)
    bundles.append(key)
    bundle_bonus[key] = row['bonus_cents']
incomp_df = pd.concat([get_latest_table('incompatible', cutoff_date, tenant), get_latest_table('incompatible', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
incomp_df = dropna_keys(incomp_df, ['item_a', 'item_b'])
incompatible_pairs = set()
for (idx, row) in incomp_df.iterrows():
    (i, j) = (row['item_a'], row['item_b'])
    incompatible_pairs.add((i, j))
    incompatible_pairs.add((j, i))
requires_df = pd.concat([get_latest_table('requires', cutoff_date, tenant), get_latest_table('requires', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
requires_df = dropna_keys(requires_df, ['item_ref', 'prerequisite_ref'])
requires_pairs = set()
for (idx, row) in requires_df.iterrows():
    (i, j) = (row['item_ref'], row['prerequisite_ref'])
    requires_pairs.add((i, j))
usage_df = pd.concat([get_latest_table('usage', cutoff_date, tenant), get_latest_table('usage', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
usage_df = dropna_keys(usage_df, ['item_ref', 'resource', 'amount', 'unit'])
usage_df['amount'] = usage_df['amount'].astype(float)
item_resource_usage = {}
resources = set()
for (idx, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    unit = row['unit']
    item_resource_usage[i, r] = (amt, unit)
    resources.add(r)
cap_ledger_df = pd.concat([get_latest_table('capacity_ledger', cutoff_date, tenant), get_latest_table('capacity_ledger', cutoff_date, tenant)], ignore_index=True).drop_duplicates()
cap_ledger_df = dropna_keys(cap_ledger_df, ['resource', 'amount', 'unit'])
cap_ledger_df['amount'] = cap_ledger_df['amount'].astype(float)
resource_capacity = {}
resource_unit = {}
for r in resources:
    rows = cap_ledger_df[cap_ledger_df['resource'].str.strip().str.casefold() == r.strip().casefold()]
    if rows.empty:
        continue
    units = rows['unit'].dropna().unique()
    if len(units) == 0:
        continue
    canonical_unit = units[0]
    total = 0.0
    for (idx2, row2) in rows.iterrows():
        amt = row2['amount']
        unit = row2['unit']
        amt_canon = convert_unit(amt, unit, canonical_unit)
        total += amt_canon
    resource_capacity[r] = total
    resource_unit[r] = canonical_unit
for i in items:
    if i not in item_activation_fee:
        item_activation_fee[i] = 0.0
for g in categories:
    if g not in cat_activation_fee:
        cat_activation_fee[g] = 0.0
for g in categories:
    if g not in cat_min_qty:
        cat_min_qty[g] = 0.0
    if g not in cat_max_qty:
        cat_max_qty[g] = float('inf')
for i in items:
    if i not in item_min_lot:
        item_min_lot[i] = 0.0
    if i not in item_max_order:
        item_max_order[i] = 0.0
    if i not in item_authorized:
        item_authorized[i] = 0.0
for i in items:
    for r in resources:
        if (i, r) not in item_resource_usage:
            item_resource_usage[i, r] = (0.0, resource_unit[r])
bundles = [b for b in bundles if b[0] in items and b[1] in items]
m = gp.Model('vehicle_dealer_replenishment')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    if item_authorized[i] > 0:
        m.addConstr(x_vars[i] >= item_min_lot[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= item_max_order[i], name=f'maxorder_{i}')
    else:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
bigM = max([item_max_order[i] for i in items] + [1])
for i in items:
    m.addConstr(x_vars[i] <= bigM * z_vars[i], name=f'link_xz1_{i}')
    m.addConstr(x_vars[i] >= z_vars[i], name=f'link_xz2_{i}')
for g in categories:
    items_in_g = [i for i in items if item_to_category[i] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_min_qty[g], name=f'catmin_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_max_qty[g], name=f'catmax_{g}')
    m.addConstr(gp.quicksum((z_vars[i] for i in items_in_g)) <= bigM * w_vars[g], name=f'catw1_{g}')
    m.addConstr(gp.quicksum((z_vars[i] for i in items_in_g)) >= w_vars[g], name=f'catw2_{g}')
for r in resources:
    usage_terms = []
    for i in items:
        (amt, unit) = item_resource_usage[i, r]
        amt_canon = convert_unit(amt, unit, resource_unit[r])
        usage_terms.append(x_vars[i] * amt_canon)
    m.addConstr(gp.quicksum(usage_terms) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompatible_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] <= z_vars[j], name=f'req_{i}_{j}')
for (i, j) in bundles:
    m.addConstr(u_vars[i, j] <= z_vars[i], name=f'bundle1_{i}_{j}')
    m.addConstr(u_vars[i, j] <= z_vars[j], name=f'bundle2_{i}_{j}')
    m.addConstr(u_vars[i, j] >= z_vars[i] + z_vars[j] - 1, name=f'bundle3_{i}_{j}')
obj = gp.quicksum((item_benefit[i] * x_vars[i] for i in items))
obj -= gp.quicksum((item_activation_fee[i] * z_vars[i] for i in items))
obj -= gp.quicksum((cat_activation_fee[g] * w_vars[g] for g in categories))
obj += gp.quicksum((bundle_bonus[i, j] * u_vars[i, j] for (i, j) in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()