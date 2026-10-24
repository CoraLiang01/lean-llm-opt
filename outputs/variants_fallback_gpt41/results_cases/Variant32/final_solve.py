import gurobipy as gp
import pandas as pd
import numpy as np
import re

def asof_latest(df, key_cols, asof_date, revision_col='revision', date_col='effective_date', action_col='action'):
    df = df[df[date_col] <= asof_date].copy()
    df[revision_col] = df[revision_col].astype(int)
    df = df.sort_values(key_cols + [date_col, revision_col])
    idx = df.groupby(key_cols)[revision_col].transform('max') == df[revision_col]
    df = df[idx]
    if action_col in df.columns:
        df = df[df[action_col].str.strip().str.casefold() != 'delete']
    return df

def canonical_unit(resource, unit):
    resource = resource.strip().casefold()
    unit = unit.strip().casefold()
    if resource == 'space':
        if unit == 'ml':
            return 1
        elif unit == 'liter':
            return 1000
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return 1
        elif unit == 'kwh':
            return 1000
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return 1
        elif unit == 'hour':
            return 60
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
csv_paths = ['benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', 'benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', 'benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', 'benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', 'benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', 'benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', 'benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', 'benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', 'benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', 'benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', 'benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', 'benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', 'benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', 'benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', 'benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', 'benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', 'benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = []
for path in csv_paths:
    dfs.append(pd.read_csv(path, dtype=str, keep_default_na=False))
asof_date = '2026-05-07'

def collect_table(table_name):
    frames = []
    for df in dfs:
        if 'table' in df.columns and 'dealership_id' in df.columns:
            mask = (df['table'].str.strip().str.casefold() == table_name.casefold()) & (df['dealership_id'].str.strip().str.casefold() == 'oslo_new_cars')
            if mask.any():
                frames.append(df[mask].copy())
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()
item_df = collect_table('item')
if not item_df.empty:
    item_df = asof_latest(item_df, ['record_id'], asof_date)
    item_df = item_df[item_df['item_ref'].str.strip() != '']
    item_df = item_df.set_index('item_ref', drop=False)
else:
    raise RuntimeError('No OSLO_NEW_CARS items found.')
items = list(item_df.index)
cat_df = collect_table('category')
if not cat_df.empty:
    cat_df = asof_latest(cat_df, ['record_id'], asof_date)
    cat_df = cat_df[cat_df['category'].str.strip() != '']
    cat_df = cat_df.set_index('category', drop=False)
else:
    cat_df = pd.DataFrame(columns=['category'])
categories = list(cat_df.index)
usage_df = collect_table('usage')
if not usage_df.empty:
    usage_df = asof_latest(usage_df, ['record_id'], asof_date)
    usage_df = usage_df[usage_df['item_ref'].str.strip() != '']
    usage_df = usage_df[usage_df['resource'].str.strip() != '']
    usage_df = usage_df[usage_df['amount'].str.strip() != '']
    usage_df = usage_df[usage_df['unit'].str.strip() != '']
else:
    usage_df = pd.DataFrame(columns=['item_ref', 'resource', 'amount', 'unit'])
resources = sorted(set(usage_df['resource'].str.strip().unique()) | set(collect_table('capacity_ledger')['resource'].str.strip().unique()))
resources = [r for r in resources if r != '']
bundle_df = collect_table('bundle')
if not bundle_df.empty:
    bundle_df = asof_latest(bundle_df, ['record_id'], asof_date)
    bundle_df = bundle_df[(bundle_df['item_a'].str.strip() != '') & (bundle_df['item_b'].str.strip() != '')]
    bundle_df = bundle_df.reset_index(drop=True)
    bundles = []
    bundle_idx = []
    for (idx, row) in bundle_df.iterrows():
        a = row['item_a'].strip()
        b = row['item_b'].strip()
        if a != '' and b != '':
            bundles.append((a, b))
            bundle_idx.append(idx)
else:
    bundles = []
    bundle_idx = []
incomp_df = collect_table('incompatible')
if not incomp_df.empty:
    incomp_df = asof_latest(incomp_df, ['record_id'], asof_date)
    incomp_df = incomp_df[(incomp_df['item_a'].str.strip() != '') & (incomp_df['item_b'].str.strip() != '')]
    incomp_pairs = set()
    for (idx, row) in incomp_df.iterrows():
        a = row['item_a'].strip()
        b = row['item_b'].strip()
        if a != '' and b != '':
            incomp_pairs.add(tuple(sorted((a, b))))
else:
    incomp_pairs = set()
req_df = collect_table('requires')
if not req_df.empty:
    req_df = asof_latest(req_df, ['record_id'], asof_date)
    req_df = req_df[(req_df['item_ref'].str.strip() != '') & (req_df['prerequisite_ref'].str.strip() != '')]
    requires_pairs = []
    for (idx, row) in req_df.iterrows():
        i = row['item_ref'].strip()
        j = row['prerequisite_ref'].strip()
        if i != '' and j != '':
            requires_pairs.append((i, j))
else:
    requires_pairs = []
fx_df = collect_table('fx')
if not fx_df.empty:
    fx_df = asof_latest(fx_df, ['record_id'], asof_date)
    fx_df = fx_df[fx_df['currency'].str.strip() != '']
    fx_df = fx_df.sort_values(['currency', 'effective_date', 'revision'])
    fx_map = {}
    for currency in fx_df['currency'].unique():
        sub = fx_df[fx_df['currency'] == currency]
        row = sub.iloc[-1]
        num = float(row['usd_cents_numerator'])
        denom = float(row['denominator'])
        fx_map[currency] = (num, denom)
else:
    fx_map = {}
benefit_df = collect_table('benefit')
if not benefit_df.empty:
    benefit_df = asof_latest(benefit_df, ['record_id'], asof_date)
    benefit_df = benefit_df[benefit_df['item_ref'].str.strip() != '']
    benefit_per_unit = {}
    for i in items:
        sub = benefit_df[benefit_df['item_ref'] == i]
        total = 0.0
        for (idx, row) in sub.iterrows():
            amt = float(row['amount'])
            currency = row['currency'].strip()
            if currency not in fx_map:
                raise RuntimeError(f'Missing FX rate for currency {currency} as of {asof_date}')
            (num, denom) = fx_map[currency]
            usd_cents = amt * num / denom
            total += usd_cents
        benefit_per_unit[i] = total
else:
    benefit_per_unit = {i: 0.0 for i in items}
item_fee_df = collect_table('item_fee')
if not item_fee_df.empty:
    item_fee_df = asof_latest(item_fee_df, ['record_id'], asof_date)
    item_fee_df = item_fee_df[item_fee_df['item_ref'].str.strip() != '']
    item_fee_map = {}
    for (idx, row) in item_fee_df.iterrows():
        i = row['item_ref'].strip()
        if i != '':
            item_fee_map[i] = float(row['activation_fee_cents'])
else:
    item_fee_map = {}
cat_fee_map = {}
for c in categories:
    row = cat_df.loc[c]
    cat_fee_map[c] = float(row['activation_fee_cents'])
bundle_bonus_map = {}
for (idx, (a, b)) in zip(bundle_idx, bundles):
    row = bundle_df.iloc[idx]
    bundle_bonus_map[a, b] = float(row['bonus_cents'])
min_lot = {}
max_order = {}
authorized = {}
item_category = {}
for i in items:
    row = item_df.loc[i]
    min_lot[i] = int(float(row['minimum_lot']))
    max_order[i] = int(float(row['maximum_order']))
    authorized[i] = int(float(row['authorized']))
    item_category[i] = row['category'].strip()
cat_min_qty = {}
cat_max_qty = {}
for c in categories:
    row = cat_df.loc[c]
    cat_min_qty[c] = int(float(row['minimum_quantity']))
    cat_max_qty[c] = int(float(row['maximum_quantity']))
usage_per_unit = {}
for (idx, row) in usage_df.iterrows():
    i = row['item_ref'].strip()
    r = row['resource'].strip()
    if i == '' or r == '':
        continue
    amt = float(row['amount'])
    unit = row['unit'].strip()
    factor = canonical_unit(r, unit)
    usage_per_unit[i, r] = amt * factor
cap_df = collect_table('capacity_ledger')
if not cap_df.empty:
    cap_df = asof_latest(cap_df, ['record_id'], asof_date)
    cap_df = cap_df[cap_df['resource'].str.strip() != '']
    resource_capacity = {}
    for r in resources:
        sub = cap_df[cap_df['resource'].str.strip() == r]
        total = 0.0
        for (idx, row) in sub.iterrows():
            amt_val = row['amount']
            if isinstance(amt_val, pd.Series):
                amt_val = amt_val.iloc[0]
            amt = float(amt_val) if str(amt_val).strip() != '' else 0.0
            unit = row['unit'].strip()
            factor = canonical_unit(r, unit)
            total += amt * factor
        resource_capacity[r] = total
else:
    resource_capacity = {}
cat_items = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    if c in cat_items:
        cat_items[c].append(i)
for i in items:
    if i not in benefit_per_unit:
        raise RuntimeError(f'Missing benefit for item {i}')
    if i not in min_lot or i not in max_order or i not in authorized:
        raise RuntimeError(f'Missing lot/order/auth for item {i}')
    if i not in item_category:
        raise RuntimeError(f'Missing category for item {i}')
for c in categories:
    if c not in cat_min_qty or c not in cat_max_qty or c not in cat_fee_map:
        raise RuntimeError(f'Missing category bounds/fee for {c}')
for r in resources:
    if r not in resource_capacity:
        resource_capacity[r] = 0.0
m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(range(len(bundles)), vtype=gp.GRB.BINARY, name='')
for i in items:
    if authorized[i] > 0:
        m.addConstr(q_vars[i] >= min_lot[i] * z_vars[i])
        m.addConstr(q_vars[i] <= max_order[i] * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
    m.addConstr(q_vars[i] >= z_vars[i])
for c in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_items[c])) >= cat_min_qty[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_items[c])) <= cat_max_qty[c])
for c in categories:
    for i in cat_items[c]:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in cat_items[c])))
for r in resources:
    m.addConstr(gp.quicksum((usage_per_unit.get((i, r), 0.0) * q_vars[i] for i in items)) <= resource_capacity[r])
for (i, j) in incomp_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(z_vars[i] <= z_vars[j])
for (b_idx, (i, j)) in enumerate(bundles):
    if i in items and j in items:
        m.addConstr(w_vars[b_idx] <= z_vars[i])
        m.addConstr(w_vars[b_idx] <= z_vars[j])
        m.addConstr(w_vars[b_idx] >= z_vars[i] + z_vars[j] - 1)
    else:
        m.addConstr(w_vars[b_idx] == 0)
obj_benefit = gp.quicksum((benefit_per_unit[i] * q_vars[i] for i in items))
obj_item_fee = gp.quicksum((item_fee_map.get(i, 0.0) * z_vars[i] for i in items))
obj_cat_fee = gp.quicksum((cat_fee_map[c] * y_vars[c] for c in categories))
obj_bundle_bonus = gp.quicksum((bundle_bonus_map.get((a, b), 0.0) * w_vars[b_idx] for (b_idx, (a, b)) in enumerate(bundles)))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle_bonus, gp.GRB.MAXIMIZE)
m.optimize()