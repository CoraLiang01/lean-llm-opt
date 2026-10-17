import gurobipy as gp
import pandas as pd
import numpy as np
import re

def asof_latest(df, key_cols, asof_date):
    """
    For a DataFrame with revisioned records, keep only the latest revision per key (excluding future events),
    and drop records whose latest revision is DELETE.
    """
    df = df.copy()
    df = df[df['effective_date'] <= asof_date]
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(key_cols + ['revision'], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(key_cols, keep='first')
    df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def canonical_unit(resource, unit):
    """
    Return canonical unit for resource, and conversion factor from given unit to canonical.
    Canonical units: space=ml, power=wh, labor=minute
    """
    resource = resource.strip().casefold()
    unit = unit.strip().casefold()
    if resource == 'space':
        if unit == 'ml':
            return ('ml', 1)
        elif unit == 'liter':
            return ('ml', 1000)
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return ('wh', 1)
        elif unit == 'kwh':
            return ('wh', 1000)
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return ('minute', 1)
        elif unit == 'hour':
            return ('minute', 60)
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')

def normalize_id(x):
    return str(x).strip()
asof_date = '2026-05-07'
dealership_id = 'OSLO_NEW_CARS'
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]

def collect_table(table_name):
    frames = []
    for df in dfs:
        if 'table' in df.columns:
            frames.append(df[df['table'].str.strip().str.casefold() == table_name.strip().casefold()])
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()
benefit_raw = collect_table('benefit')
benefit_oslo = benefit_raw[benefit_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
benefit = asof_latest(benefit_oslo, ['dealership_id', 'record_id'], asof_date)
bundle_raw = collect_table('bundle')
bundle_oslo = bundle_raw[bundle_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
bundle = asof_latest(bundle_oslo, ['dealership_id', 'record_id'], asof_date)
capacity_ledger_raw = collect_table('capacity_ledger')
capacity_ledger_oslo = capacity_ledger_raw[capacity_ledger_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
capacity_ledger = asof_latest(capacity_ledger_oslo, ['dealership_id', 'record_id'], asof_date)
category_raw = collect_table('category')
category_oslo = category_raw[category_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
category = asof_latest(category_oslo, ['dealership_id', 'record_id'], asof_date)
fx_raw = collect_table('fx')
fx_oslo = fx_raw[fx_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
fx = asof_latest(fx_oslo, ['dealership_id', 'record_id'], asof_date)
identity_raw = collect_table('identity')
identity_oslo = identity_raw[identity_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
identity = asof_latest(identity_oslo, ['dealership_id', 'record_id'], asof_date)
incompatible_raw = collect_table('incompatible')
incompatible_oslo = incompatible_raw[incompatible_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
incompatible = asof_latest(incompatible_oslo, ['dealership_id', 'record_id'], asof_date)
item_raw = collect_table('item')
item_oslo = item_raw[item_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item = asof_latest(item_oslo, ['dealership_id', 'record_id'], asof_date)
item_fee_raw = collect_table('item_fee')
item_fee_oslo = item_fee_raw[item_fee_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item_fee = asof_latest(item_fee_oslo, ['dealership_id', 'record_id'], asof_date)
requires_raw = collect_table('requires')
requires_oslo = requires_raw[requires_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
requires = asof_latest(requires_oslo, ['dealership_id', 'record_id'], asof_date)
usage_raw = collect_table('usage')
usage_oslo = usage_raw[usage_raw['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
usage = asof_latest(usage_oslo, ['dealership_id', 'record_id'], asof_date)
item['authorized'] = pd.to_numeric(item['authorized'], errors='raise')
item['minimum_lot'] = pd.to_numeric(item['minimum_lot'], errors='raise')
item['maximum_order'] = pd.to_numeric(item['maximum_order'], errors='raise')
item['category'] = item['category'].apply(normalize_id)
item['item_ref'] = item['item_ref'].apply(normalize_id)
items = list(item['item_ref'].unique())
category['category'] = category['category'].apply(normalize_id)
category['minimum_quantity'] = pd.to_numeric(category['minimum_quantity'], errors='raise')
category['maximum_quantity'] = pd.to_numeric(category['maximum_quantity'], errors='raise')
category['activation_fee_cents'] = pd.to_numeric(category['activation_fee_cents'], errors='raise')
categories = list(category['category'].unique())
usage['resource'] = usage['resource'].apply(lambda x: x.strip().casefold())
capacity_ledger['resource'] = capacity_ledger['resource'].apply(lambda x: x.strip().casefold())
resources = sorted(set(usage['resource'].unique()).union(capacity_ledger['resource'].unique()))
bundle = bundle.dropna(subset=['item_a', 'item_b'])
bundle['item_a'] = bundle['item_a'].apply(normalize_id)
bundle['item_b'] = bundle['item_b'].apply(normalize_id)
bundle['bonus_cents'] = pd.to_numeric(bundle['bonus_cents'], errors='raise')
bundles = [(row['item_a'], row['item_b']) for _, row in bundle.iterrows()]
incompatible = incompatible.dropna(subset=['item_a', 'item_b'])
incompatible['item_a'] = incompatible['item_a'].apply(normalize_id)
incompatible['item_b'] = incompatible['item_b'].apply(normalize_id)
incompatible_pairs = [(row['item_a'], row['item_b']) for _, row in incompatible.iterrows()]
requires = requires.dropna(subset=['item_ref', 'prerequisite_ref'])
requires['item_ref'] = requires['item_ref'].apply(normalize_id)
requires['prerequisite_ref'] = requires['prerequisite_ref'].apply(normalize_id)
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for _, row in requires.iterrows()]
fx['currency'] = fx['currency'].apply(lambda x: x.strip().upper())
fx['usd_cents_numerator'] = pd.to_numeric(fx['usd_cents_numerator'], errors='raise')
fx['denominator'] = pd.to_numeric(fx['denominator'], errors='raise')
fx_map = {}
for _, row in fx.iterrows():
    fx_map[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
benefit['item_ref'] = benefit['item_ref'].apply(normalize_id)
benefit['amount'] = pd.to_numeric(benefit['amount'], errors='raise')
benefit['currency'] = benefit['currency'].apply(lambda x: x.strip().upper())
item_benefit_usd = {}
for i in items:
    rows = benefit[benefit['item_ref'] == i]
    total = 0.0
    for _, row in rows.iterrows():
        currency = row['currency']
        amt = row['amount']
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        num, denom = fx_map[currency]
        total += amt * num / denom
    item_benefit_usd[i] = total
item_fee['item_ref'] = item_fee['item_ref'].apply(normalize_id)
item_fee['activation_fee_cents'] = pd.to_numeric(item_fee['activation_fee_cents'], errors='raise')
item_fee_map = {}
for _, row in item_fee.iterrows():
    item_fee_map[row['item_ref']] = row['activation_fee_cents']
category_min = dict(zip(category['category'], category['minimum_quantity']))
category_max = dict(zip(category['category'], category['maximum_quantity']))
category_fee = dict(zip(category['category'], category['activation_fee_cents']))
item_auth = dict(zip(item['item_ref'], item['authorized']))
item_minlot = dict(zip(item['item_ref'], item['minimum_lot']))
item_maxorder = dict(zip(item['item_ref'], item['maximum_order']))
item_category = dict(zip(item['item_ref'], item['category']))
usage['item_ref'] = usage['item_ref'].apply(normalize_id)
usage['amount'] = pd.to_numeric(usage['amount'], errors='raise')
usage['unit'] = usage['unit'].apply(lambda x: x.strip().casefold())
usage_per_unit = {}
for _, row in usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    unit = row['unit']
    canon_unit, factor = canonical_unit(r, unit)
    usage_per_unit[i, r] = amt * factor
capacity_ledger['resource'] = capacity_ledger['resource'].apply(lambda x: x.strip().casefold())
capacity_ledger['amount'] = pd.to_numeric(capacity_ledger['amount'], errors='raise')
capacity_ledger['unit'] = capacity_ledger['unit'].apply(lambda x: x.strip().casefold())
resource_capacity = {}
for r in resources:
    total = 0.0
    for _, row in capacity_ledger[capacity_ledger['resource'] == r].iterrows():
        amt = row['amount']
        unit = row['unit']
        canon_unit, factor = canonical_unit(r, unit)
        total += amt * factor
    resource_capacity[r] = total
bundle_bonus = {}
for _, row in bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_bonus[a, b] = row['bonus_cents']
category_items = {g: [] for g in categories}
for i in items:
    g = item_category[i]
    if g in category_items:
        category_items[g].append(i)
m = gp.Model('OsloVehicleOrderNetBenefit')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    auth = item_auth[i]
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    if auth == 0:
        m.addConstr(q_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q_vars[i] >= minlot * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= maxorder * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(q_vars[i] >= z_vars[i], name=f'z_lb_{i}')
        m.addConstr(q_vars[i] <= maxorder * z_vars[i], name=f'z_ub_{i}')
for g in categories:
    items_in_g = category_items[g]
    if not items_in_g:
        continue
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= category_min[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= category_max[g], name=f'cat_max_{g}')
    for i in items_in_g:
        m.addConstr(z_vars[i] <= y_vars[g], name=f'cat_y_ub_{g}_{i}')
    m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in items_in_g)), name=f'cat_y_lb_{g}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_unit.get((i, r), 0.0) * q_vars[i] for i in items)) <= resource_capacity[r], name=f'rescap_{r}')
for a, b in bundles:
    m.addConstr(w_vars[a, b] <= z_vars[a], name=f'bundle_w_ub1_{a}_{b}')
    m.addConstr(w_vars[a, b] <= z_vars[b], name=f'bundle_w_ub2_{a}_{b}')
    m.addConstr(w_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle_w_lb_{a}_{b}')
for i, j in incompatible_pairs:
    if i in z_vars and j in z_vars:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for i, k in requires_pairs:
    if i in z_vars and k in z_vars:
        m.addConstr(z_vars[i] <= z_vars[k], name=f'requires_{i}_{k}')
benefit_expr = gp.quicksum((item_benefit_usd.get(i, 0.0) * q_vars[i] for i in items))
item_fee_expr = gp.quicksum((item_fee_map.get(i, 0.0) * z_vars[i] for i in items))
cat_fee_expr = gp.quicksum((category_fee.get(g, 0.0) * y_vars[g] for g in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus.get((a, b), 0.0) * w_vars[a, b] for a, b in bundles))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.optimize()