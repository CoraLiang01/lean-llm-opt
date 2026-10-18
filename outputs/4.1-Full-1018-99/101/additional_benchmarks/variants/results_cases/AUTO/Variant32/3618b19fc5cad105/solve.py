import gurobipy as gp
import pandas as pd
import numpy as np
import re

def asof_filter(df, key_cols, asof_date, revision_col='revision', date_col='effective_date', action_col='action'):
    """
    For a DataFrame with revisioned records, filter to the latest revision <= asof_date for each key,
    and drop if that revision is DELETE.
    """
    df = df[df[date_col] <= asof_date].copy()
    df[revision_col] = pd.to_numeric(df[revision_col], errors='raise')
    df = df.sort_values(key_cols + [revision_col], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(subset=key_cols, keep='first')
    if action_col in df.columns:
        df = df[df[action_col].str.strip().str.casefold() != 'delete']
    return df

def convert_unit(val, from_unit, to_unit):
    """
    Convert val from from_unit to to_unit.
    Supported: ml/liter, minute/hour, wh/kwh.
    """
    from_u = from_unit.strip().lower()
    to_u = to_unit.strip().lower()
    if from_u == to_u:
        return val
    if (from_u, to_u) == ('liter', 'ml'):
        return val * 1000
    if (from_u, to_u) == ('ml', 'liter'):
        return val / 1000
    if (from_u, to_u) == ('hour', 'minute'):
        return val * 60
    if (from_u, to_u) == ('minute', 'hour'):
        return val / 60
    if (from_u, to_u) == ('kwh', 'wh'):
        return val * 1000
    if (from_u, to_u) == ('wh', 'kwh'):
        return val / 1000
    raise ValueError(f'Unsupported unit conversion: {from_unit} -> {to_unit}')

def safe_float(x):
    if pd.isnull(x) or x == '':
        return 0.0
    return float(x)

def safe_int(x):
    if pd.isnull(x) or x == '':
        return 0
    return int(float(x))
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = []
for path in csv_paths:
    dfs.append(pd.read_csv(path, dtype=str, keep_default_na=False))
asof_date = '2026-05-07'
dealership_id = 'OSLO_NEW_CARS'

def collect_table(table_name):
    frames = []
    for df in dfs:
        if 'table' in df.columns:
            frames.append(df[df['table'].str.strip().str.casefold() == table_name.strip().casefold()])
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame()
item_df = collect_table('item')
item_df = item_df[item_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item_df = asof_filter(item_df, ['dealership_id', 'record_id'], asof_date)
item_df = item_df[item_df['item_ref'].str.strip() != '']
item_keys = list(item_df['item_ref'])
category_df = collect_table('category')
category_df = category_df[category_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
category_df = asof_filter(category_df, ['dealership_id', 'record_id'], asof_date)
category_df = category_df[category_df['category'].str.strip() != '']
category_keys = list(category_df['category'])
usage_df = collect_table('usage')
usage_df = usage_df[usage_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
usage_df = asof_filter(usage_df, ['dealership_id', 'record_id'], asof_date)
usage_df = usage_df[usage_df['item_ref'].str.strip() != '']
resource_keys = sorted(usage_df['resource'].dropna().unique())
resource_keys = [r for r in resource_keys if r.strip() != '']
bundle_df = collect_table('bundle')
bundle_df = bundle_df[bundle_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
bundle_df = asof_filter(bundle_df, ['dealership_id', 'record_id'], asof_date)
bundle_df = bundle_df[(bundle_df['item_a'].str.strip() != '') & (bundle_df['item_b'].str.strip() != '')]
bundle_keys = list(bundle_df.index)
incomp_df = collect_table('incompatible')
incomp_df = incomp_df[incomp_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
incomp_df = asof_filter(incomp_df, ['dealership_id', 'record_id'], asof_date)
incomp_df = incomp_df[(incomp_df['item_a'].str.strip() != '') & (incomp_df['item_b'].str.strip() != '')]
incomp_pairs = [(row['item_a'], row['item_b']) for _, row in incomp_df.iterrows()]
requires_df = collect_table('requires')
requires_df = requires_df[requires_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
requires_df = asof_filter(requires_df, ['dealership_id', 'record_id'], asof_date)
requires_df = requires_df[(requires_df['item_ref'].str.strip() != '') & (requires_df['prerequisite_ref'].str.strip() != '')]
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for _, row in requires_df.iterrows()]
benefit_df = collect_table('benefit')
benefit_df = benefit_df[benefit_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
benefit_df = asof_filter(benefit_df, ['dealership_id', 'record_id'], asof_date)
benefit_df = benefit_df[benefit_df['item_ref'].str.strip() != '']
fx_df = collect_table('fx')
fx_df = fx_df[fx_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
fx_df = asof_filter(fx_df, ['dealership_id', 'record_id'], asof_date)
fx_df = fx_df[fx_df['currency'].str.strip() != '']
item_fee_df = collect_table('item_fee')
item_fee_df = item_fee_df[item_fee_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
item_fee_df = asof_filter(item_fee_df, ['dealership_id', 'record_id'], asof_date)
item_fee_df = item_fee_df[item_fee_df['item_ref'].str.strip() != '']
cap_ledger_df = collect_table('capacity_ledger')
cap_ledger_df = cap_ledger_df[cap_ledger_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
cap_ledger_df = asof_filter(cap_ledger_df, ['dealership_id', 'record_id'], asof_date)
cap_ledger_df = cap_ledger_df[cap_ledger_df['resource'].str.strip() != '']
identity_df = collect_table('identity')
identity_df = identity_df[identity_df['dealership_id'].str.strip().str.casefold() == dealership_id.casefold()]
identity_df = asof_filter(identity_df, ['dealership_id', 'record_id'], asof_date)
identity_df = identity_df[identity_df['ref'].str.strip() != '']
item_params = {}
for _, row in item_df.iterrows():
    i = row['item_ref']
    item_params[i] = {'category': row['category'], 'authorized': safe_int(row['authorized']), 'minimum_lot': safe_int(row['minimum_lot']), 'maximum_order': safe_int(row['maximum_order'])}
category_params = {}
for _, row in category_df.iterrows():
    c = row['category']
    category_params[c] = {'minimum_quantity': safe_int(row['minimum_quantity']), 'maximum_quantity': safe_int(row['maximum_quantity']), 'activation_fee_cents': safe_int(row['activation_fee_cents'])}
usage_per_item_resource = {}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    if i not in item_keys or r.strip() == '':
        continue
    amt = safe_float(row['amount'])
    from_unit = row['unit']
    usage_per_item_resource.setdefault(i, {})[r] = (amt, from_unit)
canonical_unit = {'space': 'ml', 'power': 'wh', 'labor': 'minute'}
resource_capacity = {}
for r in resource_keys:
    total = 0.0
    for _, row in cap_ledger_df[cap_ledger_df['resource'] == r].iterrows():
        amt = safe_float(row['amount'])
        from_unit = row['unit']
        to_unit = canonical_unit.get(r, from_unit)
        total += convert_unit(amt, from_unit, to_unit)
    resource_capacity[r] = total
usage_canonical = {}
for i in item_keys:
    usage_canonical[i] = {}
    for r in resource_keys:
        if i in usage_per_item_resource and r in usage_per_item_resource[i]:
            amt, from_unit = usage_per_item_resource[i][r]
            to_unit = canonical_unit.get(r, from_unit)
            usage_canonical[i][r] = convert_unit(amt, from_unit, to_unit)
        else:
            usage_canonical[i][r] = 0.0
fx_rates = {}
for _, row in fx_df.iterrows():
    currency = row['currency']
    usd_cents_numerator = safe_float(row['usd_cents_numerator'])
    denominator = safe_float(row['denominator'])
    if denominator == 0:
        continue
    fx_rates[currency] = (usd_cents_numerator, denominator)
benefit_per_unit = dict.fromkeys(item_keys, 0.0)
for i in item_keys:
    rows = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for _, row in rows.iterrows():
        amt = safe_float(row['amount'])
        currency = row['currency']
        if currency not in fx_rates:
            raise ValueError(f'Missing FX rate for currency {currency} as of {asof_date}')
        num, denom = fx_rates[currency]
        usd_cents = amt * num / denom
        total += usd_cents
    benefit_per_unit[i] = total
item_fee = {}
for _, row in item_fee_df.iterrows():
    i = row['item_ref']
    if i in item_keys:
        item_fee[i] = safe_float(row['activation_fee_cents'])
for i in item_keys:
    if i not in item_fee:
        item_fee[i] = 0.0
category_fee = {}
for c in category_keys:
    category_fee[c] = category_params[c]['activation_fee_cents']
bundle_bonus = {}
bundle_items = {}
for b in bundle_keys:
    row = bundle_df.loc[b]
    i1 = row['item_a']
    i2 = row['item_b']
    bonus = safe_float(row['bonus_cents'])
    bundle_bonus[b] = bonus
    bundle_items[b] = (i1, i2)
item_category = {i: item_params[i]['category'] for i in item_keys}
category_items = {c: [] for c in category_keys}
for i in item_keys:
    c = item_category[i]
    if c in category_items:
        category_items[c].append(i)
authorized = {i: item_params[i]['authorized'] > 0 for i in item_keys}
min_lot = {i: item_params[i]['minimum_lot'] for i in item_keys}
max_order = {i: item_params[i]['maximum_order'] for i in item_keys}
cat_min_qty = {c: category_params[c]['minimum_quantity'] for c in category_keys}
cat_max_qty = {c: category_params[c]['maximum_quantity'] for c in category_keys}
m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
q_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_keys, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_keys:
    if not authorized[i]:
        m.addConstr(q_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q_vars[i] >= min_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= max_order[i] * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(q_vars[i] <= max_order[i], name=f'maxorder2_{i}')
        m.addConstr(q_vars[i] >= 0, name=f'nonneg_{i}')
for i in item_keys:
    m.addConstr(q_vars[i] >= z_vars[i], name=f'link_lb_{i}')
    m.addConstr(q_vars[i] <= max_order[i] * z_vars[i], name=f'link_ub_{i}')
for c in category_keys:
    for i in category_items[c]:
        m.addConstr(z_vars[i] <= y_vars[c], name=f'cat_link_{i}_{c}')
for c in category_keys:
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) >= cat_min_qty[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) <= cat_max_qty[c], name=f'cat_max_{c}')
for r in resource_keys:
    m.addConstr(gp.quicksum((usage_canonical[i][r] * q_vars[i] for i in item_keys)) <= resource_capacity[r], name=f'rescap_{r}')
for i, j in incomp_pairs:
    if i in item_keys and j in item_keys:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
for i, j in requires_pairs:
    if i in item_keys and j in item_keys:
        m.addConstr(z_vars[i] <= z_vars[j], name=f'requires_{i}_{j}')
for b in bundle_keys:
    i1, i2 = bundle_items[b]
    if i1 in item_keys and i2 in item_keys:
        m.addConstr(w_vars[b] <= z_vars[i1], name=f'bundle1_{b}')
        m.addConstr(w_vars[b] <= z_vars[i2], name=f'bundle2_{b}')
        m.addConstr(w_vars[b] >= z_vars[i1] + z_vars[i2] - 1, name=f'bundle3_{b}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle_na_{b}')
total_benefit = gp.quicksum((benefit_per_unit[i] * q_vars[i] for i in item_keys))
total_item_fee = gp.quicksum((item_fee[i] * z_vars[i] for i in item_keys))
total_category_fee = gp.quicksum((category_fee[c] * y_vars[c] for c in category_keys))
total_bundle_bonus = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_keys))
net_benefit = total_benefit - total_item_fee - total_category_fee + total_bundle_bonus
m.setObjective(net_benefit, gp.GRB.MAXIMIZE)
m.optimize()