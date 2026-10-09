import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]

def get_table(dfs, tag):
    return pd.concat([df[df['table'].str.strip().str.casefold() == tag.casefold()] for df in dfs if 'table' in df.columns], ignore_index=True)
benefit_df = get_table(dfs, 'benefit')
bundle_df = get_table(dfs, 'bundle')
capacity_ledger_df = get_table(dfs, 'capacity_ledger')
category_df = get_table(dfs, 'category')
fx_df = get_table(dfs, 'fx')
identity_df = get_table(dfs, 'identity')
incompatible_df = get_table(dfs, 'incompatible')
item_df = get_table(dfs, 'item')
item_fee_df = get_table(dfs, 'item_fee')
market_df = get_table(dfs, 'market')
requires_df = get_table(dfs, 'requires')
usage_df = get_table(dfs, 'usage')
item_df['item_ref'] = item_df['item_ref'].astype(str)
item_refs = item_df['item_ref'].unique().tolist()
category_df['category'] = category_df['category'].astype(str)
categories = category_df['category'].unique().tolist()
platforms = sorted(set(item_df['location_id'].unique()) | set(capacity_ledger_df['resource'].unique()) | set(usage_df['resource'].unique()))
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
fx_map = {}
for (_, row) in fx_df.iterrows():
    fx_map[row['currency'].strip()] = (int(row['usd_cents_numerator']), int(row['denominator']))
benefit_per_unit = {}
for item in item_refs:
    rows = benefit_df[benefit_df['item_ref'] == item]
    total = 0
    for (_, row) in rows.iterrows():
        amount = int(row['amount'])
        currency = row['currency'].strip()
        if currency not in fx_map:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = fx_map[currency]
        benefit_cents = amount * num / denom
        total += benefit_cents
    benefit_per_unit[item] = total
item_fee_map = {}
for (_, row) in item_fee_df.iterrows():
    item_fee_map[row['item_ref']] = int(row['activation_fee_cents'])
bundle_bonus_map = {}
for (_, row) in bundle_df.iterrows():
    bundle_bonus_map[row['item_a'], row['item_b']] = int(row['bonus_cents'])
category_bounds = {}
category_fee_map = {}
for (_, row) in category_df.iterrows():
    c = row['category']
    category_bounds[c] = (int(row['minimum_quantity']), int(row['maximum_quantity']))
    category_fee_map[c] = int(row['activation_fee_cents'])
item_param_map = {}
for (_, row) in item_df.iterrows():
    item_param_map[row['item_ref']] = {'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id']}
usage_map = {}
for (_, row) in usage_df.iterrows():
    item = row['item_ref']
    resource = row['resource']
    amount = int(row['amount'])
    unit = row['unit'].strip().upper()
    if unit == 'GB':
        amount_mb = amount * 1000
    elif unit == 'MB':
        amount_mb = amount
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_map[item, resource] = amount_mb
capacity_map = {}
for resource in platforms:
    rows = capacity_ledger_df[capacity_ledger_df['resource'] == resource]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().upper()
        if unit == 'GB':
            amt_mb = amt * 1000
        elif unit == 'MB':
            amt_mb = amt
        else:
            raise ValueError(f'Unknown unit {unit} for capacity_ledger')
        total += amt_mb
    capacity_map[resource] = total
incompatible_pairs = set()
for (_, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    incompatible_pairs.add((a, b))
    incompatible_pairs.add((b, a))
prerequisite_pairs = []
for (_, row) in requires_df.iterrows():
    prerequisite_pairs.append((row['item_ref'], row['prerequisite_ref']))
category_items = {c: [] for c in categories}
for item in item_refs:
    c = item_param_map[item]['category']
    category_items[c].append(item)
m = gp.Model('GameEditionAllocation')
q_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
g_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for item in item_refs:
    params = item_param_map[item]
    authorized = params['authorized']
    min_lot = params['minimum_lot']
    max_order = params['maximum_order']
    if not authorized:
        m.addConstr(q_vars[item] == 0, name=f'unauth_q_{item}')
        m.addConstr(z_vars[item] == 0, name=f'unauth_z_{item}')
    else:
        m.addConstr(q_vars[item] >= min_lot * z_vars[item], name=f'minlot_{item}')
        m.addConstr(q_vars[item] <= max_order * z_vars[item], name=f'maxorder_{item}')
        m.addConstr(q_vars[item] >= z_vars[item], name=f'zlink_{item}')
for resource in platforms:
    items_on_resource = [item for item in item_refs if item_param_map[item]['location_id'] == resource]
    m.addConstr(gp.quicksum((usage_map.get((item, resource), 0) * q_vars[item] for item in items_on_resource)) <= capacity_map.get(resource, 0), name=f'capacity_{resource}')
for c in categories:
    items_in_c = category_items[c]
    (min_q, max_q) = category_bounds[c]
    m.addConstr(gp.quicksum((q_vars[item] for item in items_in_c)) >= min_q * g_vars[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((q_vars[item] for item in items_in_c)) <= max_q * g_vars[c], name=f'cat_max_{c}')
    for item in items_in_c:
        m.addConstr(z_vars[item] <= g_vars[c], name=f'cat_zleqg_{item}_{c}')
    m.addConstr(g_vars[c] <= gp.quicksum((z_vars[item] for item in items_in_c)), name=f'cat_gleqsumz_{c}')
for (i, j) in incompatible_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in prerequisite_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z_vars[i] <= z_vars[j], name=f'prereq_{i}_{j}')
for (i, j) in bundle_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(b_vars[i, j] <= z_vars[i], name=f'bundle_b_leq_zi_{i}_{j}')
        m.addConstr(b_vars[i, j] <= z_vars[j], name=f'bundle_b_leq_zj_{i}_{j}')
        m.addConstr(b_vars[i, j] >= z_vars[i] + z_vars[j] - 1, name=f'bundle_b_geq_sum_{i}_{j}')
    else:
        m.addConstr(b_vars[i, j] == 0, name=f'bundle_b_zero_{i}_{j}')
benefit_expr = gp.quicksum((benefit_per_unit[item] * q_vars[item] for item in item_refs))
item_fee_expr = gp.quicksum((item_fee_map.get(item, 0) * z_vars[item] for item in item_refs))
cat_fee_expr = gp.quicksum((category_fee_map[c] * g_vars[c] for c in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus_map.get((i, j), 0) * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(benefit_expr - item_fee_expr - cat_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.optimize()