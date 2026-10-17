import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
benefit_df = dfs[0]
bundle_df = dfs[1]
capacity_ledger_df = dfs[2]
category_df = dfs[3]
fx_df = dfs[4]
identity_df = dfs[5]
incompatible_df = dfs[6]
item_df = dfs[7]
item_fee_df = dfs[8]
market_df = dfs[9]
requires_df = dfs[10]
usage_df = dfs[11]
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df = item_df.set_index('item_ref', drop=False)
item_refs = list(item_df.index)
platforms = sorted(set(item_df['location_id'].unique()) | set(usage_df['resource'].unique()) | set(capacity_ledger_df['resource'].unique()))
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_df = category_df.set_index('category', drop=False)
categories = list(category_df.index)
bundle_pairs = []
for _, row in bundle_df.iterrows():
    bundle_pairs.append((row['item_a'], row['item_b']))
bundle_bonus = {(row['item_a'], row['item_b']): int(row['bonus_cents']) for _, row in bundle_df.iterrows()}
incompat_pairs = []
for _, row in incompatible_df.iterrows():
    incompat_pairs.append((row['item_a'], row['item_b']))
prereq_pairs = []
for _, row in requires_df.iterrows():
    prereq_pairs.append((row['item_ref'], row['prerequisite_ref']))
fx_df = fx_df.set_index('currency')
fx_df['usd_cents_numerator'] = fx_df['usd_cents_numerator'].astype(int)
fx_df['denominator'] = fx_df['denominator'].astype(int)
fx_map = fx_df[['usd_cents_numerator', 'denominator']].to_dict(orient='index')
benefit_df['amount'] = benefit_df['amount'].astype(int)
benefit_df = benefit_df.set_index(['item_ref', 'component'], drop=False)
benefit_per_unit = {}
for item in item_refs:
    rows = benefit_df.loc[benefit_df['item_ref'] == item]
    total = 0
    for _, row in rows.iterrows():
        currency = row['currency']
        fx = fx_map[currency]
        val = row['amount'] * fx['usd_cents_numerator'] / fx['denominator']
        total += val
    benefit_per_unit[item] = int(round(total))
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee_df = item_fee_df.set_index('item_ref')
item_activation_fee = item_fee_df['activation_fee_cents'].to_dict()
for item in item_refs:
    if item not in item_activation_fee:
        item_activation_fee[item] = 0
category_activation_fee = category_df['activation_fee_cents'].to_dict()
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()
usage_map = {}
for _, row in usage_df.iterrows():
    item = row['item_ref']
    resource = row['resource']
    amt = row['amount']
    unit = row['unit']
    if unit == 'gb':
        amt_mb = amt * 1000
    elif unit == 'mb':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown unit {unit} for item {item} resource {resource}')
    usage_map[item, resource] = amt_mb
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
capacity_ledger_df['unit'] = capacity_ledger_df['unit'].str.strip().str.casefold()
platform_capacity = {}
for resource in platforms:
    rows = capacity_ledger_df.loc[capacity_ledger_df['resource'] == resource]
    total = 0
    for _, row in rows.iterrows():
        amt = row['amount']
        unit = row['unit']
        if unit == 'mb':
            amt_mb = amt
        elif unit == 'gb':
            amt_mb = amt * 1000
        else:
            raise ValueError(f'Unknown unit {unit} for resource {resource}')
        total += amt_mb
    platform_capacity[resource] = total
item_authorized = item_df['authorized'].to_dict()
item_min_lot = item_df['minimum_lot'].to_dict()
item_max_order = item_df['maximum_order'].to_dict()
item_category = item_df['category'].to_dict()
item_platform = item_df['location_id'].to_dict()
category_min_qty = category_df['minimum_quantity'].to_dict()
category_max_qty = category_df['maximum_quantity'].to_dict()
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth == 1:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i])
        m.addConstr(x_vars[i] <= max_order * y_vars[i])
        m.addConstr(x_vars[i] <= max_order)
        m.addConstr(x_vars[i] >= 0)
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    m.addConstr(x_vars[i] >= y_vars[i])
    m.addConstr(x_vars[i] <= max_order * y_vars[i])
for resource in platforms:
    items_on_platform = [i for i in item_refs if item_platform[i] == resource]
    if not items_on_platform:
        continue
    m.addConstr(gp.quicksum((usage_map.get((i, resource), 0) * x_vars[i] for i in items_on_platform)) <= platform_capacity.get(resource, 0))
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_min_qty[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_max_qty[c])
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for i, j in incompat_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for i, prereq in prereq_pairs:
    if i in y_vars and prereq in y_vars:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for pair in bundle_pairs:
    i, j = pair
    if i in y_vars and j in y_vars:
        m.addConstr(b_vars[pair] <= y_vars[i])
        m.addConstr(b_vars[pair] <= y_vars[j])
        m.addConstr(b_vars[pair] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(b_vars[pair] == 0)
benefit_term = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in item_refs))
item_fee_term = gp.quicksum((item_activation_fee[i] * y_vars[i] for i in item_refs))
cat_fee_term = gp.quicksum((category_activation_fee[c] * z_vars[c] for c in categories))
bundle_term = gp.quicksum((bundle_bonus[pair] * b_vars[pair] for pair in bundle_pairs))
m.setObjective(benefit_term - item_fee_term - cat_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()