import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', 'benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', 'benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', 'benchmark_dataset/Variant35/inputs/batch_06/export_12.csv']
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
requires_df = dfs[10]
usage_df = dfs[11]
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_keys = list(item_df['item_ref'])
item_to_category = item_df.set_index('item_ref')['category'].to_dict()
item_to_platform = item_df.set_index('item_ref')['location_id'].to_dict()
item_to_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_to_minlot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_to_maxorder = item_df.set_index('item_ref')['maximum_order'].to_dict()
platforms = sorted(set(item_df['location_id']) | set(capacity_ledger_df['resource']) | set(usage_df['resource']))
category_keys = list(category_df['category'])
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_to_min = category_df.set_index('category')['minimum_quantity'].to_dict()
category_to_max = category_df.set_index('category')['maximum_quantity'].to_dict()
category_to_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_keys = list(bundle_df.index)
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompatible_df.iterrows()]
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
fx_df['usd_cents_numerator'] = fx_df['usd_cents_numerator'].astype(int)
fx_df['denominator'] = fx_df['denominator'].astype(int)
fx_map = fx_df.set_index(fx_df['currency'].str.strip().str.casefold())[['usd_cents_numerator', 'denominator']].to_dict('index')
benefit_df['amount'] = benefit_df['amount'].astype(int)
benefit_df['currency_norm'] = benefit_df['currency'].str.strip().str.casefold()

def convert_to_usd_cents(row):
    fx = fx_map[row['currency_norm']]
    return row['amount'] * fx['usd_cents_numerator'] // fx['denominator']
benefit_df['usd_cents'] = benefit_df.apply(convert_to_usd_cents, axis=1)
item_benefit = benefit_df.groupby('item_ref')['usd_cents'].sum().to_dict()
for i in item_keys:
    if i not in item_benefit:
        item_benefit[i] = 0
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee_map = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in item_keys:
    if i not in item_fee_map:
        item_fee_map[i] = 0
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['unit_norm'] = usage_df['unit'].str.strip().str.casefold()

def usage_gb_to_mb(row):
    if row['unit_norm'] == 'gb':
        return row['amount'] * 1000
    elif row['unit_norm'] == 'mb':
        return row['amount']
    else:
        raise ValueError(f"Unknown unit {row['unit']} for usage")
usage_df['amount_mb'] = usage_df.apply(usage_gb_to_mb, axis=1)
usage_map = {(row['item_ref'], row['resource']): row['amount_mb'] for (_, row) in usage_df.iterrows()}
for i in item_keys:
    for p in platforms:
        if (i, p) not in usage_map:
            usage_map[i, p] = 0
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
platform_capacity = {}
for p in platforms:
    rows = capacity_ledger_df[capacity_ledger_df['resource'] == p]
    platform_capacity[p] = rows['amount'].sum() if not rows.empty else 0
category_to_items = {c: set(item_df[item_df['category'] == c]['item_ref']) for c in category_keys}
platform_to_items = {p: set(item_df[item_df['location_id'] == p]['item_ref']) for p in platforms}
m = gp.Model('GameEditionAllocation')
x_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_keys, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(len(bundle_pairs), vtype=gp.GRB.BINARY, name='')
for i in item_keys:
    auth = item_to_authorized[i]
    minlot = item_to_minlot[i]
    maxorder = item_to_maxorder[i]
    if auth == 1:
        m.addConstr(x_vars[i] >= minlot * y_vars[i])
        m.addConstr(x_vars[i] <= maxorder * y_vars[i])
        m.addConstr(x_vars[i] <= maxorder)
        m.addConstr(y_vars[i] <= 1)
        m.addConstr(x_vars[i] >= 0)
        m.addConstr(x_vars[i] >= y_vars[i])
        m.addConstr(x_vars[i] <= maxorder * y_vars[i])
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for p in platforms:
    m.addConstr(gp.quicksum((usage_map[i, p] * x_vars[i] for i in item_keys)) <= platform_capacity[p])
for c in category_keys:
    items_in_c = category_to_items[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_to_min[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_to_max[c])
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for (a, b) in incompat_pairs:
    if a in item_keys and b in item_keys:
        m.addConstr(y_vars[a] + y_vars[b] <= 1)
for (i, pre) in requires_pairs:
    if i in item_keys and pre in item_keys:
        m.addConstr(y_vars[i] <= y_vars[pre])
for (idx, (a, b)) in enumerate(bundle_pairs):
    if a in item_keys and b in item_keys:
        m.addConstr(b_vars[idx] <= y_vars[a])
        m.addConstr(b_vars[idx] <= y_vars[b])
        m.addConstr(b_vars[idx] >= y_vars[a] + y_vars[b] - 1)
    else:
        m.addConstr(b_vars[idx] == 0)
benefit_term = gp.quicksum((item_benefit[i] * x_vars[i] for i in item_keys))
item_fee_term = gp.quicksum((item_fee_map[i] * y_vars[i] for i in item_keys))
cat_fee_term = gp.quicksum((category_to_fee[c] * z_vars[c] for c in category_keys))
bundle_term = gp.quicksum((bundle_bonus.get(bundle_pairs[idx], 0) * b_vars[idx] for idx in range(len(bundle_pairs))))
m.setObjective(benefit_term - item_fee_term - cat_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()