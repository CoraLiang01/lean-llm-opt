import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm(x):
    return str(x).strip().casefold()
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
df_benefit = dfs[0]
df_bundle = dfs[1]
df_capacity_ledger = dfs[2]
df_category = dfs[3]
df_identity = dfs[4]
df_incompatible = dfs[5]
df_item_1 = dfs[6]
df_item_2 = dfs[7]
df_item_fee = dfs[8]
df_market = dfs[9]
df_requires = dfs[10]
df_usage_1 = dfs[11]
df_usage_2 = dfs[12]
item_tables = [df_item_1, df_item_2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref_norm'] = item_rows['item_ref'].apply(norm)
options = list(item_rows['item_ref_norm'].unique())
item_ref_to_orig = dict(zip(item_rows['item_ref_norm'], item_rows['item_ref']))
sections = sorted(item_rows['location_id'].unique())
categories = sorted(set(item_rows['category'].unique()) | set(df_category['category'].unique()))
resources = sorted(set(df_capacity_ledger['resource'].unique()) | set(df_usage_1['resource'].unique()) | set(df_usage_2['resource'].unique()))
df_bundle['item_a_norm'] = df_bundle['item_a'].apply(norm)
df_bundle['item_b_norm'] = df_bundle['item_b'].apply(norm)
bundles = [(row['item_a_norm'], row['item_b_norm']) for (_, row) in df_bundle.iterrows()]
df_incompatible['item_a_norm'] = df_incompatible['item_a'].apply(norm)
df_incompatible['item_b_norm'] = df_incompatible['item_b'].apply(norm)
incompatibles = [(row['item_a_norm'], row['item_b_norm']) for (_, row) in df_incompatible.iterrows()]
df_requires['item_ref_norm'] = df_requires['item_ref'].apply(norm)
df_requires['prerequisite_ref_norm'] = df_requires['prerequisite_ref'].apply(norm)
requires = [(row['item_ref_norm'], row['prerequisite_ref_norm']) for (_, row) in df_requires.iterrows()]
df_benefit['item_ref_norm'] = df_benefit['item_ref'].apply(norm)
benefit_per_unit = df_benefit.groupby('item_ref_norm')['amount_cents'].apply(lambda s: s.astype(int).sum()).to_dict()
for i in options:
    if i not in benefit_per_unit:
        benefit_per_unit[i] = 0
df_item_fee['item_ref_norm'] = df_item_fee['item_ref'].apply(norm)
item_fee = {row['item_ref_norm']: int(row['activation_fee_cents']) for (_, row) in df_item_fee.iterrows()}
for i in options:
    if i not in item_fee:
        item_fee[i] = 0
category_fee = {row['category']: int(row['activation_fee_cents']) for (_, row) in df_category.iterrows()}
for c in categories:
    if c not in category_fee:
        category_fee[c] = 0
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
df_capacity_ledger['unit_norm'] = df_capacity_ledger['unit'].apply(norm)
df_capacity_ledger['resource_norm'] = df_capacity_ledger['resource'].apply(norm)

def to_ml(amount, unit):
    if norm(unit) == 'ml':
        return int(amount)
    elif norm(unit) == 'liter':
        return int(amount) * 1000
    else:
        raise ValueError(f'Unknown unit: {unit}')
df_capacity_ledger['amount_ml'] = df_capacity_ledger.apply(lambda row: to_ml(row['amount'], row['unit']), axis=1)
capacity_by_resource = df_capacity_ledger.groupby('resource')['amount_ml'].sum().to_dict()
for r in resources:
    if r not in capacity_by_resource:
        capacity_by_resource[r] = 0

def usage_table_to_dict(df_usage):
    df_usage['item_ref_norm'] = df_usage['item_ref'].apply(norm)
    df_usage['resource_norm'] = df_usage['resource'].apply(norm)
    df_usage['amount'] = df_usage['amount'].astype(int)
    df_usage['unit_norm'] = df_usage['unit'].apply(norm)
    df_usage['amount_ml'] = df_usage.apply(lambda row: to_ml(row['amount'], row['unit']), axis=1)
    usage = {}
    for (_, row) in df_usage.iterrows():
        usage[row['item_ref_norm'], row['resource']] = row['amount_ml']
    return usage
usage_1 = usage_table_to_dict(df_usage_1)
usage_2 = usage_table_to_dict(df_usage_2)
usage = usage_1.copy()
usage.update(usage_2)
usage_per_pack = {}
for i in options:
    for r in resources:
        usage_per_pack[i, r] = usage.get((i, r), 0)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['category'] = item_rows['category'].astype(str)
item_rows['location_id'] = item_rows['location_id'].astype(str)
option_min_lot = {row['item_ref_norm']: row['minimum_lot'] for (_, row) in item_rows.iterrows()}
option_max_order = {row['item_ref_norm']: row['maximum_order'] for (_, row) in item_rows.iterrows()}
option_authorized = {row['item_ref_norm']: row['authorized'] for (_, row) in item_rows.iterrows()}
option_category = {row['item_ref_norm']: row['category'] for (_, row) in item_rows.iterrows()}
option_section = {row['item_ref_norm']: row['location_id'] for (_, row) in item_rows.iterrows()}
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
category_min_qty = {row['category']: row['minimum_quantity'] for (_, row) in df_category.iterrows()}
category_max_qty = {row['category']: row['maximum_quantity'] for (_, row) in df_category.iterrows()}
for c in categories:
    if c not in category_min_qty:
        category_min_qty[c] = 0
    if c not in category_max_qty:
        category_max_qty[c] = 999999
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    bundle_bonus[row['item_a_norm'], row['item_b_norm']] = int(row['bonus_cents'])
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, ub=[option_max_order[i] if option_authorized[i] else 0 for i in options], name='')
z_vars = m.addVars(options, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in options:
    if not option_authorized[i]:
        m.addConstr(x_vars[i] == 0, name=f'auth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'z0_{i}')
    else:
        m.addConstr(x_vars[i] >= option_min_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= option_max_order[i] * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(x_vars[i] <= option_max_order[i] * z_vars[i], name=f'linkz_{i}')
        m.addConstr(x_vars[i] >= 0, name=f'xnonneg_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_pack[i, r] * x_vars[i] for i in options)) <= capacity_by_resource[r], name=f'cap_{r}')
for c in categories:
    options_in_c = [i for i in options if option_category[i] == c]
    if options_in_c:
        m.addConstr(gp.quicksum((x_vars[i] for i in options_in_c)) >= category_min_qty[c] * w_vars[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((x_vars[i] for i in options_in_c)) <= category_max_qty[c] * w_vars[c], name=f'catmax_{c}')
        for i in options_in_c:
            m.addConstr(z_vars[i] <= w_vars[c], name=f'catw_{c}_{i}')
        m.addConstr(gp.quicksum((z_vars[i] for i in options_in_c)) >= w_vars[c], name=f'catw_sum_{c}')
    else:
        m.addConstr(w_vars[c] == 0, name=f'catw0_{c}')
for (i, j) in incompatibles:
    if i in options and j in options:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, prereq) in requires:
    if i in options and prereq in options:
        m.addConstr(z_vars[i] <= z_vars[prereq], name=f'req_{i}_{prereq}')
for (a, b) in bundles:
    if a in options and b in options:
        m.addConstr(b_vars[a, b] <= z_vars[a], name=f'bundle_a_{a}_{b}')
        m.addConstr(b_vars[a, b] <= z_vars[b], name=f'bundle_b_{a}_{b}')
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle_and_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle0_{a}_{b}')
obj = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in options)) - gp.quicksum((item_fee[i] * z_vars[i] for i in options)) - gp.quicksum((category_fee[c] * w_vars[c] for c in categories)) + gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()