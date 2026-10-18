import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
items_df = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_refs = items_df['item_ref'].tolist()
categories_df = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = categories_df['category'].tolist()
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
resources = sorted(set(usage_df['resource'].tolist()) | set(df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']['resource'].tolist()))
bundles_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_keys = list(bundles_df.index)
incompat_df = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = list(incompat_df[['item_a', 'item_b']].itertuples(index=False, name=None))
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = list(requires_df[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))

def to_int_series(df, col):
    return df[col].astype(int)
authorized = dict(zip(items_df['item_ref'], to_int_series(items_df, 'authorized')))
minimum_lot = dict(zip(items_df['item_ref'], to_int_series(items_df, 'minimum_lot')))
maximum_order = dict(zip(items_df['item_ref'], to_int_series(items_df, 'maximum_order')))
unit_benefit_cents = dict(zip(items_df['item_ref'], to_int_series(items_df, 'unit_benefit_cents')))
item_fee_cents = dict(zip(items_df['item_ref'], to_int_series(items_df, 'item_fee_cents')))
item_category = dict(zip(items_df['item_ref'], items_df['category']))
minimum_quantity = dict(zip(categories_df['category'], categories_df['minimum_quantity'].astype(int)))
maximum_quantity = dict(zip(categories_df['category'], categories_df['maximum_quantity'].astype(int)))
activation_fee_cents = dict(zip(categories_df['category'], categories_df['activation_fee_cents'].astype(int)))
usage_amount = {}
for _, row in usage_df.iterrows():
    usage_amount[row['item_ref'], row['resource']] = int(row['amount'])
capacity_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
resource_capacity = {}
for r in resources:
    cap_rows = capacity_df[capacity_df['resource'] == r]
    total = cap_rows['amount'].astype(int).sum()
    resource_capacity[r] = total
bundle_item_a = dict(zip(bundle_keys, bundles_df['item_a']))
bundle_item_b = dict(zip(bundle_keys, bundles_df['item_b']))
bundle_bonus_cents = dict(zip(bundle_keys, bundles_df['bonus_cents'].astype(int)))
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if authorized[i] == 1:
        m.addConstr(q_vars[i] >= minimum_lot[i] * z_vars[i])
        m.addConstr(q_vars[i] <= maximum_order[i] * z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for c in categories:
    items_in_c = [i for i in item_refs if item_category[i] == c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= maximum_quantity[c])
    for i in items_in_c:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)))
for r in resources:
    m.addConstr(gp.quicksum((usage_amount.get((i, r), 0) * q_vars[i] for i in item_refs)) <= resource_capacity[r])
for i, j in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for i, prereq in requires_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(z_vars[i] <= z_vars[prereq])
for b in bundle_keys:
    i_a = bundle_item_a[b]
    i_b = bundle_item_b[b]
    if i_a in item_refs and i_b in item_refs:
        m.addConstr(b_vars[b] <= z_vars[i_a])
        m.addConstr(b_vars[b] <= z_vars[i_b])
        m.addConstr(b_vars[b] >= z_vars[i_a] + z_vars[i_b] - 1)
    else:
        m.addConstr(b_vars[b] == 0)
obj = gp.quicksum((unit_benefit_cents[i] * q_vars[i] for i in item_refs)) - gp.quicksum((item_fee_cents[i] * z_vars[i] for i in item_refs)) - gp.quicksum((activation_fee_cents[c] * y_vars[c] for c in categories)) + gp.quicksum((bundle_bonus_cents[b] * b_vars[b] for b in bundle_keys))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()