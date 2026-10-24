import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant31/inputs/batch_03/export_09.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
df_bundle = dfs[0]
df_capacity_ledger = dfs[1]
df_category = dfs[2]
df_identity = dfs[3]
df_incompatible = dfs[4]
df_item = dfs[5]
df_market = dfs[6]
df_requires = dfs[7]
df_usage = dfs[8]
items_df = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_ids = items_df['item_ref'].tolist()
categories_df = df_category[df_category['table'].str.strip().str.casefold() == 'category']
category_ids = categories_df['category'].tolist()
item_to_category = items_df.set_index('item_ref')['category'].to_dict()

def to_int_series(df, col):
    return df[col].astype(int)
authorized = items_df.set_index('item_ref')['authorized'].astype(int).to_dict()
minimum_lot = items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
unit_benefit_cents = items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
minimum_quantity = categories_df.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = categories_df.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = categories_df.set_index('category')['activation_fee_cents'].astype(int).to_dict()
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
resource_ids = usage_df['resource'].unique().tolist()
usage_amount = {i: {r: 0 for r in resource_ids} for i in item_ids}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    if i in usage_amount:
        usage_amount[i][r] = amt
cap_ledger_df = df_capacity_ledger[df_capacity_ledger['table'].str.strip().str.casefold() == 'capacity_ledger']
resource_capacity = {}
for r in resource_ids:
    resource_capacity[r] = cap_ledger_df[cap_ledger_df['resource'] == r]['amount'].astype(int).sum()
bundles_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_tuples = list(zip(bundles_df['item_a'], bundles_df['item_b']))
bundle_bonus_cents = {(row['item_a'], row['item_b']): int(row['bonus_cents']) for (_, row) in bundles_df.iterrows()}
incompat_df = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = set()
for (_, row) in incompat_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    incompat_pairs.add((a, b))
    incompat_pairs.add((b, a))
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
items_by_category = {c: [] for c in category_ids}
for i in item_ids:
    c = item_to_category[i]
    if c in items_by_category:
        items_by_category[c].append(i)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if authorized[i] == 0:
        m.addConstr(q_vars[i] == 0, name=f'auth0_{i}')
        m.addConstr(z_vars[i] == 0, name=f'auth0z_{i}')
    else:
        m.addConstr(q_vars[i] >= minimum_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= maximum_order[i] * z_vars[i], name=f'maxord_{i}')
        m.addConstr(q_vars[i] <= z_vars[i] * maximum_order[i], name=f'linkz_{i}')
for c in category_ids:
    for i in items_by_category[c]:
        m.addConstr(z_vars[i] <= y_vars[c], name=f'catlink_{i}_{c}')
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_by_category[c])), name=f'catany_{c}')
for c in category_ids:
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) >= minimum_quantity[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) <= maximum_quantity[c], name=f'catmax_{c}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage_amount[i][r] * q_vars[i] for i in item_ids)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in requires_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(z_vars[i] <= z_vars[prereq], name=f'requires_{i}_{prereq}')
for (i, j) in bundle_tuples:
    if i in item_ids and j in item_ids:
        m.addConstr(b_vars[i, j] <= z_vars[i], name=f'bundle1_{i}_{j}')
        m.addConstr(b_vars[i, j] <= z_vars[j], name=f'bundle2_{i}_{j}')
        m.addConstr(b_vars[i, j] >= z_vars[i] + z_vars[j] - 1, name=f'bundle3_{i}_{j}')
    else:
        m.addConstr(b_vars[i, j] == 0, name=f'bundle_forbidden_{i}_{j}')
item_benefit = gp.quicksum((unit_benefit_cents[i] * q_vars[i] - item_fee_cents[i] * z_vars[i] for i in item_ids))
category_fees = gp.quicksum((activation_fee_cents[c] * y_vars[c] for c in category_ids))
bundle_bonuses = gp.quicksum((bundle_bonus_cents[i, j] * b_vars[i, j] for (i, j) in bundle_tuples))
m.setObjective(item_benefit - category_fees + bundle_bonuses, gp.GRB.MAXIMIZE)
m.optimize()