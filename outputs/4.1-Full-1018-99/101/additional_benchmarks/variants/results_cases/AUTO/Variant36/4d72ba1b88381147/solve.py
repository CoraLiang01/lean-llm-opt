import gurobipy as gp
import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
df_bundle = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv')
df_capacity_ledger = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv')
df_category = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv')
df_identity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv')
df_incompatible = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv')
df_item_1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv')
df_item_2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv')
df_market = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv')
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv')
df_usage_1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv')
df_usage_2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv')
item_tables = []
for df in [df_item_1, df_item_2]:
    mask = df['table'].str.strip().str.casefold() == 'item'
    item_tables.append(df[mask].copy())
df_items = pd.concat(item_tables, ignore_index=True)
item_refs = df_items['item_ref'].tolist()
cat_mask = df_category['table'].str.strip().str.casefold() == 'category'
df_categories = df_category[cat_mask].copy()
categories = df_categories['category'].tolist()
cl_mask = df_capacity_ledger['table'].str.strip().str.casefold() == 'capacity_ledger'
df_cl = df_capacity_ledger[cl_mask].copy()
resources_cl = df_cl['resource'].unique().tolist()
usage_tables = []
for df in [df_usage_1, df_usage_2]:
    mask = df['table'].str.strip().str.casefold() == 'usage'
    usage_tables.append(df[mask].copy())
df_usages = pd.concat(usage_tables, ignore_index=True)
resources_usage = df_usages['resource'].unique().tolist()
resources = sorted(set(resources_cl) | set(resources_usage))
bundle_mask = df_bundle['table'].str.strip().str.casefold() == 'bundle'
df_bundles = df_bundle[bundle_mask].copy()
bundle_indices = df_bundles.index.tolist()
incomp_mask = df_incompatible['table'].str.strip().str.casefold() == 'incompatible'
df_incomp = df_incompatible[incomp_mask].copy()
incomp_pairs = list(df_incomp[['item_a', 'item_b']].itertuples(index=False, name=None))
req_mask = df_requires['table'].str.strip().str.casefold() == 'requires'
df_req = df_requires[req_mask].copy()
req_pairs = list(df_req[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))

def to_int_series(df, col):
    return df[col].astype(int)
unit_benefit_cents = df_items.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = df_items.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
minimum_lot = df_items.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = df_items.set_index('item_ref')['maximum_order'].astype(int).to_dict()
authorized = df_items.set_index('item_ref')['authorized'].astype(int).to_dict()
item_category = df_items.set_index('item_ref')['category'].to_dict()
item_location = df_items.set_index('item_ref')['location_id'].to_dict()
minimum_quantity = df_categories.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = df_categories.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = df_categories.set_index('category')['activation_fee_cents'].astype(int).to_dict()
capacity = {}
for r in resources:
    mask = df_cl['resource'].str.strip() == r
    capacity[r] = df_cl.loc[mask, 'amount'].astype(int).sum()
usage = {}
for _, row in df_usages.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    usage.setdefault(i, {})[r] = amt
for i in item_refs:
    usage.setdefault(i, {})
    for r in resources:
        usage[i].setdefault(r, 0)
bundle_item_a = df_bundles['item_a'].tolist()
bundle_item_b = df_bundles['item_b'].tolist()
bundle_bonus_cents = df_bundles['bonus_cents'].astype(int).tolist()
bundle_keys = bundle_indices
cat_items = {g: [] for g in categories}
for i in item_refs:
    g = item_category[i]
    cat_items[g].append(i)
m = gp.Model('FC_EAST_HVAC_AC_Placement')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for r in resources:
    m.addConstr(gp.quicksum((usage[i][r] * x_vars[i] for i in item_refs)) <= capacity[r], name=f'capacity_{r}')
for i in item_refs:
    if authorized[i] == 0:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i], name=f'maxorder_{i}')
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i], name=f'link_yx_{i}')
        m.addConstr(x_vars[i] >= y_vars[i], name=f'link_xy_{i}')
for g in categories:
    m.addConstr(gp.quicksum((x_vars[i] for i in cat_items[g])) >= minimum_quantity[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in cat_items[g])) <= maximum_quantity[g], name=f'cat_max_{g}')
for g in categories:
    for i in cat_items[g]:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'cat_act1_{g}_{i}')
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in cat_items[g])), name=f'cat_act2_{g}')
for i, j in incomp_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incomp_{i}_{j}')
for i, j in req_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] <= y_vars[j], name=f'req_{i}_{j}')
for b, i, j in zip(bundle_keys, bundle_item_a, bundle_item_b):
    if i in y_vars and j in y_vars:
        m.addConstr(w_vars[b] <= y_vars[i], name=f'bundle1_{b}')
        m.addConstr(w_vars[b] <= y_vars[j], name=f'bundle2_{b}')
        m.addConstr(w_vars[b] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{b}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle0_{b}')
item_obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] - item_fee_cents[i] * y_vars[i] for i in item_refs))
cat_obj = gp.quicksum((-activation_fee_cents[g] * z_vars[g] for g in categories))
bundle_obj = gp.quicksum((bundle_bonus_cents[k] * w_vars[bundle_keys[k]] for k in range(len(bundle_keys))))
m.setObjective(item_obj + cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()