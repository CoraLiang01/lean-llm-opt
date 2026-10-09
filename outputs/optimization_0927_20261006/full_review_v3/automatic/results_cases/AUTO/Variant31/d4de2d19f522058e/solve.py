import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
item_rows = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_ids = item_rows['item_ref'].tolist()
item_to_category = item_rows.set_index('item_ref')['category'].to_dict()
item_authorized = item_rows.set_index('item_ref')['authorized'].astype(int).to_dict()
item_min_lot = item_rows.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_max_order = item_rows.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_unit_benefit = item_rows.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee = item_rows.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
cat_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
cat_ids = cat_rows['category'].tolist()
cat_min_qty = cat_rows.set_index('category')['minimum_quantity'].astype(int).to_dict()
cat_max_qty = cat_rows.set_index('category')['maximum_quantity'].astype(int).to_dict()
cat_activation_fee = cat_rows.set_index('category')['activation_fee_cents'].astype(int).to_dict()
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
resource_ids = sorted(usage_rows['resource'].unique())
usage_amount = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    usage_amount[i, r] = amt
cap_rows = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
resource_capacity = {}
for r in cap_rows['resource'].unique():
    rows_r = cap_rows[cap_rows['resource'] == r]
    total = rows_r['amount'].astype(int).sum()
    resource_capacity[r] = total
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_keys = bundle_rows.index.tolist()
bundle_item_a = bundle_rows['item_a'].tolist()
bundle_item_b = bundle_rows['item_b'].tolist()
bundle_bonus = bundle_rows['bonus_cents'].astype(int).tolist()
bundle_tuples = list(zip(bundle_item_a, bundle_item_b))
bundle_idx = list(range(len(bundle_tuples)))
incompat_rows = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = list(zip(incompat_rows['item_a'], incompat_rows['item_b']))
requires_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
requires_pairs = list(zip(requires_rows['item_ref'], requires_rows['prerequisite_ref']))
cat_to_items = {c: [] for c in cat_ids}
for i in item_ids:
    c = item_to_category[i]
    cat_to_items[c].append(i)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(cat_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_idx, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if item_authorized[i] == 0:
        m.addConstr(q_vars[i] == 0, name=f'auth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'z0_{i}')
    else:
        m.addConstr(q_vars[i] >= 0, name=f'q0_{i}')
        m.addConstr(q_vars[i] >= item_min_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= item_max_order[i] * z_vars[i], name=f'maxorder_{i}')
        m.addConstr(q_vars[i] >= z_vars[i], name=f'zlink_lb_{i}')
        m.addConstr(q_vars[i] <= item_max_order[i] * z_vars[i], name=f'zlink_ub_{i}')
for c in cat_ids:
    for i in cat_to_items[c]:
        m.addConstr(z_vars[i] <= w_vars[c], name=f'catlink_{c}_{i}')
    m.addConstr(gp.quicksum((z_vars[i] for i in cat_to_items[c])) >= w_vars[c], name=f'catw_lb_{c}')
for c in cat_ids:
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_to_items[c])) >= cat_min_qty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in cat_to_items[c])) <= cat_max_qty[c], name=f'catmax_{c}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage_amount.get((i, r), 0) * q_vars[i] for i in item_ids)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in requires_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(q_vars[i] <= item_max_order[i] * z_vars[prereq], name=f'req_{i}_{prereq}')
for (idx, (a, b)) in enumerate(bundle_tuples):
    if a in item_ids and b in item_ids:
        m.addConstr(b_vars[idx] <= z_vars[a], name=f'bundle_a_{idx}')
        m.addConstr(b_vars[idx] <= z_vars[b], name=f'bundle_b_{idx}')
        m.addConstr(b_vars[idx] >= z_vars[a] + z_vars[b] - 1, name=f'bundle_ab_{idx}')
    else:
        m.addConstr(b_vars[idx] == 0, name=f'bundle_forbid_{idx}')
obj_benefit = gp.quicksum((item_unit_benefit[i] * q_vars[i] for i in item_ids))
obj_item_fee = gp.quicksum((item_fee[i] * z_vars[i] for i in item_ids))
obj_cat_fee = gp.quicksum((cat_activation_fee[c] * w_vars[c] for c in cat_ids))
obj_bundle = gp.quicksum((bundle_bonus[idx] * b_vars[idx] for idx in bundle_idx))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()