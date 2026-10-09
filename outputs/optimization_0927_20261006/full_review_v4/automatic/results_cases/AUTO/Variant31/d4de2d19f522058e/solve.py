import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_cap['amount'] = df_cap['amount'].astype(int)
df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_cat['minimum_quantity'] = df_cat['minimum_quantity'].astype(int)
df_cat['maximum_quantity'] = df_cat['maximum_quantity'].astype(int)
df_cat['activation_fee_cents'] = df_cat['activation_fee_cents'].astype(int)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_usage['amount'] = df_usage['amount'].astype(int)
item_ids = df_item['item_ref'].tolist()
cat_ids = df_cat['category'].tolist()
resource_ids = sorted(set(df_usage['resource']).union(df_cap['resource']))
bundle_keys = list(df_bundle.index)
bundle_item_a = df_bundle['item_a'].tolist()
bundle_item_b = df_bundle['item_b'].tolist()
bundle_bonus = df_bundle['bonus_cents'].tolist()
incompat_pairs = list(df_incompat.itertuples(index=False, name=None))
requires_pairs = list(df_requires.itertuples(index=False, name=None))
authorized = df_item.set_index('item_ref')['authorized'].to_dict()
minimum_lot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
maximum_order = df_item.set_index('item_ref')['maximum_order'].to_dict()
unit_benefit_cents = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
item_category = df_item.set_index('item_ref')['category'].to_dict()
cat_min_qty = df_cat.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = df_cat.set_index('category')['maximum_quantity'].to_dict()
cat_activation_fee = df_cat.set_index('category')['activation_fee_cents'].to_dict()
usage_dict = {}
for row in df_usage.itertuples(index=False):
    usage_dict[row.item_ref, row.resource] = row.amount
cap_opening = df_cap[df_cap['entry'].str.strip().str.casefold() == 'opening']
cap_reservation = df_cap[df_cap['entry'].str.strip().str.casefold() == 'reservation']
resource_capacity = {}
for r in resource_ids:
    opening = cap_opening[cap_opening['resource'] == r]['amount'].sum()
    reservation = cap_reservation[cap_reservation['resource'] == r]['amount'].sum()
    resource_capacity[r] = opening + reservation
cat_items = {c: set(df_item[df_item['category'] == c]['item_ref']) for c in cat_ids}
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(cat_ids, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(len(bundle_keys), vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if authorized[i] == 0:
        m.addConstr(q_vars[i] == 0, name=f'auth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'z0_{i}')
    else:
        m.addConstr(q_vars[i] >= minimum_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= maximum_order[i] * z_vars[i], name=f'maxlot_{i}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * q_vars[i] for i in item_ids)) <= resource_capacity[r], name=f'res_{r}')
for c in cat_ids:
    items_in_c = cat_items[c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= cat_min_qty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= cat_max_qty[c], name=f'catmax_{c}')
for c in cat_ids:
    for i in cat_items[c]:
        m.addConstr(y_vars[c] >= z_vars[i], name=f'catact_{c}_{i}')
for (_, i, j) in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (_, i, prereq) in requires_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(z_vars[i] <= z_vars[prereq], name=f'req_{i}_{prereq}')
for (idx, (i_a, i_b)) in enumerate(zip(bundle_item_a, bundle_item_b)):
    if i_a in item_ids and i_b in item_ids:
        m.addConstr(b_vars[idx] <= z_vars[i_a], name=f'bundle1_{idx}')
        m.addConstr(b_vars[idx] <= z_vars[i_b], name=f'bundle2_{idx}')
        m.addConstr(b_vars[idx] >= z_vars[i_a] + z_vars[i_b] - 1, name=f'bundle3_{idx}')
    else:
        m.addConstr(b_vars[idx] == 0, name=f'bundle0_{idx}')
obj_benefit = gp.quicksum((unit_benefit_cents[i] * q_vars[i] for i in item_ids))
obj_item_fee = gp.quicksum((item_fee_cents[i] * z_vars[i] for i in item_ids))
obj_cat_fee = gp.quicksum((cat_activation_fee[c] * y_vars[c] for c in cat_ids))
obj_bundle = gp.quicksum((bundle_bonus[idx] * b_vars[idx] for idx in range(len(bundle_keys))))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()