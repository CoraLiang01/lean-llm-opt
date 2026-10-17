import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
item_rows = df_item[df_item['table'].str.strip().str.casefold() == 'item']
item_refs = item_rows['item_ref'].tolist()
item_rows['authorized'] = item_rows['authorized'].astype(int)
authorized_items = [i for i in item_refs if int(item_rows.loc[item_rows['item_ref'] == i, 'authorized'].iloc[0]) == 1]
cat_rows = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = cat_rows['category'].tolist()
usage_rows = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']
capacity_rows = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']
resources = sorted(set(usage_rows['resource'].tolist()) | set(capacity_rows['resource'].tolist()))
incompat_rows = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for _, row in incompat_rows.iterrows()]
requires_rows = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for _, row in requires_rows.iterrows()]
bundle_rows = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for _, row in bundle_rows.iterrows()]
item_rows = item_rows.set_index('item_ref')
minimum_lot = item_rows['minimum_lot'].astype(int).to_dict()
maximum_order = item_rows['maximum_order'].astype(int).to_dict()
unit_benefit_cents = item_rows['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = item_rows['item_fee_cents'].astype(int).to_dict()
item_category = item_rows['category'].to_dict()
cat_rows = cat_rows.set_index('category')
minimum_quantity = cat_rows['minimum_quantity'].astype(int).to_dict()
maximum_quantity = cat_rows['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = cat_rows['activation_fee_cents'].astype(int).to_dict()
usage_rows = usage_rows.copy()
usage_rows['amount'] = usage_rows['amount'].astype(int)
usage_map = {}
for _, row in usage_rows.iterrows():
    usage_map[row['item_ref'], row['resource']] = (row['amount'], row['unit'].strip().casefold())
capacity_rows = capacity_rows.copy()
capacity_rows['amount'] = capacity_rows['amount'].astype(int)
capacity_agg = {}
for r in resources:
    rows = capacity_rows[capacity_rows['resource'] == r]
    total = 0
    unit = None
    for _, row in rows.iterrows():
        total += row['amount']
        unit = row['unit'].strip().casefold()
    capacity_agg[r] = (total, unit)
bundle_bonus = {}
for _, row in bundle_rows.iterrows():
    bundle_bonus[row['item_a'], row['item_b']] = int(row['bonus_cents'])

def get_unit_factor(from_unit, to_unit):
    u_from = from_unit.strip().casefold()
    u_to = to_unit.strip().casefold()
    if u_from == u_to:
        return 1
    if u_from == 'kwh' and u_to == 'wh':
        return 1000
    if u_from == 'wh' and u_to == 'kwh':
        return 1 / 1000
    if u_from == 'liter' and u_to == 'ml':
        return 1000
    if u_from == 'ml' and u_to == 'liter':
        return 1 / 1000
    if u_from == 'hour' and u_to == 'minute':
        return 60
    if u_from == 'minute' and u_to == 'hour':
        return 1 / 60
    raise ValueError(f'Unknown unit conversion: {u_from} -> {u_to}')
m = gp.Model('BakeryOrder')
q_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
g_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(q_vars[i] >= minimum_lot[i] * z_vars[i], name=f'minlot_{i}')
    m.addConstr(q_vars[i] <= maximum_order[i] * z_vars[i], name=f'maxorder_{i}')
    m.addConstr(q_vars[i] >= 0, name=f'nonneg_{i}')
    m.addConstr(q_vars[i] <= maximum_order[i], name=f'maxorder2_{i}')
for r in resources:
    cap_total, cap_unit = capacity_agg[r]
    expr = gp.LinExpr()
    for i in authorized_items:
        if (i, r) in usage_map:
            amt, u = usage_map[i, r]
            factor = get_unit_factor(u, cap_unit)
            expr += amt * factor * q_vars[i]
    m.addConstr(expr <= cap_total, name=f'resource_{r}')
for c in categories:
    items_in_c = [i for i in authorized_items if item_category[i] == c]
    if items_in_c:
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= minimum_quantity[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= maximum_quantity[c], name=f'catmax_{c}')
    else:
        m.addConstr(0 >= minimum_quantity[c], name=f'catmin_{c}_empty')
        m.addConstr(0 <= maximum_quantity[c], name=f'catmax_{c}_empty')
for c in categories:
    items_in_c = [i for i in authorized_items if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(z_vars[i] <= g_vars[c], name=f'catact_{c}_{i}')
for a, b in incompat_pairs:
    if a in authorized_items and b in authorized_items:
        m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incompat_{a}_{b}')
for i, prereq in prereq_pairs:
    if i in authorized_items and prereq in authorized_items:
        m.addConstr(z_vars[i] <= z_vars[prereq], name=f'prereq_{i}_{prereq}')
for a, b in bundle_pairs:
    if a in authorized_items and b in authorized_items:
        m.addConstr(b_vars[a, b] <= z_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(b_vars[a, b] <= z_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle_unauth_{a}_{b}')
obj_benefit = gp.quicksum((unit_benefit_cents[i] * q_vars[i] for i in authorized_items))
obj_item_fees = gp.quicksum((item_fee_cents[i] * z_vars[i] for i in authorized_items))
obj_cat_fees = gp.quicksum((activation_fee_cents[c] * g_vars[c] for c in categories))
obj_bundles = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for a, b in bundle_pairs))
m.setObjective(obj_benefit + obj_bundles - obj_item_fees - obj_cat_fees, gp.GRB.MAXIMIZE)
m.optimize()