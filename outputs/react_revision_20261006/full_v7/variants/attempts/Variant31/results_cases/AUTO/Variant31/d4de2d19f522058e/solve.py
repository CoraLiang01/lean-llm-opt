import gurobipy as gp
import pandas as pd
import numpy as np
path_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv'
path_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv'
path_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv'
path_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv'
path_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv'
path_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv'
path_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv'
path_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv'
path_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv'
bundle_df = pd.read_csv(path_bundle, dtype=str, keep_default_na=False)
capacity_ledger_df = pd.read_csv(path_capacity_ledger, dtype=str, keep_default_na=False)
category_df = pd.read_csv(path_category, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(path_identity, dtype=str, keep_default_na=False)
incompatible_df = pd.read_csv(path_incompatible, dtype=str, keep_default_na=False)
item_df = pd.read_csv(path_item, dtype=str, keep_default_na=False)
market_df = pd.read_csv(path_market, dtype=str, keep_default_na=False)
requires_df = pd.read_csv(path_requires, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(path_usage, dtype=str, keep_default_na=False)
item_rows = item_df[item_df['table'].str.casefold().str.strip() == 'item']
item_ids = item_rows['item_ref'].tolist()
category_rows = category_df[category_df['table'].str.casefold().str.strip() == 'category']
category_ids = category_rows['category'].tolist()
capacity_ledger_rows = capacity_ledger_df[capacity_ledger_df['table'].str.casefold().str.strip() == 'capacity_ledger']
resource_ids = sorted(set(capacity_ledger_rows['resource'].tolist()) | set(usage_df[usage_df['table'].str.casefold().str.strip() == 'usage']['resource'].tolist()))
bundle_rows = bundle_df[bundle_df['table'].str.casefold().str.strip() == 'bundle']
bundle_keys = list(bundle_rows.index)
incompatible_rows = incompatible_df[incompatible_df['table'].str.casefold().str.strip() == 'incompatible']
incompatible_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompatible_rows.iterrows()]
requires_rows = requires_df[requires_df['table'].str.casefold().str.strip() == 'requires']
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_rows.iterrows()]

def to_int(series):
    return pd.to_numeric(series, errors='raise').astype(int)
authorized = dict(zip(item_rows['item_ref'], to_int(item_rows['authorized'])))
minimum_lot = dict(zip(item_rows['item_ref'], to_int(item_rows['minimum_lot'])))
maximum_order = dict(zip(item_rows['item_ref'], to_int(item_rows['maximum_order'])))
unit_benefit_cents = dict(zip(item_rows['item_ref'], to_int(item_rows['unit_benefit_cents'])))
item_fee_cents = dict(zip(item_rows['item_ref'], to_int(item_rows['item_fee_cents'])))
item_category = dict(zip(item_rows['item_ref'], item_rows['category']))
minimum_quantity = dict(zip(category_rows['category'], to_int(category_rows['minimum_quantity'])))
maximum_quantity = dict(zip(category_rows['category'], to_int(category_rows['maximum_quantity'])))
activation_fee_cents = dict(zip(category_rows['category'], to_int(category_rows['activation_fee_cents'])))
usage_rows = usage_df[usage_df['table'].str.casefold().str.strip() == 'usage']
usage_per_item_resource = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    usage_per_item_resource[i, r] = amt
capacity_ledger_rows = capacity_ledger_df[capacity_ledger_df['table'].str.casefold().str.strip() == 'capacity_ledger']
resource_capacity = {}
for r in resource_ids:
    opening = capacity_ledger_rows[(capacity_ledger_rows['resource'] == r) & (capacity_ledger_rows['entry'].str.casefold() == 'opening')]
    reservation = capacity_ledger_rows[(capacity_ledger_rows['resource'] == r) & (capacity_ledger_rows['entry'].str.casefold() == 'reservation')]
    opening_amt = opening['amount'].astype(int).sum() if not opening.empty else 0
    reservation_amt = reservation['amount'].astype(int).sum() if not reservation.empty else 0
    resource_capacity[r] = opening_amt + reservation_amt
bundle_bonus = {}
for (idx, row) in bundle_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    bonus = int(row['bonus_cents'])
    bundle_bonus[idx] = {'item_a': a, 'item_b': b, 'bonus_cents': bonus}
category_items = {c: [] for c in category_ids}
for i in item_ids:
    c = item_category[i]
    if c in category_items:
        category_items[c].append(i)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
quantity_vars = m.addVars(item_ids, lb=0, ub=[maximum_order[i] if authorized[i] else 0 for i in item_ids], vtype=gp.GRB.INTEGER, name='')
selection_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
category_activation_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
bundle_activation_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if authorized[i]:
        m.addConstr(quantity_vars[i] >= minimum_lot[i] * selection_vars[i], name=f'lot_lb_{i}')
        m.addConstr(quantity_vars[i] <= maximum_order[i] * selection_vars[i], name=f'lot_ub_{i}')
        m.addConstr(quantity_vars[i] <= maximum_order[i], name=f'max_order_{i}')
        m.addConstr(quantity_vars[i] >= 0, name=f'nonneg_{i}')
    else:
        m.addConstr(quantity_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(selection_vars[i] == 0, name=f'unauth_z_{i}')
for i in item_ids:
    m.addConstr(quantity_vars[i] >= selection_vars[i], name=f'zlink_lb_{i}')
for r in resource_ids:
    expr = gp.quicksum((usage_per_item_resource.get((i, r), 0) * quantity_vars[i] for i in item_ids))
    m.addConstr(expr <= resource_capacity[r], name=f'res_cap_{r}')
for c in category_ids:
    items_in_c = category_items[c]
    m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) >= minimum_quantity[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) <= maximum_quantity[c], name=f'cat_max_{c}')
    for i in items_in_c:
        m.addConstr(selection_vars[i] <= category_activation_vars[c], name=f'cat_act1_{c}_{i}')
    m.addConstr(category_activation_vars[c] <= gp.quicksum((selection_vars[i] for i in items_in_c)), name=f'cat_act2_{c}')
for (i, j) in incompatible_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(selection_vars[i] + selection_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, prereq) in requires_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(selection_vars[i] <= selection_vars[prereq], name=f'req_{i}_{prereq}')
for b in bundle_keys:
    a = bundle_bonus[b]['item_a']
    b_ = bundle_bonus[b]['item_b']
    if a in item_ids and b_ in item_ids:
        m.addConstr(bundle_activation_vars[b] <= selection_vars[a], name=f'bundle1_{b}')
        m.addConstr(bundle_activation_vars[b] <= selection_vars[b_], name=f'bundle2_{b}')
        m.addConstr(bundle_activation_vars[b] >= selection_vars[a] + selection_vars[b_] - 1, name=f'bundle3_{b}')
    else:
        m.addConstr(bundle_activation_vars[b] == 0, name=f'bundle0_{b}')
item_benefit_expr = gp.quicksum((unit_benefit_cents[i] * quantity_vars[i] for i in item_ids))
item_fee_expr = gp.quicksum((item_fee_cents[i] * selection_vars[i] for i in item_ids))
category_fee_expr = gp.quicksum((activation_fee_cents[c] * category_activation_vars[c] for c in category_ids))
bundle_bonus_expr = gp.quicksum((bundle_bonus[b]['bonus_cents'] * bundle_activation_vars[b] for b in bundle_keys))
m.setObjective(item_benefit_expr - item_fee_expr - category_fee_expr + bundle_bonus_expr, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')