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
df_item['authorized'] = df_item['authorized'].astype(int)
authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].tolist()
item_params = df_item.set_index('item_ref').loc[authorized_items]
minimum_lot = item_params['minimum_lot'].astype(int).to_dict()
maximum_order = item_params['maximum_order'].astype(int).to_dict()
unit_benefit_cents = item_params['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = item_params['item_fee_cents'].astype(int).to_dict()
item_category = item_params['category'].to_dict()
categories = df_category['category'].tolist()
category_params = df_category.set_index('category')
minimum_quantity = category_params['minimum_quantity'].astype(int).to_dict()
maximum_quantity = category_params['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = category_params['activation_fee_cents'].astype(int).to_dict()
items_by_category = {c: [i for i in authorized_items if item_category[i] == c] for c in categories}
df_usage_auth = df_usage[df_usage['item_ref'].isin(authorized_items)].copy()
df_usage_auth['amount'] = df_usage_auth['amount'].astype(int)
usage_by_item_resource = {}
for (_, row) in df_usage_auth.iterrows():
    i = row['item_ref']
    r = row['resource']
    usage_by_item_resource.setdefault(i, {})[r] = (row['amount'], row['unit'])
df_capacity['amount'] = df_capacity['amount'].astype(int)
resource_capacity = {}
resource_unit = {}
for r in df_capacity['resource'].unique():
    df_r = df_capacity[df_capacity['resource'] == r]
    units = set(df_r['unit'])
    if len(units) != 1:
        raise ValueError(f'Resource {r} has multiple units in capacity_ledger: {units}')
    unit = units.pop()
    resource_unit[r] = unit
    resource_capacity[r] = df_r['amount'].sum()
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    (i, j) = (row['item_ref'], row['prerequisite_ref'])
    if i in authorized_items and j in authorized_items:
        prereq_pairs.append((i, j))
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_tuples = []
bundle_bonus = {}
for (idx, row) in df_bundle.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in authorized_items and b in authorized_items:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
m = gp.Model('BakeryOrder')
q_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(q_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(q_vars[i] <= maximum_order[i] * y_vars[i])
for c in categories:
    for i in items_by_category[c]:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_by_category[c])))
for c in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) <= maximum_quantity[c])
unit_conv = {('kwh', 'wh'): 1000, ('wh', 'wh'): 1, ('hour', 'minute'): 60, ('minute', 'minute'): 1, ('liter', 'ml'): 1000, ('ml', 'ml'): 1}
for r in resource_capacity:
    cap_unit = resource_unit[r]
    usage_expr = []
    for i in authorized_items:
        if r in usage_by_item_resource.get(i, {}):
            (amt, u_unit) = usage_by_item_resource[i][r]
            key = (u_unit.strip().casefold(), cap_unit.strip().casefold())
            if key not in unit_conv:
                raise ValueError(f'Cannot convert {u_unit} to {cap_unit} for resource {r}')
            factor = unit_conv[key]
            usage_expr.append(amt * factor * q_vars[i])
    if usage_expr:
        m.addConstr(gp.quicksum(usage_expr) <= resource_capacity[r])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[j])
for (a, b) in bundle_tuples:
    m.addConstr(b_vars[a, b] <= y_vars[a])
    m.addConstr(b_vars[a, b] <= y_vars[b])
    m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
obj = gp.quicksum((unit_benefit_cents[i] * q_vars[i] for i in authorized_items)) + gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_tuples)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in authorized_items)) - gp.quicksum((activation_fee_cents[c] * z_vars[c] for c in categories))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()