import gurobipy as gp
import pandas as pd
import numpy as np
import re

def convert_to_base(amount, unit):
    u = unit.strip().lower()
    if u == 'ml':
        return amount
    elif u == 'liter':
        return amount * 1000
    elif u == 'wh':
        return amount
    elif u == 'kwh':
        return amount * 1000
    elif u == 'minute':
        return amount
    elif u == 'hour':
        return amount * 60
    else:
        raise ValueError(f'Unknown unit: {unit}')
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
categories = df_category['category'].tolist()
item_to_category = df_item.set_index('item_ref')['category'].to_dict()
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
minimum_lot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
maximum_order = df_item.set_index('item_ref')['maximum_order'].to_dict()
df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
unit_benefit_cents = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
minimum_quantity = df_category.set_index('category')['minimum_quantity'].to_dict()
maximum_quantity = df_category.set_index('category')['maximum_quantity'].to_dict()
activation_fee_cents = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_usage['amount'] = df_usage['amount'].astype(int)
usage_rows = df_usage[df_usage['item_ref'].isin(authorized_items)]
resources = sorted(usage_rows['resource'].unique())
usage = {}
for (_, row) in usage_rows.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = convert_to_base(row['amount'], row['unit'])
    usage.setdefault(i, {})[r] = amt
df_capacity['amount'] = df_capacity['amount'].astype(int)
capacity = {}
for r in df_capacity['resource'].unique():
    rows = df_capacity[df_capacity['resource'] == r]
    total = 0
    for (_, row) in rows.iterrows():
        amt = convert_to_base(row['amount'], row['unit'])
        total += amt
    capacity[r] = total
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if i in authorized_items and k in authorized_items:
        prereq_pairs.append((i, k))
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    i = row['item_a']
    j = row['item_b']
    if i in authorized_items and j in authorized_items:
        bundle_pairs.append((i, j))
        bundle_bonus[i, j] = row['bonus_cents']
items_in_category = {g: [] for g in categories}
for i in authorized_items:
    g = item_to_category[i]
    items_in_category[g].append(i)
m = gp.Model('BakeryOrderNetBenefit')
x_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    min_lot = minimum_lot[i]
    max_ord = maximum_order[i]
    m.addConstr(x_vars[i] <= max_ord * y_vars[i], name=f'xmax_{i}')
    m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'xmin_{i}')
    m.addConstr(x_vars[i] <= max_ord, name=f'xmax2_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'xnonneg_{i}')
for g in categories:
    items = items_in_category[g]
    if items:
        m.addConstr(gp.quicksum((x_vars[i] for i in items)) >= minimum_quantity[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items)) <= maximum_quantity[g], name=f'catmax_{g}')
        for i in items:
            m.addConstr(y_vars[i] <= z_vars[g], name=f'y2z_{i}_{g}')
        m.addConstr(gp.quicksum((y_vars[i] for i in items)) >= z_vars[g], name=f'z_lb_{g}')
    else:
        m.addConstr(z_vars[g] == 0, name=f'z0_{g}')
for r in resources:
    m.addConstr(gp.quicksum((usage.get(i, {}).get(r, 0) * x_vars[i] for i in authorized_items)) <= capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, k) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[k], name=f'prereq_{i}_{k}')
for (i, j) in bundle_pairs:
    m.addConstr(b_vars[i, j] <= y_vars[i], name=f'b_le_y_{i}_{j}')
    m.addConstr(b_vars[i, j] <= y_vars[j], name=f'b_le_y_{j}_{i}')
    m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1, name=f'b_ge_sum_{i}_{j}')
obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in authorized_items)) + gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs)) - gp.quicksum((item_fee_cents[i] * y_vars[i] for i in authorized_items)) - gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in categories))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()