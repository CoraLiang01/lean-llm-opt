import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
item_df = pd.read_csv(f_item, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(f_usage, dtype=str, keep_default_na=False)
category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
incompat_df = pd.read_csv(f_incompat, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(f_capacity, dtype=str, keep_default_na=False)
item_df['authorized'] = item_df['authorized'].astype(int)
authorized_items_df = item_df[item_df['authorized'] == 1].copy()
authorized_items = list(authorized_items_df['item_ref'])
categories = list(category_df['category'])
usage_resources = set(usage_df['resource'].unique())
capacity_resources = set(capacity_df['resource'].unique())
resources = sorted(usage_resources.union(capacity_resources))
incompat_pairs = []
for (_, row) in incompat_df.iterrows():
    incompat_pairs.append((row['item_a'], row['item_b']))
prereq_pairs = []
for (_, row) in requires_df.iterrows():
    prereq_pairs.append((row['item_ref'], row['prerequisite_ref']))
bundle_pairs = []
bundle_bonus = {}
for (idx, row) in bundle_df.iterrows():
    bundle_pairs.append((row['item_a'], row['item_b']))
    bundle_bonus[row['item_a'], row['item_b']] = int(row['bonus_cents'])
item_params = {}
for (_, row) in authorized_items_df.iterrows():
    i = row['item_ref']
    item_params[i] = {'category': row['category'], 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'unit_benefit_cents': int(row['unit_benefit_cents']), 'item_fee_cents': int(row['item_fee_cents'])}
category_params = {}
for (_, row) in category_df.iterrows():
    c = row['category']
    category_params[c] = {'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity']), 'activation_fee_cents': int(row['activation_fee_cents'])}
unit_to_base = {'liter': ('ml', 1000), 'ml': ('ml', 1), 'hour': ('minute', 60), 'minute': ('minute', 1), 'kwh': ('wh', 1000), 'wh': ('wh', 1)}
usage_amounts = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amount = int(row['amount'])
    unit = row['unit'].strip().casefold()
    if unit not in unit_to_base:
        raise ValueError(f'Unknown unit in usage: {unit}')
    (base_unit, factor) = unit_to_base[unit]
    amount_base = amount * factor
    if i in authorized_items:
        usage_amounts[i, r] = amount_base
capacity_amounts = {}
for r in resources:
    total = 0
    for (_, row) in capacity_df[capacity_df['resource'] == r].iterrows():
        amount = int(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit not in unit_to_base:
            raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
        (base_unit, factor) = unit_to_base[unit]
        amount_base = amount * factor
        total += amount_base
    capacity_amounts[r] = total
items_by_category = {c: [] for c in categories}
for i in authorized_items:
    c = item_params[i]['category']
    items_by_category[c].append(i)
m = gp.Model('BakeryOrderNetBenefit')
q_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    min_lot = item_params[i]['minimum_lot']
    max_order = item_params[i]['maximum_order']
    m.addConstr(q_vars[i] >= min_lot * z_vars[i])
    m.addConstr(q_vars[i] <= max_order * z_vars[i])
for c in categories:
    for i in items_by_category[c]:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_by_category[c])))
for c in categories:
    minq = category_params[c]['minimum_quantity']
    maxq = category_params[c]['maximum_quantity']
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) >= minq)
    m.addConstr(gp.quicksum((q_vars[i] for i in items_by_category[c])) <= maxq)
for r in resources:
    m.addConstr(gp.quicksum((usage_amounts.get((i, r), 0) * q_vars[i] for i in authorized_items)) <= capacity_amounts[r])
for (i, j) in incompat_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, j) in prereq_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(z_vars[i] <= z_vars[j])
for (i, j) in bundle_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(b_vars[i, j] <= z_vars[i])
        m.addConstr(b_vars[i, j] <= z_vars[j])
        m.addConstr(b_vars[i, j] >= z_vars[i] + z_vars[j] - 1)
    else:
        m.addConstr(b_vars[i, j] == 0)
item_obj = gp.quicksum((item_params[i]['unit_benefit_cents'] * q_vars[i] - item_params[i]['item_fee_cents'] * z_vars[i] for i in authorized_items))
cat_obj = gp.quicksum((category_params[c]['activation_fee_cents'] * y_vars[c] for c in categories))
bundle_obj = gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(item_obj - cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()