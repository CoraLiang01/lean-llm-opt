import gurobipy as gp
import pandas as pd
import numpy as np
import re
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
item_df = pd.read_csv(f_item, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(f_usage, dtype=str, keep_default_na=False)
category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
incompatible_df = pd.read_csv(f_incompatible, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(f_capacity, dtype=str, keep_default_na=False)
item_df['authorized'] = item_df['authorized'].astype(int)
authorized_items_df = item_df[item_df['authorized'] == 1].copy()
Items = list(authorized_items_df['item_ref'])
Categories = list(category_df['category'])
item_to_category = authorized_items_df.set_index('item_ref')['category'].to_dict()
minimum_lot = authorized_items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = authorized_items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
unit_benefit_cents = authorized_items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = authorized_items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
minimum_quantity = category_df.set_index('category')['minimum_quantity'].to_dict()
maximum_quantity = category_df.set_index('category')['maximum_quantity'].to_dict()
activation_fee_cents = category_df.set_index('category')['activation_fee_cents'].to_dict()
usage_df = usage_df[usage_df['table'].str.strip().str.casefold() == 'usage']
usage_df['amount'] = usage_df['amount'].astype(int)

def usage_unit_factor(row):
    u = row['unit'].strip().casefold()
    if u == 'kwh':
        return 1000
    elif u == 'wh':
        return 1
    elif u == 'liter':
        return 1000
    elif u == 'ml':
        return 1
    elif u == 'hour':
        return 60
    elif u == 'minute':
        return 1
    else:
        raise ValueError(f'Unknown usage unit: {u}')
usage_df['amount_base'] = usage_df['amount'] * usage_df.apply(usage_unit_factor, axis=1)
Resources = sorted(usage_df['resource'].unique())
usage_dict = {}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    if i in Items:
        usage_dict[i, r] = usage_dict.get((i, r), 0) + row['amount_base']
capacity_df = capacity_df[capacity_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_df['amount'] = capacity_df['amount'].astype(int)

def capacity_unit_factor(row):
    u = row['unit'].strip().casefold()
    if u == 'kwh':
        return 1000
    elif u == 'wh':
        return 1
    elif u == 'liter':
        return 1000
    elif u == 'ml':
        return 1
    elif u == 'hour':
        return 60
    elif u == 'minute':
        return 1
    else:
        raise ValueError(f'Unknown capacity unit: {u}')
capacity_df['amount_base'] = capacity_df['amount'] * capacity_df.apply(capacity_unit_factor, axis=1)
total_capacity = capacity_df.groupby('resource')['amount_base'].sum().to_dict()
incompatible_df = incompatible_df[incompatible_df['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = []
for _, row in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in Items and b in Items:
        incompat_pairs.append((a, b))
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = []
for _, row in requires_df.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    if i in Items and prereq in Items:
        prereq_pairs.append((i, prereq))
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_tuples = []
bundle_bonus = {}
for _, row in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in Items and b in Items:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
category_items = {c: [] for c in Categories}
for i in Items:
    c = item_to_category[i]
    category_items[c].append(i)
m = gp.Model('BakeryOrderNetBenefit')
q_vars = m.addVars(Items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(Items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(Categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in Items:
    m.addConstr(q_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(q_vars[i] <= maximum_order[i] * y_vars[i])
for c in Categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) <= maximum_quantity[c])
for c in Categories:
    for i in category_items[c]:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in category_items[c])))
for r in Resources:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * q_vars[i] for i in Items)) <= total_capacity[r])
for i, j in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for i, prereq in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[prereq])
for a, b in bundle_tuples:
    m.addConstr(b_vars[a, b] <= y_vars[a])
    m.addConstr(b_vars[a, b] <= y_vars[b])
    m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
item_benefit = gp.quicksum((unit_benefit_cents[i] * q_vars[i] - item_fee_cents[i] * y_vars[i] for i in Items))
category_fees = gp.quicksum((activation_fee_cents[c] * z_vars[c] for c in Categories))
bundle_bonuses = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for a, b in bundle_tuples))
m.setObjective(item_benefit - category_fees + bundle_bonuses, gp.GRB.MAXIMIZE)
m.optimize()