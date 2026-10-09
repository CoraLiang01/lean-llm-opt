import gurobipy as gp
import pandas as pd
import numpy as np
import re

def convert_to_base_unit(amount, from_unit, to_unit):
    u = from_unit.strip().casefold()
    t = to_unit.strip().casefold()
    if u == t:
        return amount
    if u == 'liter' and t == 'ml':
        return amount * 1000
    if u == 'ml' and t == 'liter':
        return amount / 1000
    if u == 'hour' and t == 'minute':
        return amount * 60
    if u == 'minute' and t == 'hour':
        return amount / 60
    if u == 'kwh' and t == 'wh':
        return amount * 1000
    if u == 'wh' and t == 'kwh':
        return amount / 1000
    raise ValueError(f'Unknown unit conversion: {u} -> {t}')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
items_df = df_item[df_item['table'].str.strip().str.casefold() == 'item'].copy()
item_ids = items_df['item_ref'].tolist()
categories_df = df_category[df_category['table'].str.strip().str.casefold() == 'category'].copy()
category_ids = categories_df['category'].tolist()
item_to_category = items_df.set_index('item_ref')['category'].to_dict()
usage_resources = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']['resource'].unique()
capacity_resources = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']['resource'].unique()
resource_ids = sorted(set([r.strip() for r in usage_resources]).union([r.strip() for r in capacity_resources]))
incompat_df = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_df.iterrows()]
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
bundle_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
bundle_bonus = {(row['item_a'], row['item_b']): int(row['bonus_cents']) for (_, row) in bundle_df.iterrows()}
items_df['authorized'] = items_df['authorized'].astype(int)
authorized_items = [i for i in item_ids if items_df.loc[items_df['item_ref'] == i, 'authorized'].iloc[0] == 1]
unauthorized_items = [i for i in item_ids if i not in authorized_items]
items_df['minimum_lot'] = items_df['minimum_lot'].astype(int)
items_df['maximum_order'] = items_df['maximum_order'].astype(int)
items_df['unit_benefit_cents'] = items_df['unit_benefit_cents'].astype(int)
items_df['item_fee_cents'] = items_df['item_fee_cents'].astype(int)
minimum_lot = items_df.set_index('item_ref')['minimum_lot'].to_dict()
maximum_order = items_df.set_index('item_ref')['maximum_order'].to_dict()
unit_benefit_cents = items_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = items_df.set_index('item_ref')['item_fee_cents'].to_dict()
categories_df['minimum_quantity'] = categories_df['minimum_quantity'].astype(int)
categories_df['maximum_quantity'] = categories_df['maximum_quantity'].astype(int)
categories_df['activation_fee_cents'] = categories_df['activation_fee_cents'].astype(int)
minimum_quantity = categories_df.set_index('category')['minimum_quantity'].to_dict()
maximum_quantity = categories_df.set_index('category')['maximum_quantity'].to_dict()
activation_fee_cents = categories_df.set_index('category')['activation_fee_cents'].to_dict()
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage'].copy()
usage_dict = {}
for (_, row) in usage_df.iterrows():
    usage_dict[row['item_ref'], row['resource']] = (int(row['amount']), row['unit'])
capacity_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger'].copy()
resource_capacity = {}
resource_base_unit = {}
for r in resource_ids:
    entries = capacity_df[capacity_df['resource'].str.strip() == r]
    units = entries['unit'].unique()
    base_unit = units[0].strip()
    total = 0.0
    for (_, row) in entries.iterrows():
        amt = int(row['amount'])
        unit = row['unit']
        amt_in_base = convert_to_base_unit(amt, unit, base_unit)
        total += amt_in_base
    resource_capacity[r] = total
    resource_base_unit[r] = base_unit
m = gp.Model('central_fresh_order')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if i in authorized_items:
        m.addConstr(x_vars[i] >= 0)
        m.addConstr(x_vars[i] <= maximum_order[i])
        m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for g in category_ids:
    items_in_g = [i for i in item_ids if item_to_category[i] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= minimum_quantity[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= maximum_quantity[g])
for g in category_ids:
    items_in_g = [i for i in item_ids if item_to_category[i] == g]
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for r in resource_ids:
    expr = []
    for i in item_ids:
        if (i, r) in usage_dict:
            (amt, unit) = usage_dict[i, r]
            amt_in_base = convert_to_base_unit(amt, unit, resource_base_unit[r])
            expr.append(x_vars[i] * amt_in_base)
    m.addConstr(gp.quicksum(expr) <= resource_capacity[r])
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, p) in prereq_pairs:
    if i in item_ids and p in item_ids:
        m.addConstr(y_vars[i] <= y_vars[p])
for (i, j) in bundle_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(w_vars[i, j] <= y_vars[i])
        m.addConstr(w_vars[i, j] <= y_vars[j])
        m.addConstr(w_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(w_vars[i, j] == 0)
item_obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] - item_fee_cents[i] * y_vars[i] for i in item_ids))
bundle_obj = gp.quicksum((bundle_bonus[i, j] * w_vars[i, j] for (i, j) in bundle_pairs))
cat_obj = gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in category_ids))
m.setObjective(item_obj + bundle_obj - cat_obj, gp.GRB.MAXIMIZE)
m.optimize()