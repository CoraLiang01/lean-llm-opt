import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def convert_to_base_unit(amount, unit):
    u = unit.strip().casefold()
    if u == 'liter':
        return amount * 1000
    elif u == 'ml':
        return amount
    elif u == 'kwh':
        return amount * 1000
    elif u == 'wh':
        return amount
    elif u == 'hour':
        return amount * 60
    elif u == 'minute':
        return amount
    else:
        raise ValueError(f'Unknown unit: {unit}')
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
usage_resources = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage']['resource'].unique()
capacity_resources = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger']['resource'].unique()
resource_ids = sorted(set(usage_resources).union(set(capacity_resources)))
incompat_df = df_incompat[df_incompat['table'].str.strip().str.casefold() == 'incompatible'].copy()
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompat_df.iterrows()]
requires_df = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires'].copy()
prereq_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
bundle_df = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle'].copy()
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]

def to_int_series(df, col):
    return df[col].astype(int)
authorized = items_df.set_index('item_ref')['authorized'].astype(int).to_dict()
minimum_lot = items_df.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = items_df.set_index('item_ref')['maximum_order'].astype(int).to_dict()
unit_benefit_cents = items_df.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = items_df.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
item_category = items_df.set_index('item_ref')['category'].to_dict()
minimum_quantity = categories_df.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = categories_df.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = categories_df.set_index('category')['activation_fee_cents'].astype(int).to_dict()
bundle_bonus_cents = {}
for (_, row) in bundle_df.iterrows():
    (i, j) = (row['item_a'], row['item_b'])
    bundle_bonus_cents[i, j] = int(row['bonus_cents'])
usage_df = df_usage[df_usage['table'].str.strip().str.casefold() == 'usage'].copy()
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit']
    amt_base = convert_to_base_unit(amt, unit)
    usage_dict[i, r] = amt_base
capacity_df = df_capacity[df_capacity['table'].str.strip().str.casefold() == 'capacity_ledger'].copy()
capacity_sum = {}
for r in resource_ids:
    rows = capacity_df[capacity_df['resource'] == r]
    total = 0
    for (_, row) in rows.iterrows():
        amt = int(row['amount'])
        unit = row['unit']
        amt_base = convert_to_base_unit(amt, unit)
        total += amt_base
    capacity_sum[r] = total
m = gp.Model('central_fresh_produce_order')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
b_vars = {}
for (i, j) in bundle_pairs:
    if i in item_ids and j in item_ids:
        b_vars[i, j] = m.addVar(vtype=gp.GRB.BINARY, name=f'b_{i}_{j}')
for i in item_ids:
    if authorized[i] == 1:
        m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
    else:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for g in category_ids:
    items_in_g = [i for i in item_ids if item_category[i] == g]
    if items_in_g:
        for i in items_in_g:
            m.addConstr(z_vars[g] >= y_vars[i])
        m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= minimum_quantity[g])
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= maximum_quantity[g])
    else:
        m.addConstr(z_vars[g] == 0)
for r in resource_ids:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x_vars[i] for i in item_ids)) <= capacity_sum[r])
for (i, j) in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, p) in prereq_pairs:
    if i in item_ids and p in item_ids:
        m.addConstr(y_vars[i] <= y_vars[p])
for (i, j) in bundle_pairs:
    if (i, j) in b_vars:
        if authorized.get(i, 0) == 1 and authorized.get(j, 0) == 1:
            m.addConstr(b_vars[i, j] <= y_vars[i])
            m.addConstr(b_vars[i, j] <= y_vars[j])
            m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
        else:
            m.addConstr(b_vars[i, j] == 0)
benefit = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in item_ids))
item_fees = gp.quicksum((item_fee_cents[i] * y_vars[i] for i in item_ids))
category_fees = gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in category_ids))
bundle_bonuses = gp.quicksum((bundle_bonus_cents[i, j] * b_vars[i, j] for (i, j) in b_vars))
m.setObjective(benefit - item_fees - category_fees + bundle_bonuses, gp.GRB.MAXIMIZE)
m.optimize()