import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv']
df_bundle = pd.read_csv(paths[0], dtype=str, keep_default_na=False)
df_capacity = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
df_category = pd.read_csv(paths[2], dtype=str, keep_default_na=False)
df_identity = pd.read_csv(paths[3], dtype=str, keep_default_na=False)
df_incompat = pd.read_csv(paths[4], dtype=str, keep_default_na=False)
df_item = pd.read_csv(paths[5], dtype=str, keep_default_na=False)
df_market = pd.read_csv(paths[6], dtype=str, keep_default_na=False)
df_requires = pd.read_csv(paths[7], dtype=str, keep_default_na=False)
df_usage = pd.read_csv(paths[8], dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
items = df_item[df_item['authorized'] == 1]['item_ref'].tolist()
categories = df_category['category'].tolist()
resources_usage = df_usage['resource'].unique().tolist()
resources_capacity = df_capacity['resource'].unique().tolist()
resources = sorted(set(resources_usage) | set(resources_capacity))
bundles = list(df_bundle.itertuples(index=False, name=None))
incompat_pairs = list(df_incompat.itertuples(index=False, name=None))
prereq_pairs = list(df_requires.itertuples(index=False, name=None))
item_params = df_item.set_index('item_ref').to_dict(orient='index')
for i in items:
    for field in ['minimum_lot', 'maximum_order', 'unit_benefit_cents', 'item_fee_cents']:
        item_params[i][field] = int(item_params[i][field])
    item_params[i]['category'] = item_params[i]['category']
cat_params = df_category.set_index('category').to_dict(orient='index')
for g in categories:
    for field in ['minimum_quantity', 'maximum_quantity', 'activation_fee_cents']:
        cat_params[g][field] = int(cat_params[g][field])
bundle_list = []
bundle_bonus = {}
for row in df_bundle.itertuples(index=False):
    b = (row[1], row[2])
    bundle_list.append(b)
    bundle_bonus[b] = int(row[3])
incompat_set = set()
for row in incompat_pairs:
    incompat_set.add(frozenset([row[1], row[2]]))
prereq_list = []
for row in prereq_pairs:
    prereq_list.append((row[1], row[2]))
usage_dict = {}
for row in df_usage.itertuples(index=False):
    i = row[1]
    r = row[2]
    amt = int(row[3])
    unit = row[4].strip().lower()
    usage_dict[i, r] = (amt, unit)
capacity_dict = {}
unit_dict = {}
for r in resources:
    df_r = df_capacity[df_capacity['resource'] == r]
    total = 0
    base_unit = None
    for row in df_r.itertuples(index=False):
        amt = int(row[3])
        unit = row[4].strip().lower()
        if base_unit is None:
            base_unit = unit
        total += amt
    capacity_dict[r] = total
    unit_dict[r] = base_unit

def to_base_unit(resource, amount, unit):
    unit = unit.strip().lower()
    if resource == 'space':
        if unit == 'ml':
            return amount
        elif unit == 'liter':
            return amount * 1000
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'power':
        if unit == 'wh':
            return amount
        elif unit == 'kwh':
            return amount * 1000
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    elif resource == 'labor':
        if unit == 'minute':
            return amount
        elif unit == 'hour':
            return amount * 60
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')

def build_model():
    m = gp.Model('central_fresh_produce_order')
    x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w_vars = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
    for i in items:
        min_lot = item_params[i]['minimum_lot']
        max_order = item_params[i]['maximum_order']
        m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'x_lb_{i}')
        m.addConstr(x_vars[i] <= max_order * y_vars[i], name=f'x_ub_{i}')
    for g in categories:
        items_in_g = [i for i in items if item_params[i]['category'] == g]
        for i in items_in_g:
            m.addConstr(z_vars[g] >= y_vars[i], name=f'z_{g}_covers_{i}')
        if not items_in_g:
            m.addConstr(z_vars[g] == 0, name=f'z_{g}_no_items')
    for g in categories:
        items_in_g = [i for i in items if item_params[i]['category'] == g]
        min_q = cat_params[g]['minimum_quantity']
        max_q = cat_params[g]['maximum_quantity']
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= min_q, name=f'cat_min_{g}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= max_q, name=f'cat_max_{g}')
    for r in resources:
        usage_sum = []
        for i in items:
            if (i, r) in usage_dict:
                (amt, unit) = usage_dict[i, r]
                amt_base = to_base_unit(r, amt, unit)
                usage_sum.append(amt_base * x_vars[i])
        m.addConstr(gp.quicksum(usage_sum) <= capacity_dict[r], name=f'res_cap_{r}')
    for pair in incompat_set:
        pair_list = list(pair)
        i_list = [i for i in pair_list if i in items]
        if len(i_list) == 2:
            m.addConstr(y_vars[i_list[0]] + y_vars[i_list[1]] <= 1, name=f'incompat_{i_list[0]}_{i_list[1]}')
    for (i, j) in prereq_list:
        if i in items and j in items:
            m.addConstr(y_vars[i] <= y_vars[j], name=f'prereq_{i}_req_{j}')
    for b in bundle_list:
        (i, j) = b
        if i in items and j in items:
            m.addConstr(w_vars[b] <= y_vars[i], name=f'bundle_{b}_i')
            m.addConstr(w_vars[b] <= y_vars[j], name=f'bundle_{b}_j')
            m.addConstr(w_vars[b] >= y_vars[i] + y_vars[j] - 1, name=f'bundle_{b}_both')
        else:
            m.addConstr(w_vars[b] == 0, name=f'bundle_{b}_unauth')
    item_benefit = gp.quicksum((item_params[i]['unit_benefit_cents'] * x_vars[i] - item_params[i]['item_fee_cents'] * y_vars[i] for i in items))
    bundle_benefit = gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_list))
    category_fees = gp.quicksum((cat_params[g]['activation_fee_cents'] * z_vars[g] for g in categories))
    m.setObjective(item_benefit + bundle_benefit - category_fees, gp.GRB.MAXIMIZE)
    return m
m = build_model()
m.optimize()