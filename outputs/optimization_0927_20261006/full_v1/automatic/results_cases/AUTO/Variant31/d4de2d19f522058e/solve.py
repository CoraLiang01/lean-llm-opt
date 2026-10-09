import gurobipy as gp
import pandas as pd
import numpy as np
import re
path_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv'
path_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv'
path_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv'
path_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv'
path_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv'
path_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv'
path_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv'
path_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv'
path_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv'
df_bundle = pd.read_csv(path_bundle, dtype=str, keep_default_na=False)
df_capacity_ledger = pd.read_csv(path_capacity_ledger, dtype=str, keep_default_na=False)
df_category = pd.read_csv(path_category, dtype=str, keep_default_na=False)
df_identity = pd.read_csv(path_identity, dtype=str, keep_default_na=False)
df_incompatible = pd.read_csv(path_incompatible, dtype=str, keep_default_na=False)
df_item = pd.read_csv(path_item, dtype=str, keep_default_na=False)
df_market = pd.read_csv(path_market, dtype=str, keep_default_na=False)
df_requires = pd.read_csv(path_requires, dtype=str, keep_default_na=False)
df_usage = pd.read_csv(path_usage, dtype=str, keep_default_na=False)
item_ids = df_item['item_ref'].tolist()
category_ids = df_category['category'].tolist()
resource_ids = sorted(set(df_capacity_ledger['resource'].tolist()) | set(df_usage['resource'].tolist()))
bundle_tuples = []
for (_, row) in df_bundle.iterrows():
    bundle_tuples.append((row['item_a'], row['item_b']))
incompatible_pairs = []
for (_, row) in df_incompatible.iterrows():
    incompatible_pairs.append((row['item_a'], row['item_b']))
requires_pairs = []
for (_, row) in df_requires.iterrows():
    requires_pairs.append((row['item_ref'], row['prerequisite_ref']))
df_item['authorized'] = df_item['authorized'].astype(int)
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
item_to_category = dict(zip(df_item['item_ref'], df_item['category']))
authorized = dict(zip(df_item['item_ref'], df_item['authorized']))
minimum_lot = dict(zip(df_item['item_ref'], df_item['minimum_lot']))
maximum_order = dict(zip(df_item['item_ref'], df_item['maximum_order']))
unit_benefit_cents = dict(zip(df_item['item_ref'], df_item['unit_benefit_cents']))
item_fee_cents = dict(zip(df_item['item_ref'], df_item['item_fee_cents']))
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
minimum_quantity = dict(zip(df_category['category'], df_category['minimum_quantity']))
maximum_quantity = dict(zip(df_category['category'], df_category['maximum_quantity']))
activation_fee_cents = dict(zip(df_category['category'], df_category['activation_fee_cents']))
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_bonus_cents = dict()
for (_, row) in df_bundle.iterrows():
    bundle_bonus_cents[row['item_a'], row['item_b']] = row['bonus_cents']
usage_amount = dict()
for (_, row) in df_usage.iterrows():
    usage_amount[row['item_ref'], row['resource']] = int(row['amount'])
resource_total = dict()
for r in resource_ids:
    df_r = df_capacity_ledger[df_capacity_ledger['resource'] == r]
    total = df_r['amount'].astype(int).sum()
    resource_total[r] = total
category_to_items = {g: [] for g in category_ids}
for i in item_ids:
    g = item_to_category[i]
    category_to_items[g].append(i)

def build_model():
    m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
    x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
    z_vars = m.addVars(category_ids, vtype=gp.GRB.BINARY, name='')
    w_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
    for r in resource_ids:
        m.addConstr(gp.quicksum((usage_amount.get((i, r), 0) * x_vars[i] for i in item_ids)) <= resource_total[r], name=f'resource_{r}')
    for i in item_ids:
        if authorized[i] == 0:
            m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
            m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
        else:
            m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i], name=f'minlot_{i}')
            m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i], name=f'maxorder_{i}')
    for g in category_ids:
        items_in_g = category_to_items[g]
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= minimum_quantity[g], name=f'cat_min_{g}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= maximum_quantity[g], name=f'cat_max_{g}')
    for g in category_ids:
        items_in_g = category_to_items[g]
        for i in items_in_g:
            m.addConstr(z_vars[g] >= y_vars[i], name=f'catact_{g}_{i}')
    for (a, b) in bundle_tuples:
        m.addConstr(w_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(w_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
    for (i, j) in incompatible_pairs:
        if i in item_ids and j in item_ids:
            m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, j) in requires_pairs:
        if i in item_ids and j in item_ids:
            m.addConstr(y_vars[i] <= y_vars[j], name=f'requires_{i}_{j}')
    obj_benefit = gp.quicksum((unit_benefit_cents[i] * x_vars[i] for i in item_ids))
    obj_item_fees = gp.quicksum((item_fee_cents[i] * y_vars[i] for i in item_ids))
    obj_cat_fees = gp.quicksum((activation_fee_cents[g] * z_vars[g] for g in category_ids))
    obj_bundle_bonus = gp.quicksum((bundle_bonus_cents[a, b] * w_vars[a, b] for (a, b) in bundle_tuples))
    m.setObjective(obj_benefit - obj_item_fees - obj_cat_fees + obj_bundle_bonus, gp.GRB.MAXIMIZE)
    return m
m = build_model()
m.optimize()