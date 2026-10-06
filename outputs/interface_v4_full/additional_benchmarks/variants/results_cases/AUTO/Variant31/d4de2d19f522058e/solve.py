import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv'
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
items = df_item['item_ref'].astype(str).tolist()
categories = df_category['category'].astype(str).tolist()
resources = sorted(set(df_usage['resource'].astype(str)).union(df_capacity['resource'].astype(str)))
bundles = [(str(row['item_a']), str(row['item_b'])) for _, row in df_bundle.iterrows()]
incompat_pairs = [(str(row['item_a']), str(row['item_b'])) for _, row in df_incompat.iterrows()]
requires_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for _, row in df_requires.iterrows()]
item_params = df_item.set_index('item_ref').astype({'authorized': int, 'minimum_lot': int, 'maximum_order': int, 'unit_benefit_cents': int, 'item_fee_cents': int, 'category': str})
authorized = item_params['authorized'].to_dict()
minimum_lot = item_params['minimum_lot'].to_dict()
maximum_order = item_params['maximum_order'].to_dict()
unit_benefit_cents = item_params['unit_benefit_cents'].to_dict()
item_fee_cents = item_params['item_fee_cents'].to_dict()
item_category = item_params['category'].to_dict()
cat_params = df_category.set_index('category').astype({'minimum_quantity': int, 'maximum_quantity': int, 'activation_fee_cents': int})
minimum_quantity = cat_params['minimum_quantity'].to_dict()
maximum_quantity = cat_params['maximum_quantity'].to_dict()
activation_fee_cents = cat_params['activation_fee_cents'].to_dict()
usage = {}
for _, row in df_usage.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage[i, r] = int(row['amount'])
df_capacity['amount'] = df_capacity['amount'].astype(int)
resource_capacity = {}
for r in resources:
    cap = df_capacity.loc[df_capacity['resource'].astype(str) == r, 'amount'].sum()
    resource_capacity[r] = cap
bundle_bonus = {(str(row['item_a']), str(row['item_b'])): int(row['bonus_cents']) for _, row in df_bundle.iterrows()}

def solve_problem():
    m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if authorized[i] == 0:
            m.addConstr(x[i] == 0, name=f'auth_{i}')
            m.addConstr(y[i] == 0, name=f'authy_{i}')
        else:
            m.addConstr(x[i] <= maximum_order[i] * y[i], name=f'link_ub_{i}')
            m.addConstr(x[i] >= minimum_lot[i] * y[i], name=f'link_lb_{i}')
            m.addConstr(x[i] <= maximum_order[i], name=f'maxorder_{i}')
            m.addConstr(x[i] >= 0, name=f'minzero_{i}')
    items_in_cat = {g: [i for i in items if item_category[i] == g] for g in categories}
    M_cat = max(maximum_order.values()) * len(items) + 1
    for g in categories:
        for i in items_in_cat[g]:
            m.addConstr(x[i] <= M_cat * z[g], name=f'cat_link_{i}_{g}')
            m.addConstr(z[g] >= y[i], name=f'cat_zg_ge_y_{i}_{g}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_cat[g])) >= minimum_quantity[g], name=f'cat_min_{g}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_cat[g])) <= maximum_quantity[g], name=f'cat_max_{g}')
    for r in resources:
        m.addConstr(gp.quicksum((usage.get((i, r), 0) * x[i] for i in items)) <= resource_capacity[r], name=f'res_{r}')
    for a, b_ in incompat_pairs:
        if a in items and b_ in items:
            m.addConstr(y[a] + y[b_] <= 1, name=f'incompat_{a}_{b_}')
    for i, prereq in requires_pairs:
        if i in items and prereq in items:
            m.addConstr(y[i] <= y[prereq], name=f'requires_{i}_{prereq}')
    for a, b_ in bundles:
        if a in items and b_ in items:
            m.addConstr(b[a, b_] <= y[a], name=f'bundle_b_le_ya_{a}_{b_}')
            m.addConstr(b[a, b_] <= y[b_], name=f'bundle_b_le_yb_{a}_{b_}')
            m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle_b_ge_sum_{a}_{b_}')
    obj = gp.LinExpr()
    obj += gp.quicksum((unit_benefit_cents[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee_cents[i] * y[i] for i in items))
    obj -= gp.quicksum((activation_fee_cents[g] * z[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[a, b_] * b[a, b_] for a, b_ in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()