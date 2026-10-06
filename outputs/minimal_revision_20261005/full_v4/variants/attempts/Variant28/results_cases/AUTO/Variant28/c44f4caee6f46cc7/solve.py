import gurobipy as gp
import pandas as pd
import numpy as np
import math

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', sep=',')
    df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', sep=',')
    df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', sep=',')
    df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', sep=',')
    df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', sep=',')
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', sep=',')
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_07.csv', sep=',')
    df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', sep=',')
    df_item_auth = df_item[df_item['authorized'] == 1].copy()
    items = list(df_item_auth['item_ref'])
    items_set = set(items)
    categories = list(df_category['category'])
    categories_set = set(categories)
    resources_usage = set(df_usage['resource'].unique())
    resources_capacity = set(df_capacity['resource'].unique())
    resources = sorted(resources_usage | resources_capacity)
    resources_set = set(resources)
    incompat_pairs = []
    for (_, row) in df_incompat.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items_set and b in items_set:
            incompat_pairs.append((a, b))
    prereq_pairs = []
    for (_, row) in df_requires.iterrows():
        i = row['item_ref']
        p = row['prerequisite_ref']
        if i in items_set and p in items_set:
            prereq_pairs.append((i, p))
    bundle_rows = []
    for (idx, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items_set and b in items_set:
            bundle_rows.append((a, b, row['bonus_cents']))
    bundles = [(a, b) for (a, b, _) in bundle_rows]
    bundle_bonus = {(a, b): bonus for (a, b, bonus) in bundle_rows}
    item_minlot = df_item_auth.set_index('item_ref')['minimum_lot'].to_dict()
    item_maxorder = df_item_auth.set_index('item_ref')['maximum_order'].to_dict()
    item_benefit = df_item_auth.set_index('item_ref')['unit_benefit_cents'].to_dict()
    item_fee = df_item_auth.set_index('item_ref')['item_fee_cents'].to_dict()
    item_category = df_item_auth.set_index('item_ref')['category'].to_dict()
    cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
    cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
    unit_factor = {'liter': 1000, 'ml': 1, 'hour': 60, 'minute': 1, 'kwh': 1000, 'wh': 1}
    usage_dict = {}
    for (_, row) in df_usage.iterrows():
        i = row['item_ref']
        if i not in items_set:
            continue
        r = row['resource']
        amt = row['amount']
        unit = row['unit'].strip().casefold()
        if unit not in unit_factor:
            raise ValueError(f'Unknown unit {unit} for resource usage')
        amt_base = amt * unit_factor[unit]
        usage_dict[i, r] = usage_dict.get((i, r), 0) + amt_base
    capacity_dict = {}
    for (_, row) in df_capacity.iterrows():
        r = row['resource']
        amt = row['amount']
        unit = row['unit'].strip().casefold()
        if unit not in unit_factor:
            raise ValueError(f'Unknown unit {unit} for resource capacity')
        amt_base = amt * unit_factor[unit]
        capacity_dict[r] = capacity_dict.get(r, 0) + amt_base
    m = gp.Model('central_fresh_order')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'item_minlot_{i}')
        m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'item_maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'item_nonneg_{i}')
    cat2items = {c: [i for i in items if item_category[i] == c] for c in categories}
    for c in categories:
        items_in_c = cat2items[c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_minqty[c] * z[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_maxqty[c] * z[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
    for r in resources:
        expr = gp.LinExpr()
        for i in items:
            amt = usage_dict.get((i, r), 0)
            expr += amt * x[i]
        cap = capacity_dict.get(r, 0)
        m.addConstr(expr <= cap, name=f'res_cap_{r}')
    for (i, j) in incompat_pairs:
        m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
    for (i, p) in prereq_pairs:
        m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
    for (a, b_) in bundles:
        m.addConstr(b[a, b_] <= y[a], name=f'bundle1_{a}_{b_}')
        m.addConstr(b[a, b_] <= y[b_], name=f'bundle2_{a}_{b_}')
        m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle3_{a}_{b_}')
    obj = gp.LinExpr()
    obj += gp.quicksum((item_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
    obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[a, b_] * b[a, b_] for (a, b_) in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')