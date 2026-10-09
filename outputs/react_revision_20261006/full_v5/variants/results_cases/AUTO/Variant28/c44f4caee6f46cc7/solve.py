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
    items = df_item[df_item['authorized'] == 1]['item_ref'].astype(str).unique()
    items_set = set(items)
    categories = df_category['category'].astype(str).unique()
    categories_set = set(categories)
    resources = pd.concat([df_usage['resource'].astype(str), df_capacity['resource'].astype(str)]).unique()
    resources_set = set(resources)
    bundles = df_bundle[['item_a', 'item_b']].astype(str).apply(tuple, axis=1).tolist()
    bundle_keys = [tuple(row) for row in df_bundle[['item_a', 'item_b']].astype(str).values]
    incompat_pairs = df_incompat[['item_a', 'item_b']].astype(str).apply(tuple, axis=1).tolist()
    prereq_pairs = df_requires[['item_ref', 'prerequisite_ref']].astype(str).apply(tuple, axis=1).tolist()
    item_param = df_item.set_index('item_ref').loc[items]
    item_category = item_param['category'].astype(str).to_dict()
    min_lot = item_param['minimum_lot'].to_dict()
    max_order = item_param['maximum_order'].to_dict()
    unit_benefit = item_param['unit_benefit_cents'].to_dict()
    item_fee = item_param['item_fee_cents'].to_dict()
    cat_param = df_category.set_index('category')
    min_cat_qty = cat_param['minimum_quantity'].to_dict()
    max_cat_qty = cat_param['maximum_quantity'].to_dict()
    cat_fee = cat_param['activation_fee_cents'].to_dict()
    bundle_bonus = {}
    for (idx, row) in df_bundle.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        bundle_bonus[a, b] = int(row['bonus_cents'])
    unit_to_base = {'ml': ('ml', 1), 'liter': ('ml', 1000), 'wh': ('wh', 1), 'kwh': ('wh', 1000), 'minute': ('minute', 1), 'hour': ('minute', 60)}
    usage = {}
    for (idx, row) in df_usage.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        amt = float(row['amount'])
        unit = str(row['unit']).casefold().strip()
        if unit not in unit_to_base:
            raise ValueError(f'Unknown unit {unit} in usage table')
        (base_unit, factor) = unit_to_base[unit]
        amt_base = amt * factor
        if (i, r) in usage:
            usage[i, r] += amt_base
        else:
            usage[i, r] = amt_base
    capacity = {}
    for r in resources:
        df_r = df_capacity[df_capacity['resource'].astype(str) == r]
        total = 0.0
        for (idx, row) in df_r.iterrows():
            amt = float(row['amount'])
            unit = str(row['unit']).casefold().strip()
            if unit not in unit_to_base:
                raise ValueError(f'Unknown unit {unit} in capacity_ledger')
            (base_unit, factor) = unit_to_base[unit]
            amt_base = amt * factor
            total += amt_base
        capacity[r] = total
    m = gp.Model('central_fresh_order')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, ub={i: max_order[i] for i in items}, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
    for i in items:
        m.addConstr(x[i] >= min_lot[i] * y[i], name='minlot_' + i)
        m.addConstr(x[i] <= max_order[i] * y[i], name='maxorder_' + i)
    for c in categories:
        items_in_c = [i for i in items if item_category[i] == c]
        for i in items_in_c:
            m.addConstr(z[c] >= y[i], name='catact1_' + c + '_' + i)
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= min_cat_qty[c] * z[c], name='catmin_' + c)
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= max_cat_qty[c] * z[c], name='catmax_' + c)
    for r in resources:
        expr = gp.LinExpr()
        for i in items:
            if (i, r) in usage:
                expr += usage[i, r] * x[i]
        m.addConstr(expr <= capacity[r], name='res_' + r)
    for (i, j) in incompat_pairs:
        if i in items_set and j in items_set:
            m.addConstr(y[i] + y[j] <= 1, name='incompat_' + i + '_' + j)
    for (i, j) in prereq_pairs:
        if i in items_set and j in items_set:
            m.addConstr(y[i] <= y[j], name='prereq_' + i + '_' + j)
    for (a, b_) in bundle_keys:
        if a in items_set and b_ in items_set:
            m.addConstr(b[a, b_] <= y[a], name='bundle1_' + a + '_' + b_)
            m.addConstr(b[a, b_] <= y[b_], name='bundle2_' + a + '_' + b_)
            m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name='bundle3_' + a + '_' + b_)
        else:
            m.addConstr(b[a, b_] == 0, name='bundle_forbidden_' + a + '_' + b_)
    obj = gp.LinExpr()
    obj += gp.quicksum((unit_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
    obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[a, b_] * b[a, b_] for (a, b_) in bundle_keys))
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