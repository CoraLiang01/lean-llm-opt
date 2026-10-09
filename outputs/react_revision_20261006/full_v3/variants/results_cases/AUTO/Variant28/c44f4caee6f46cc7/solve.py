import gurobipy as gp
import pandas as pd
import numpy as np
import math

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', sep=',')
    df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', sep=',')
    df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', sep=',')
    df_id = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', sep=',')
    df_incomp = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', sep=',')
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', sep=',')
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_07.csv', sep=',')
    df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', sep=',')
    df_item_auth = df_item[df_item['authorized'] == 1].copy()
    items = list(df_item_auth['item_ref'])
    categories = list(df_cat['category'])
    resources = sorted(set(df_usage['resource']).union(df_cap['resource']))
    bundle_rows = []
    for (idx, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items and b in items:
            bundle_rows.append((a, b, row['bonus_cents']))
    bundles = [(a, b) for (a, b, _) in bundle_rows]
    bundle_bonus = {(a, b): bonus for (a, b, bonus) in bundle_rows}
    incomp_pairs = []
    for (idx, row) in df_incomp.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items and b in items:
            incomp_pairs.append((a, b))
    req_pairs = []
    for (idx, row) in df_req.iterrows():
        i = row['item_ref']
        j = row['prerequisite_ref']
        if i in items and j in items:
            req_pairs.append((i, j))
    min_lot = df_item_auth.set_index('item_ref')['minimum_lot'].to_dict()
    max_order = df_item_auth.set_index('item_ref')['maximum_order'].to_dict()
    unit_benefit = df_item_auth.set_index('item_ref')['unit_benefit_cents'].to_dict()
    item_fee = df_item_auth.set_index('item_ref')['item_fee_cents'].to_dict()
    item_cat = df_item_auth.set_index('item_ref')['category'].to_dict()
    cat_min = df_cat.set_index('category')['minimum_quantity'].to_dict()
    cat_max = df_cat.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_cat.set_index('category')['activation_fee_cents'].to_dict()
    cat_items = {c: [] for c in categories}
    for i in items:
        c = item_cat[i]
        cat_items[c].append(i)
    unit_conv = {('kwh', 'wh'): 1000, ('wh', 'wh'): 1, ('hour', 'minute'): 60, ('minute', 'minute'): 1, ('liter', 'ml'): 1000, ('ml', 'ml'): 1}
    usage = {(i, r): 0 for i in items for r in resources}
    for (idx, row) in df_usage.iterrows():
        i = row['item_ref']
        r = row['resource']
        if i not in items:
            continue
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if r == 'power':
            base_unit = 'wh'
        elif r == 'labor':
            base_unit = 'minute'
        elif r == 'space':
            base_unit = 'ml'
        else:
            raise ValueError(f'Unknown resource {r}')
        if (u, base_unit) in unit_conv:
            factor = unit_conv[u, base_unit]
        else:
            raise ValueError(f'Unknown unit conversion: {u} to {base_unit}')
        usage[i, r] = amt * factor
    cap = {r: 0 for r in resources}
    for (idx, row) in df_cap.iterrows():
        r = row['resource']
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if r == 'power':
            base_unit = 'wh'
        elif r == 'labor':
            base_unit = 'minute'
        elif r == 'space':
            base_unit = 'ml'
        else:
            raise ValueError(f'Unknown resource {r}')
        if (u, base_unit) in unit_conv:
            factor = unit_conv[u, base_unit]
        else:
            raise ValueError(f'Unknown unit conversion: {u} to {base_unit}')
        cap[r] += amt * factor
    m = gp.Model('central_fresh_order')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, ub=[max_order[i] for i in items], vtype=gp.GRB.INTEGER, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        m.addConstr(x[i] >= min_lot[i] * z[i], name=f'item_min_{i}')
        m.addConstr(x[i] <= max_order[i] * z[i], name=f'item_max_{i}')
        m.addConstr(x[i] >= 0, name=f'item_nonneg_{i}')
    for c in categories:
        for i in cat_items[c]:
            m.addConstr(w[c] >= z[i], name=f'cat_act_{c}_{i}')
        m.addConstr(w[c] <= gp.quicksum((z[i] for i in cat_items[c])), name=f'cat_act_sum_{c}')
    for c in categories:
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_min[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_max[c], name=f'cat_max_{c}')
    for r in resources:
        m.addConstr(gp.quicksum((usage[i, r] * x[i] for i in items)) <= cap[r], name=f'res_cap_{r}')
    for (i, j) in incomp_pairs:
        m.addConstr(z[i] + z[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, j) in req_pairs:
        m.addConstr(z[i] <= z[j], name=f'req_{i}_{j}')
    for (i, j) in bundles:
        m.addConstr(b[i, j] <= z[i], name=f'bundle1_{i}_{j}')
        m.addConstr(b[i, j] <= z[j], name=f'bundle2_{i}_{j}')
        m.addConstr(b[i, j] >= z[i] + z[j] - 1, name=f'bundle3_{i}_{j}')
    obj = gp.LinExpr()
    obj += gp.quicksum((unit_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * z[i] for i in items))
    obj -= gp.quicksum((cat_fee[c] * w[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')