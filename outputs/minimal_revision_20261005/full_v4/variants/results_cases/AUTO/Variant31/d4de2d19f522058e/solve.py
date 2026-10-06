import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', sep=',')
    df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', sep=',')
    df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', sep=',')
    df_id = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', sep=',')
    df_incomp = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', sep=',')
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', sep=',')
    df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', sep=',')
    items = df_item['item_ref'].astype(str).unique().tolist()
    categories = df_cat['category'].astype(str).unique().tolist()
    resources = sorted(set(df_cap['resource'].astype(str)).union(df_usage['resource'].astype(str)))
    bundles = df_bundle.index.tolist()
    incomp_pairs = [(str(row['item_a']), str(row['item_b'])) for (_, row) in df_incomp.iterrows()]
    req_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for (_, row) in df_req.iterrows()]
    df_item['item_ref'] = df_item['item_ref'].astype(str)
    item_minlot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
    item_maxorder = df_item.set_index('item_ref')['maximum_order'].to_dict()
    item_authorized = df_item.set_index('item_ref')['authorized'].to_dict()
    item_unit_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
    item_fee = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
    item_category = df_item.set_index('item_ref')['category'].astype(str).to_dict()
    df_cat['category'] = df_cat['category'].astype(str)
    cat_minqty = df_cat.set_index('category')['minimum_quantity'].to_dict()
    cat_maxqty = df_cat.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_cat.set_index('category')['activation_fee_cents'].to_dict()
    bundle_item_a = df_bundle['item_a'].astype(str).tolist()
    bundle_item_b = df_bundle['item_b'].astype(str).tolist()
    bundle_bonus = df_bundle['bonus_cents'].tolist()
    df_cap['resource'] = df_cap['resource'].astype(str)
    cap_opening = df_cap[df_cap['entry'].str.casefold() == 'opening'].groupby('resource')['amount'].sum()
    cap_resv = df_cap[df_cap['entry'].str.casefold() == 'reservation'].groupby('resource')['amount'].sum()
    cap_total = (cap_opening + cap_resv).to_dict()
    for r in resources:
        if r not in cap_total:
            cap_total[r] = cap_opening.get(r, 0)
    df_usage['item_ref'] = df_usage['item_ref'].astype(str)
    df_usage['resource'] = df_usage['resource'].astype(str)
    usage = {(row['item_ref'], row['resource']): row['amount'] for (_, row) in df_usage.iterrows()}
    usage_full = {}
    for i in items:
        for r in resources:
            usage_full[i, r] = usage.get((i, r), 0)
    cat_items = {c: [] for c in categories}
    for i in items:
        c = item_category[i]
        if c in cat_items:
            cat_items[c].append(i)
        else:
            cat_items[c] = [i]
    m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        auth = int(item_authorized[i])
        minlot = int(item_minlot[i])
        maxorder = int(item_maxorder[i])
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'auth0_x_{i}')
            m.addConstr(z[i] == 0, name=f'auth0_z_{i}')
        else:
            m.addConstr(x[i] >= minlot * z[i], name=f'lot_lb_{i}')
            m.addConstr(x[i] <= maxorder * z[i], name=f'lot_ub_{i}')
    for r in resources:
        m.addConstr(gp.quicksum((usage_full[i, r] * x[i] for i in items)) <= cap_total[r], name=f'res_{r}')
    for c in categories:
        items_in_c = cat_items.get(c, [])
        if not items_in_c:
            m.addConstr(w[c] == 0, name=f'cat_empty_{c}')
            continue
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= int(cat_minqty[c]) * w[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= int(cat_maxqty[c]) * w[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(z[i] <= w[c], name=f'cat_link_{c}_{i}')
    for (i, j) in incomp_pairs:
        if i in items and j in items:
            m.addConstr(z[i] + z[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, prereq) in req_pairs:
        if i in items and prereq in items:
            m.addConstr(z[i] <= z[prereq], name=f'req_{i}_{prereq}')
    for (idx, (ia, ib)) in enumerate(zip(bundle_item_a, bundle_item_b)):
        if ia in items and ib in items:
            m.addConstr(b[idx] <= z[ia], name=f'bundle_a_{idx}')
            m.addConstr(b[idx] <= z[ib], name=f'bundle_b_{idx}')
            m.addConstr(b[idx] >= z[ia] + z[ib] - 1, name=f'bundle_link_{idx}')
        else:
            m.addConstr(b[idx] == 0, name=f'bundle_invalid_{idx}')
    obj = gp.LinExpr()
    obj += gp.quicksum((item_unit_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * z[i] for i in items))
    obj -= gp.quicksum((cat_fee[c] * w[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[idx] * b[idx] for idx in bundles))
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