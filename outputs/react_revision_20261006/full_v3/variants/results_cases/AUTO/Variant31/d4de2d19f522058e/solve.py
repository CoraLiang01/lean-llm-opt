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
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', sep=',')
    df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', sep=',')
    items = df_item['item_ref'].astype(str).unique().tolist()
    categories = df_cat['category'].astype(str).unique().tolist()
    resources = sorted(set(df_cap['resource'].astype(str).unique()).union(df_usage['resource'].astype(str).unique()))
    bundle_keys = list(df_bundle.index)
    incomp_keys = list(df_incomp.index)
    req_keys = list(df_req.index)
    item_df = df_item.set_index('item_ref')
    for col in ['category', 'authorized', 'minimum_lot', 'maximum_order', 'unit_benefit_cents', 'item_fee_cents']:
        if col not in item_df.columns:
            raise KeyError(f'Missing column {col} in item table')
    item_to_cat = item_df['category'].astype(str).to_dict()
    authorized = item_df['authorized'].astype(int).to_dict()
    min_lot = item_df['minimum_lot'].astype(int).to_dict()
    max_order = item_df['maximum_order'].astype(int).to_dict()
    unit_benefit = item_df['unit_benefit_cents'].astype(int).to_dict()
    item_fee = item_df['item_fee_cents'].astype(int).to_dict()
    cat_df = df_cat.set_index('category')
    for col in ['minimum_quantity', 'maximum_quantity', 'activation_fee_cents']:
        if col not in cat_df.columns:
            raise KeyError(f'Missing column {col} in category table')
    min_qty = cat_df['minimum_quantity'].astype(int).to_dict()
    max_qty = cat_df['maximum_quantity'].astype(int).to_dict()
    activation_fee = cat_df['activation_fee_cents'].astype(int).to_dict()
    usage_df = df_usage.copy()
    usage_df['item_ref'] = usage_df['item_ref'].astype(str)
    usage_df['resource'] = usage_df['resource'].astype(str)
    usage_map = {}
    for (_, row) in usage_df.iterrows():
        usage_map[row['item_ref'], row['resource']] = int(row['amount'])
    cap_df = df_cap.copy()
    cap_df['resource'] = cap_df['resource'].astype(str)
    cap_df['entry'] = cap_df['entry'].astype(str)
    cap_sum = cap_df.groupby(['resource', 'entry'])['amount'].sum().unstack(fill_value=0)
    total_capacity = {}
    for r in resources:
        opening = cap_sum.loc[r, 'opening'] if 'opening' in cap_sum.columns and r in cap_sum.index else 0
        reservation = cap_sum.loc[r, 'reservation'] if 'reservation' in cap_sum.columns and r in cap_sum.index else 0
        total_capacity[r] = opening + reservation
    bundle_list = []
    for (idx, row) in df_bundle.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        bonus = int(row['bonus_cents'])
        bundle_list.append((idx, a, b, bonus))
    incomp_pairs = []
    for (idx, row) in df_incomp.iterrows():
        i = str(row['item_a'])
        j = str(row['item_b'])
        incomp_pairs.append((i, j))
    req_pairs = []
    for (idx, row) in df_req.iterrows():
        i = str(row['item_ref'])
        prereq = str(row['prerequisite_ref'])
        req_pairs.append((i, prereq))
    m = gp.Model('RIVERSIDE_AUTO_VEHICLE_SELECTION')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars([idx for (idx, _, _, _) in bundle_list], vtype=gp.GRB.BINARY, name='')
    for i in items:
        if authorized[i] == 1:
            m.addConstr(x[i] >= min_lot[i] * y[i], name=f'item_min_{i}')
            m.addConstr(x[i] <= max_order[i] * y[i], name=f'item_max_{i}')
        else:
            m.addConstr(x[i] == 0, name=f'item_unauth_x_{i}')
            m.addConstr(y[i] == 0, name=f'item_unauth_y_{i}')
    cat_to_items = {c: [] for c in categories}
    for i in items:
        c = item_to_cat[i]
        cat_to_items[c].append(i)
    for c in categories:
        items_in_c = cat_to_items[c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= min_qty[c] * z[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= max_qty[c] * z[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
    for r in resources:
        expr = gp.LinExpr()
        for i in items:
            amt = usage_map.get((i, r), 0)
            expr += amt * x[i]
        m.addConstr(expr <= total_capacity[r], name=f'res_cap_{r}')
    for (i, j) in incomp_pairs:
        if i in items and j in items:
            m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, prereq) in req_pairs:
        if i in items and prereq in items:
            m.addConstr(y[i] <= y[prereq], name=f'req_{i}_{prereq}')
    for (idx, a, b, _) in bundle_list:
        if a in items and b in items:
            m.addConstr(b[idx] <= y[a], name=f'bundle1_{idx}')
            m.addConstr(b[idx] <= y[b], name=f'bundle2_{idx}')
            m.addConstr(b[idx] >= y[a] + y[b] - 1, name=f'bundle3_{idx}')
        else:
            m.addConstr(b[idx] == 0, name=f'bundle_na_{idx}')
    for i in items:
        m.addConstr(y[i] <= gp.quicksum([x[i] >= 1]), name=f'y_link_{i}')
    obj = gp.LinExpr()
    obj += gp.quicksum((unit_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
    obj -= gp.quicksum((activation_fee[c] * z[c] for c in categories))
    for (idx, _, _, bonus) in bundle_list:
        obj += bonus * b[idx]
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