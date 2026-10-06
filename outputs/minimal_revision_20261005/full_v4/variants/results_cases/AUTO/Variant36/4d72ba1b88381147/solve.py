import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv'
f_item_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv'
f_item_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv'
f_usage_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv'
f_usage_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv'

def solve_problem():
    df_bundle = pd.read_csv(f_bundle, sep=',')
    df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
    df_category = pd.read_csv(f_category, sep=',')
    df_identity = pd.read_csv(f_identity, sep=',')
    df_incompatible = pd.read_csv(f_incompatible, sep=',')
    df_requires = pd.read_csv(f_requires, sep=',')
    df_item_1 = pd.read_csv(f_item_1, sep=',')
    df_item_2 = pd.read_csv(f_item_2, sep=',')
    df_usage_1 = pd.read_csv(f_usage_1, sep=',')
    df_usage_2 = pd.read_csv(f_usage_2, sep=',')
    df_items = pd.concat([df_item_1, df_item_2], ignore_index=True)
    df_items = df_items[df_items['authorized'] == 1].copy()
    df_items = df_items.drop_duplicates(subset=['item_ref'])
    item_refs = df_items['item_ref'].astype(str).unique().tolist()
    categories = df_category['category'].astype(str).unique().tolist()
    areas = pd.concat([df_items['location_id'].astype(str), df_capacity_ledger['resource'].astype(str)]).unique().tolist()
    item_param = df_items.set_index('item_ref').to_dict(orient='index')
    cat_param = df_category.set_index('category').to_dict(orient='index')
    df_usage = pd.concat([df_usage_1, df_usage_2], ignore_index=True)
    df_usage = df_usage[df_usage['item_ref'].astype(str).isin(item_refs)].copy()
    usage = {}
    for (_, row) in df_usage.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        usage[i, r] = int(row['amount'])
    df_capacity_ledger['resource'] = df_capacity_ledger['resource'].astype(str)
    cap_ledger = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
    df_incompatible = df_incompatible.copy()
    df_incompatible['item_a'] = df_incompatible['item_a'].astype(str)
    df_incompatible['item_b'] = df_incompatible['item_b'].astype(str)
    incompatible_pairs = set()
    for (_, row) in df_incompatible.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in item_refs and b in item_refs:
            incompatible_pairs.add(tuple(sorted((a, b))))
    df_requires = df_requires.copy()
    df_requires['item_ref'] = df_requires['item_ref'].astype(str)
    df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
    requires_pairs = []
    for (_, row) in df_requires.iterrows():
        i = row['item_ref']
        p = row['prerequisite_ref']
        if i in item_refs and p in item_refs:
            requires_pairs.append((i, p))
    df_bundle = df_bundle.copy()
    df_bundle['item_a'] = df_bundle['item_a'].astype(str)
    df_bundle['item_b'] = df_bundle['item_b'].astype(str)
    bundle_list = []
    for (_, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in item_refs and b in item_refs:
            bundle_list.append((a, b, int(row['bonus_cents'])))
    bundle_keys = []
    bundle_bonus = {}
    for (a, b, bonus) in bundle_list:
        key = tuple(sorted((a, b)))
        bundle_keys.append(key)
        bundle_bonus[key] = bonus
    bundle_keys = list(set(bundle_keys))
    cat_items = {c: [] for c in categories}
    for i in item_refs:
        c = str(item_param[i]['category'])
        cat_items[c].append(i)
    area_items = {}
    for i in item_refs:
        area = str(item_param[i]['location_id'])
        area_items.setdefault(area, []).append(i)
    m = gp.Model('FC_EAST_HVAC_Placement')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
    for i in item_refs:
        min_lot = int(item_param[i]['minimum_lot'])
        max_order = int(item_param[i]['maximum_order'])
        m.addConstr(x[i] >= min_lot * y[i], name='minlot_' + i)
        m.addConstr(x[i] <= max_order * y[i], name='maxorder_' + i)
        m.addConstr(x[i] <= max_order * y[i], name='link_y_' + i)
        m.addConstr(x[i] >= y[i], name='link_y2_' + i)
    for area in cap_ledger:
        items_in_area = [i for i in item_refs if str(item_param[i]['location_id']) == area]
        if not items_in_area:
            continue
        expr = gp.LinExpr()
        for i in items_in_area:
            if (i, area) in usage:
                expr += usage[i, area] * x[i]
            else:
                expr += 0
        m.addConstr(expr <= cap_ledger[area], name='cap_' + area)
    for c in categories:
        items_in_cat = cat_items[c]
        minq = int(cat_param[c]['minimum_quantity'])
        maxq = int(cat_param[c]['maximum_quantity'])
        m.addConstr(gp.quicksum((x[i] for i in items_in_cat)) >= minq, name='cat_min_' + c)
        m.addConstr(gp.quicksum((x[i] for i in items_in_cat)) <= maxq, name='cat_max_' + c)
        for i in items_in_cat:
            m.addConstr(y[i] <= z[c], name='cat_act_' + c + '_' + i)
        m.addConstr(gp.quicksum((y[i] for i in items_in_cat)) >= z[c], name='cat_act2_' + c)
    for (a, b) in incompatible_pairs:
        m.addConstr(y[a] + y[b] <= 1, name='incomp_' + a + '_' + b)
    for (i, p) in requires_pairs:
        m.addConstr(y[i] <= y[p], name='req_' + i + '_' + p)
    for key in bundle_keys:
        (a, b) = key
        m.addConstr(b[key] <= y[a], name='bundle1_' + a + '_' + b)
        m.addConstr(b[key] <= y[b], name='bundle2_' + a + '_' + b)
        m.addConstr(b[key] >= y[a] + y[b] - 1, name='bundle3_' + a + '_' + b)
    obj = gp.LinExpr()
    for i in item_refs:
        obj += int(item_param[i]['unit_benefit_cents']) * x[i]
        obj -= int(item_param[i]['item_fee_cents']) * y[i]
    for c in categories:
        obj -= int(cat_param[c]['activation_fee_cents']) * z[c]
    for key in bundle_keys:
        obj += int(bundle_bonus[key]) * b[key]
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