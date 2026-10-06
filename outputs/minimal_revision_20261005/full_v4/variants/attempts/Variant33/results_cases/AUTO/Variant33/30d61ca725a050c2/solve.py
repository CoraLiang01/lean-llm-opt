import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv'

def solve_problem():
    df_benefit = pd.read_csv(f_benefit, sep=',')
    df_bundle = pd.read_csv(f_bundle, sep=',')
    df_capacity = pd.read_csv(f_capacity, sep=',')
    df_category = pd.read_csv(f_category, sep=',')
    df_identity = pd.read_csv(f_identity, sep=',')
    df_incompatible = pd.read_csv(f_incompatible, sep=',')
    df_item1 = pd.read_csv(f_item1, sep=',')
    df_item2 = pd.read_csv(f_item2, sep=',')
    df_item_fee = pd.read_csv(f_item_fee, sep=',')
    df_requires = pd.read_csv(f_requires, sep=',')
    df_usage1 = pd.read_csv(f_usage1, sep=',')
    df_usage2 = pd.read_csv(f_usage2, sep=',')
    items1 = df_item1['item_ref'].astype(str).unique().tolist()
    items2 = df_item2['item_ref'].astype(str).unique().tolist()
    all_items = sorted(set(items1) | set(items2))
    df_items = pd.concat([df_item1, df_item2], ignore_index=True)
    df_items['item_ref'] = df_items['item_ref'].astype(str)
    df_items['category'] = df_items['category'].astype(str)
    df_items['authorized'] = df_items['authorized'].astype(int)
    df_items['minimum_lot'] = df_items['minimum_lot'].astype(int)
    df_items['maximum_order'] = df_items['maximum_order'].astype(int)
    authorized_items = df_items[df_items['authorized'] == 1]['item_ref'].unique().tolist()
    item_category = df_items.set_index('item_ref')['category'].to_dict()
    item_minlot = df_items.set_index('item_ref')['minimum_lot'].to_dict()
    item_maxorder = df_items.set_index('item_ref')['maximum_order'].to_dict()
    df_category['category'] = df_category['category'].astype(str)
    categories = df_category['category'].unique().tolist()
    cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
    cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
    df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
    benefit = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
    benefit = {i: benefit.get(i, 0) for i in all_items}
    df_item_fee['item_ref'] = df_item_fee['item_ref'].astype(str)
    item_fee = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
    item_fee = {i: item_fee.get(i, 0) for i in all_items}
    df_usage1['item_ref'] = df_usage1['item_ref'].astype(str)
    df_usage2['item_ref'] = df_usage2['item_ref'].astype(str)
    df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
    df_usage['resource'] = df_usage['resource'].astype(str)
    df_usage['amount'] = df_usage['amount'].astype(int)
    usage = df_usage.groupby(['item_ref', 'resource'])['amount'].sum().to_dict()
    resources = df_usage['resource'].unique().tolist()
    df_capacity['resource'] = df_capacity['resource'].astype(str)
    df_capacity['amount'] = df_capacity['amount'].astype(int)
    cap_ledger = df_capacity.groupby('resource')['amount'].sum().to_dict()
    cap_ledger = {r: cap_ledger.get(r, 0) for r in resources}
    df_bundle['item_a'] = df_bundle['item_a'].astype(str)
    df_bundle['item_b'] = df_bundle['item_b'].astype(str)
    bundles = []
    bundle_bonus = {}
    for (_, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        bundles.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
    df_incompatible['item_a'] = df_incompatible['item_a'].astype(str)
    df_incompatible['item_b'] = df_incompatible['item_b'].astype(str)
    incompatibles = []
    for (_, row) in df_incompatible.iterrows():
        incompatibles.append((row['item_a'], row['item_b']))
    df_requires['item_ref'] = df_requires['item_ref'].astype(str)
    df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
    requires = []
    for (_, row) in df_requires.iterrows():
        requires.append((row['item_ref'], row['prerequisite_ref']))
    item_to_category = {}
    for i in all_items:
        if i in item_category:
            item_to_category[i] = item_category[i]
        else:
            row = df_items[df_items['item_ref'] == i]
            if not row.empty:
                item_to_category[i] = row.iloc[0]['category']
            else:
                raise ValueError(f'Item {i} missing category assignment.')
    cat_items = {c: [i for i in all_items if item_to_category[i] == c] for c in categories}
    min_lot = {i: item_minlot.get(i, 0) for i in all_items}
    max_order = {i: item_maxorder.get(i, 0) for i in all_items}
    is_authorized = {i: int(i in authorized_items) for i in all_items}
    m = gp.Model('NY_Module_Portfolio')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(all_items, lb=0, ub={i: max_order[i] for i in all_items}, vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(all_items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in all_items:
        if is_authorized[i]:
            m.addConstr(x[i] <= max_order[i] * y[i], name='')
            m.addConstr(x[i] >= min_lot[i] * y[i], name='')
        else:
            m.addConstr(x[i] == 0, name='')
            m.addConstr(y[i] == 0, name='')
    for c in categories:
        for i in cat_items[c]:
            m.addConstr(z[c] >= y[i], name='')
        m.addConstr(z[c] <= gp.quicksum((y[i] for i in cat_items[c])), name='')
    for c in categories:
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c] * z[c], name='')
        m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c] * z[c], name='')
    for r in resources:
        expr = gp.LinExpr()
        for i in all_items:
            amt = usage.get((i, r), 0)
            expr += amt * x[i]
        m.addConstr(expr <= cap_ledger[r], name='')
    for (i, j) in incompatibles:
        if i in all_items and j in all_items:
            m.addConstr(y[i] + y[j] <= 1, name='')
    for (i, prereq) in requires:
        if i in all_items and prereq in all_items:
            m.addConstr(y[i] <= y[prereq], name='')
    for (i, j) in bundles:
        if i in all_items and j in all_items:
            m.addConstr(b[i, j] <= y[i], name='')
            m.addConstr(b[i, j] <= y[j], name='')
            m.addConstr(b[i, j] >= y[i] + y[j] - 1, name='')
        else:
            m.addConstr(b[i, j] == 0, name='')
    obj = gp.LinExpr()
    obj += gp.quicksum((benefit[i] * x[i] for i in all_items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in all_items))
    obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundles))
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