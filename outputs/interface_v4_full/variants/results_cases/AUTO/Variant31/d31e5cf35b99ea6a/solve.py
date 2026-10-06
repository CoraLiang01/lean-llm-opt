import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_01/export_01.csv', sep=',')
    df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_02/export_02.csv', sep=',')
    df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_03/export_03.csv', sep=',')
    df_id = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_04/export_04.csv', sep=',')
    df_incomp = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_05/export_05.csv', sep=',')
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_06/export_06.csv', sep=',')
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_01/export_07.csv', sep=',')
    df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_02/export_08.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant31/inputs/batch_03/export_09.csv', sep=',')
    items = df_item['item_ref'].astype(str).tolist()
    categories = df_cat['category'].astype(str).tolist()
    resources = df_cap['resource'].astype(str).unique().tolist()
    bundle_tuples = list(df_bundle[['item_a', 'item_b']].astype(str).itertuples(index=False, name=None))
    bundle_keys = list(range(len(bundle_tuples)))
    incomp_pairs = list(df_incomp[['item_a', 'item_b']].astype(str).itertuples(index=False, name=None))
    req_pairs = list(df_req[['item_ref', 'prerequisite_ref']].astype(str).itertuples(index=False, name=None))
    item_data = df_item.set_index('item_ref').astype({'authorized': int, 'minimum_lot': int, 'maximum_order': int, 'unit_benefit_cents': int, 'item_fee_cents': int, 'category': str})
    authorized = item_data['authorized'].to_dict()
    minimum_lot = item_data['minimum_lot'].to_dict()
    maximum_order = item_data['maximum_order'].to_dict()
    unit_benefit_cents = item_data['unit_benefit_cents'].to_dict()
    item_fee_cents = item_data['item_fee_cents'].to_dict()
    item_category = item_data['category'].to_dict()
    cat_data = df_cat.set_index('category').astype({'minimum_quantity': int, 'maximum_quantity': int, 'activation_fee_cents': int})
    minimum_quantity = cat_data['minimum_quantity'].to_dict()
    maximum_quantity = cat_data['maximum_quantity'].to_dict()
    activation_fee_cents = cat_data['activation_fee_cents'].to_dict()
    df_cap['resource'] = df_cap['resource'].astype(str)
    df_cap['entry'] = df_cap['entry'].astype(str)
    cap_opening = df_cap[df_cap['entry'] == 'opening'].groupby('resource')['amount'].sum()
    cap_reservation = df_cap[df_cap['entry'] == 'reservation'].groupby('resource')['amount'].sum()
    total_capacity = (cap_opening + cap_reservation).to_dict()
    for r in resources:
        if r not in total_capacity:
            total_capacity[r] = cap_opening.get(r, 0)
    df_usage['item_ref'] = df_usage['item_ref'].astype(str)
    df_usage['resource'] = df_usage['resource'].astype(str)
    usage = {}
    for _, row in df_usage.iterrows():
        usage[row['item_ref'], row['resource']] = int(row['amount'])
    for i in items:
        for r in resources:
            if (i, r) not in usage:
                usage[i, r] = 0
    bundle_bonus = {}
    for idx, row in df_bundle.iterrows():
        bundle_bonus[idx] = int(row['bonus_cents'])
    bundle_idx_to_items = {idx: (row['item_a'], row['item_b']) for idx, row in df_bundle.iterrows()}
    m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
    x = {}
    y = {}
    for i in items:
        if authorized[i] == 1:
            x[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=maximum_order[i], name=f'x_{i}')
            y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
        else:
            x[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=0, name=f'x_{i}')
            y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
            m.addConstr(x[i] == 0, name=f'unauth_x_{i}')
            m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
    z = {c: m.addVar(vtype=gp.GRB.BINARY, name=f'z_{c}') for c in categories}
    b = {idx: m.addVar(vtype=gp.GRB.BINARY, name=f'b_{idx}') for idx in bundle_keys}
    m.update()
    for i in items:
        if authorized[i] == 1:
            m.addConstr(x[i] >= minimum_lot[i] * y[i], name=f'minlot_{i}')
            m.addConstr(x[i] <= maximum_order[i] * y[i], name=f'maxorder_{i}')
            m.addConstr(y[i] <= 1, name=f'ybin_{i}')
    for c in categories:
        items_in_c = [i for i in items if item_category[i] == c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= maximum_quantity[c], name=f'cat_max_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= minimum_quantity[c] * z[c], name=f'cat_min_{c}')
        for i in items_in_c:
            m.addConstr(y[i] <= z[c], name=f'link_yz_{i}_{c}')
    for r in resources:
        m.addConstr(gp.quicksum((usage[i, r] * x[i] for i in items)) <= total_capacity[r], name=f'res_{r}')
    for i, j in incomp_pairs:
        if i in y and j in y:
            m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for i, prereq in req_pairs:
        if i in y and prereq in y:
            m.addConstr(y[i] <= y[prereq], name=f'req_{i}_{prereq}')
    for idx, (i_a, i_b) in bundle_idx_to_items.items():
        if i_a in y and i_b in y:
            m.addConstr(b[idx] <= y[i_a], name=f'bundle1_{idx}')
            m.addConstr(b[idx] <= y[i_b], name=f'bundle2_{idx}')
            m.addConstr(b[idx] >= y[i_a] + y[i_b] - 1, name=f'bundle3_{idx}')
        else:
            m.addConstr(b[idx] == 0, name=f'bundle0_{idx}')
    obj = gp.quicksum((unit_benefit_cents[i] * x[i] for i in items)) - gp.quicksum((item_fee_cents[i] * y[i] for i in items)) - gp.quicksum((activation_fee_cents[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[idx] * b[idx] for idx in bundle_keys))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()