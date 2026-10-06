import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv'
f_item_06 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv'
f_item_07 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv'
f_usage_10 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv'
f_usage_11 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv'

def solve_problem():
    df_bundle = pd.read_csv(f_bundle)
    df_capacity_ledger = pd.read_csv(f_capacity_ledger)
    df_category = pd.read_csv(f_category)
    df_identity = pd.read_csv(f_identity)
    df_incompatible = pd.read_csv(f_incompatible)
    df_item_06 = pd.read_csv(f_item_06)
    df_item_07 = pd.read_csv(f_item_07)
    df_requires = pd.read_csv(f_requires)
    df_usage_10 = pd.read_csv(f_usage_10)
    df_usage_11 = pd.read_csv(f_usage_11)
    df_items = pd.concat([df_item_06, df_item_07], ignore_index=True)
    df_items = df_items.drop_duplicates(subset=['item_ref'], keep='first').reset_index(drop=True)
    items = df_items['item_ref'].astype(str).tolist()
    item_authorized = df_items.set_index('item_ref')['authorized'].astype(int).to_dict()
    item_min_lot = df_items.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
    item_max_order = df_items.set_index('item_ref')['maximum_order'].astype(int).to_dict()
    item_category = df_items.set_index('item_ref')['category'].astype(str).to_dict()
    item_location = df_items.set_index('item_ref')['location_id'].astype(str).to_dict()
    item_unit_benefit = df_items.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
    item_fee = df_items.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
    categories = df_category['category'].astype(str).tolist()
    cat_min_qty = df_category.set_index('category')['minimum_quantity'].astype(int).to_dict()
    cat_max_qty = df_category.set_index('category')['maximum_quantity'].astype(int).to_dict()
    cat_activation_fee = df_category.set_index('category')['activation_fee_cents'].astype(int).to_dict()
    resources = df_capacity_ledger['resource'].astype(str).unique().tolist()
    df_cap = df_capacity_ledger.groupby('resource')['amount'].sum()
    resource_capacity = df_cap.to_dict()
    df_usage = pd.concat([df_usage_10, df_usage_11], ignore_index=True)
    usage_dict = {}
    for _, row in df_usage.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        usage_dict[i, r] = int(row['amount'])
    incompatible_pairs = set()
    for _, row in df_incompatible.iterrows():
        i = str(row['item_a'])
        j = str(row['item_b'])
        incompatible_pairs.add((i, j))
        incompatible_pairs.add((j, i))
    requires_pairs = []
    for _, row in df_requires.iterrows():
        i = str(row['item_ref'])
        prereq = str(row['prerequisite_ref'])
        requires_pairs.append((i, prereq))
    bundle_list = []
    bundle_bonus = {}
    for idx, row in df_bundle.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        bundle_list.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
    m = gp.Model('FC_EAST_HVAC_Placement')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if item_authorized[i] == 0:
            m.addConstr(x[i] == 0, name=f'auth_x_{i}')
            m.addConstr(y[i] == 0, name=f'auth_y_{i}')
        else:
            m.addConstr(x[i] <= item_max_order[i] * y[i], name=f'link_x_y_ub_{i}')
            m.addConstr(x[i] >= item_min_lot[i] * y[i], name=f'link_x_y_lb_{i}')
            m.addConstr(x[i] <= item_max_order[i], name=f'max_order_{i}')
            m.addConstr((x[i] == 0) | (x[i] >= item_min_lot[i]), name=f'min_lot_{i}')
    for r in resources:
        items_in_r = [i for i in items if item_location[i] == r]
        m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x[i] for i in items_in_r)) <= resource_capacity[r], name=f'cap_{r}')
    for c in categories:
        items_in_c = [i for i in items if item_category[i] == c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_min_qty[c] * z[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_max_qty[c] * z[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(y[i] <= z[c], name=f'cat_link_{i}_{c}')
    for i, j in incompatible_pairs:
        if i in items and j in items:
            m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for i, prereq in requires_pairs:
        if i in items and prereq in items:
            m.addConstr(y[i] <= y[prereq], name=f'req_{i}_{prereq}')
    for a, b_ in bundle_list:
        if a in items and b_ in items:
            m.addConstr(b[a, b_] <= y[a], name=f'bundle1_{a}_{b_}')
            m.addConstr(b[a, b_] <= y[b_], name=f'bundle2_{a}_{b_}')
            m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle3_{a}_{b_}')
        else:
            m.addConstr(b[a, b_] == 0, name=f'bundle_na_{a}_{b_}')
    obj = gp.quicksum((item_unit_benefit[i] * x[i] for i in items)) - gp.quicksum((item_fee[i] * y[i] for i in items)) - gp.quicksum((cat_activation_fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[a, b_] * b[a, b_] for a, b_ in bundle_list))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()