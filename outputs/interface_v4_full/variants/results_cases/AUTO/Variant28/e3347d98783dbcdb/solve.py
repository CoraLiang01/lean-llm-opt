import gurobipy as gp
import pandas as pd
import numpy as np
import math

def solve_problem():
    path_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_01/export_01.csv'
    path_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_02/export_02.csv'
    path_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_03/export_03.csv'
    path_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_04/export_04.csv'
    path_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_05/export_05.csv'
    path_items = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_06/export_06.csv'
    path_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_01/export_07.csv'
    path_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_02/export_08.csv'
    path_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant28/inputs/batch_03/export_09.csv'
    df_bundle = pd.read_csv(path_bundle, sep=',')
    df_capacity = pd.read_csv(path_capacity, sep=',')
    df_category = pd.read_csv(path_category, sep=',')
    df_identity = pd.read_csv(path_identity, sep=',')
    df_incompat = pd.read_csv(path_incompat, sep=',')
    df_items = pd.read_csv(path_items, sep=',')
    df_market = pd.read_csv(path_market, sep=',')
    df_requires = pd.read_csv(path_requires, sep=',')
    df_usage = pd.read_csv(path_usage, sep=',')
    items_df = df_items[df_items['authorized'] == 1].copy()
    items = list(items_df['item_ref'])
    item_to_cat = items_df.set_index('item_ref')['category'].to_dict()
    min_lot = items_df.set_index('item_ref')['minimum_lot'].to_dict()
    max_order = items_df.set_index('item_ref')['maximum_order'].to_dict()
    unit_benefit = items_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
    item_fee = items_df.set_index('item_ref')['item_fee_cents'].to_dict()
    categories = list(df_category['category'])
    cat_min_qty = df_category.set_index('category')['minimum_quantity'].to_dict()
    cat_max_qty = df_category.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
    cat_to_items = {c: [i for i in items if item_to_cat[i] == c] for c in categories}
    usage_df = df_usage[df_usage['item_ref'].isin(items)].copy()

    def convert_to_base(row):
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if u == 'kwh':
            return amt * 1000
        elif u == 'liter':
            return amt * 1000
        elif u == 'hour':
            return amt * 60
        else:
            return amt
    usage_df['amount_base'] = usage_df.apply(convert_to_base, axis=1)
    usage_per_item_resource = {}
    for _, row in usage_df.iterrows():
        i = row['item_ref']
        r = row['resource']
        usage_per_item_resource[i, r] = row['amount_base']
    resources = sorted(set(usage_df['resource']))
    cap_df = df_capacity.copy()

    def cap_convert_to_base(row):
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if u == 'kwh':
            return amt * 1000
        elif u == 'liter':
            return amt * 1000
        elif u == 'hour':
            return amt * 60
        else:
            return amt
    cap_df['amount_base'] = cap_df.apply(cap_convert_to_base, axis=1)
    cap_by_resource = cap_df.groupby('resource')['amount_base'].sum().to_dict()
    cap_by_resource = {r: cap_by_resource[r] for r in resources if r in cap_by_resource}
    bundle_df = df_bundle.copy()
    bundle_df = bundle_df[bundle_df['item_a'].isin(items) & bundle_df['item_b'].isin(items)]
    bundles = list(bundle_df.index)
    bundle_pairs = bundle_df[['item_a', 'item_b']].to_dict('index')
    bundle_bonus = bundle_df['bonus_cents'].to_dict()
    incompat_df = df_incompat.copy()
    incompat_df = incompat_df[incompat_df['item_a'].isin(items) & incompat_df['item_b'].isin(items)]
    incompat_pairs = list(zip(incompat_df['item_a'], incompat_df['item_b']))
    req_df = df_requires.copy()
    req_df = req_df[req_df['item_ref'].isin(items) & req_df['prerequisite_ref'].isin(items)]
    prereq_pairs = list(zip(req_df['item_ref'], req_df['prerequisite_ref']))
    m = gp.Model('CENTRAL_FRESH_Produce_Order')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        m.addConstr(x[i] >= y[i], name=f'link_lb_{i}')
        m.addConstr(x[i] <= max_order[i] * y[i], name=f'link_ub_{i}')
        m.addConstr(x[i] == 0 or x[i] >= min_lot[i], name=f'minlot_{i}')
        m.addConstr(x[i] >= min_lot[i] * y[i], name=f'minlot2_{i}')
    for c in categories:
        items_in_c = cat_to_items[c]
        if items_in_c:
            m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_min_qty[c] * z[c], name=f'cat_min_{c}')
            m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_max_qty[c] * z[c], name=f'cat_max_{c}')
            for i in items_in_c:
                m.addConstr(y[i] <= z[c], name=f'cat_link_{i}_{c}')
        else:
            m.addConstr(z[c] == 0, name=f'cat_empty_{c}')
    for r in resources:
        m.addConstr(gp.quicksum((usage_per_item_resource.get((i, r), 0) * x[i] for i in items)) <= cap_by_resource[r], name=f'res_cap_{r}')
    for i, j in incompat_pairs:
        m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
    for i, j in prereq_pairs:
        m.addConstr(y[i] <= y[j], name=f'prereq_{i}_{j}')
    for bundle_idx in bundles:
        i = bundle_pairs[bundle_idx]['item_a']
        j = bundle_pairs[bundle_idx]['item_b']
        m.addConstr(b[bundle_idx] <= y[i], name=f'bundle_{bundle_idx}_a')
        m.addConstr(b[bundle_idx] <= y[j], name=f'bundle_{bundle_idx}_b')
        m.addConstr(b[bundle_idx] >= y[i] + y[j] - 1, name=f'bundle_{bundle_idx}_both')
    obj = gp.quicksum((unit_benefit[i] * x[i] for i in items)) - gp.quicksum((item_fee[i] * y[i] for i in items)) - gp.quicksum((cat_fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[bundle_idx] * b[bundle_idx] for bundle_idx in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()