import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_01/export_01.csv', sep=',')
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_02/export_02.csv', sep=',')
    df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_03/export_03.csv', sep=',')
    df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_04/export_04.csv', sep=',')
    df_fx = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_05/export_05.csv', sep=',')
    df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_06/export_06.csv', sep=',')
    df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_01/export_07.csv', sep=',')
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_02/export_08.csv', sep=',')
    df_itemfee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_03/export_09.csv', sep=',')
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_04/export_10.csv', sep=',')
    df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_05/export_11.csv', sep=',')
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant35/inputs/batch_06/export_12.csv', sep=',')
    items = df_item['item_ref'].astype(str).tolist()
    categories = df_category['category'].astype(str).tolist()
    platforms = sorted(df_capacity['resource'].astype(str).unique())
    bundle_idx = df_bundle.index.tolist()
    fx_map = df_fx.set_index('currency')[['usd_cents_numerator', 'denominator']].to_dict(orient='index')

    def benefit_row_to_usd_cents(row):
        currency = str(row['currency'])
        amt = row['amount']
        fx = fx_map[currency]
        return amt * fx['usd_cents_numerator'] / fx['denominator']
    df_benefit['usd_cents'] = df_benefit.apply(benefit_row_to_usd_cents, axis=1)
    benefit_per_item = df_benefit.groupby('item_ref')['usd_cents'].sum().to_dict()
    for i in items:
        if i not in benefit_per_item:
            benefit_per_item[i] = 0.0
    item_fee = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
    for i in items:
        if i not in item_fee:
            item_fee[i] = 0
    category_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
    bundle_bonus = df_bundle['bonus_cents'].to_dict()
    bundle_item_a = df_bundle['item_a'].astype(str).to_dict()
    bundle_item_b = df_bundle['item_b'].astype(str).to_dict()
    df_usage['usage_MB'] = df_usage['amount'] * 1000
    usage = {}
    for _, row in df_usage.iterrows():
        usage[str(row['item_ref']), str(row['resource'])] = row['usage_MB']
    df_capacity['amount_MB'] = df_capacity['amount']
    platform_capacity = df_capacity.groupby('resource')['amount_MB'].sum().to_dict()
    for p in platforms:
        if p not in platform_capacity:
            platform_capacity[p] = 0
    category_min = df_category.set_index('category')['minimum_quantity'].to_dict()
    category_max = df_category.set_index('category')['maximum_quantity'].to_dict()
    item_info = df_item.set_index('item_ref')[['category', 'authorized', 'minimum_lot', 'maximum_order', 'location_id']]
    item_category = item_info['category'].astype(str).to_dict()
    item_authorized = item_info['authorized'].to_dict()
    item_minlot = item_info['minimum_lot'].to_dict()
    item_maxorder = item_info['maximum_order'].to_dict()
    item_platform = item_info['location_id'].astype(str).to_dict()
    incompat_pairs = set()
    for _, row in df_incompat.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        incompat_pairs.add((a, b))
        incompat_pairs.add((b, a))
    prereq_pairs = []
    for _, row in df_requires.iterrows():
        i = str(row['item_ref'])
        j = str(row['prerequisite_ref'])
        prereq_pairs.append((i, j))
    m = gp.Model('GameEditionAllocation')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_idx, vtype=gp.GRB.BINARY, name='')
    for i in items:
        auth = int(item_authorized[i])
        minlot = int(item_minlot[i])
        maxorder = int(item_maxorder[i])
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'auth_{i}')
            m.addConstr(z[i] == 0, name=f'authz_{i}')
        else:
            m.addConstr(x[i] >= 0, name=f'xnonneg_{i}')
            m.addConstr(x[i] <= maxorder * z[i], name=f'xzlink_ub_{i}')
            m.addConstr(x[i] >= minlot * z[i], name=f'xzlink_lb_{i}')
            m.addConstr(x[i] <= maxorder, name=f'xmax_{i}')
            m.addConstr(z[i] <= 1, name=f'zbin_{i}')
    for c in categories:
        items_in_c = [i for i in items if item_category[i] == c and int(item_authorized[i]) == 1]
        minlot_c = min([item_minlot[i] for i in items_in_c]) if items_in_c else 0
        maxorder_sum = sum([item_maxorder[i] for i in items_in_c]) if items_in_c else 0
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= minlot_c * w[c], name=f'catw_lb_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= maxorder_sum * w[c], name=f'catw_ub_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= category_min[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= category_max[c], name=f'catmax_{c}')
    for p in platforms:
        items_on_p = [i for i in items if item_platform[i] == p and int(item_authorized[i]) == 1]
        m.addConstr(gp.quicksum((usage.get((i, p), 0) * x[i] for i in items_on_p)) <= platform_capacity[p], name=f'cap_{p}')
    for i, j in incompat_pairs:
        if i in items and j in items:
            m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
    for i, j in prereq_pairs:
        if i in items and j in items:
            m.addConstr(z[i] <= z[j], name=f'prereq_{i}_{j}')
    for idx in bundle_idx:
        i = bundle_item_a[idx]
        j = bundle_item_b[idx]
        if i in items and j in items:
            m.addConstr(b[idx] <= z[i], name=f'bundle_ba_{idx}')
            m.addConstr(b[idx] <= z[j], name=f'bundle_bb_{idx}')
            m.addConstr(b[idx] >= z[i] + z[j] - 1, name=f'bundle_bc_{idx}')
        else:
            m.addConstr(b[idx] == 0, name=f'bundle_bzero_{idx}')
    obj = gp.quicksum((benefit_per_item[i] * x[i] for i in items)) - gp.quicksum((item_fee[i] * z[i] for i in items)) - gp.quicksum((category_fee[c] * w[c] for c in categories)) + gp.quicksum((bundle_bonus[idx] * b[idx] for idx in bundle_idx))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()