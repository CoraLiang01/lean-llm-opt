import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
    df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
    df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
    df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
    df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
    df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
    df_item['authorized'] = df_item['authorized'].astype(int)
    items = df_item.loc[df_item['authorized'] == 1, 'item_ref'].tolist()
    item_to_category = df_item.set_index('item_ref')['category'].to_dict()
    min_lot = df_item.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
    max_order = df_item.set_index('item_ref')['maximum_order'].astype(int).to_dict()
    unit_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
    item_fee = df_item.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
    categories = df_category['category'].tolist()
    min_cat_qty = df_category.set_index('category')['minimum_quantity'].astype(int).to_dict()
    max_cat_qty = df_category.set_index('category')['maximum_quantity'].astype(int).to_dict()
    cat_fee = df_category.set_index('category')['activation_fee_cents'].astype(int).to_dict()
    unit_to_base = {'ml': ('ml', 1), 'liter': ('ml', 1000), 'wh': ('wh', 1), 'kwh': ('wh', 1000), 'minute': ('minute', 1), 'hour': ('minute', 60)}
    df_usage['amount'] = df_usage['amount'].astype(int)

    def convert_to_base(row):
        unit = row['unit'].strip().casefold()
        if unit not in unit_to_base:
            raise ValueError(f'Unknown unit in usage: {unit}')
        (base_unit, factor) = unit_to_base[unit]
        return (row['resource'].strip(), base_unit, row['amount'] * factor)
    usage_tuples = []
    for (idx, row) in df_usage.iterrows():
        item = row['item_ref']
        (resource, base_unit, amount_base) = convert_to_base(row)
        usage_tuples.append((item, resource, base_unit, amount_base))
    usage = {}
    for (item, resource, base_unit, amount_base) in usage_tuples:
        key = (item, resource)
        if key in usage:
            usage[key] += amount_base
        else:
            usage[key] = amount_base
    df_capacity['amount'] = df_capacity['amount'].astype(int)

    def cap_convert_to_base(row):
        unit = row['unit'].strip().casefold()
        if unit not in unit_to_base:
            raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
        (base_unit, factor) = unit_to_base[unit]
        return (row['resource'].strip(), base_unit, row['amount'] * factor)
    cap_tuples = []
    for (idx, row) in df_capacity.iterrows():
        (resource, base_unit, amount_base) = cap_convert_to_base(row)
        cap_tuples.append((resource, base_unit, amount_base))
    resource_caps = {}
    for (resource, base_unit, amount_base) in cap_tuples:
        key = (resource, base_unit)
        if key in resource_caps:
            resource_caps[key] += amount_base
        else:
            resource_caps[key] = amount_base
    bundles = []
    bundle_bonus = {}
    for (idx, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items and b in items:
            bundle_id = (a, b)
            bundles.append(bundle_id)
            bundle_bonus[bundle_id] = int(row['bonus_cents'])
    incompat_pairs = []
    for (idx, row) in df_incompat.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in items and b in items:
            incompat_pairs.append((a, b))
    prereq_pairs = []
    for (idx, row) in df_requires.iterrows():
        i = row['item_ref']
        prereq = row['prerequisite_ref']
        if i in items and prereq in items:
            prereq_pairs.append((i, prereq))
    cat_to_items = {c: [] for c in categories}
    for i in items:
        c = item_to_category[i]
        if c in categories:
            cat_to_items[c].append(i)
    m = gp.Model('central_fresh_order')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    activation_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    cat_activation_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    bundle_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        m.addConstr(quantity_vars[i] >= min_lot[i] * activation_vars[i], name='')
        m.addConstr(quantity_vars[i] <= max_order[i] * activation_vars[i], name='')
    for c in categories:
        S_c = cat_to_items[c]
        if len(S_c) == 0:
            m.addConstr(cat_activation_vars[c] == 0, name='')
            continue
        m.addConstr(gp.quicksum((quantity_vars[i] for i in S_c)) >= min_cat_qty[c], name='')
        m.addConstr(gp.quicksum((quantity_vars[i] for i in S_c)) <= max_cat_qty[c], name='')
        for i in S_c:
            m.addConstr(activation_vars[i] <= cat_activation_vars[c], name='')
        m.addConstr(gp.quicksum((activation_vars[i] for i in S_c)) >= cat_activation_vars[c], name='')
    for ((resource, base_unit), cap) in resource_caps.items():
        expr = gp.LinExpr()
        for i in items:
            u = usage.get((i, resource), 0)
            expr.addTerms(u, quantity_vars[i])
        m.addConstr(expr <= cap, name='')
    for (a, b) in incompat_pairs:
        m.addConstr(activation_vars[a] + activation_vars[b] <= 1, name='')
    for (i, prereq) in prereq_pairs:
        m.addConstr(activation_vars[i] <= activation_vars[prereq], name='')
    for (a, b) in bundles:
        m.addConstr(bundle_vars[a, b] <= activation_vars[a], name='')
        m.addConstr(bundle_vars[a, b] <= activation_vars[b], name='')
        m.addConstr(bundle_vars[a, b] >= activation_vars[a] + activation_vars[b] - 1, name='')
    obj = gp.LinExpr()
    obj.add(gp.quicksum((unit_benefit[i] * quantity_vars[i] for i in items)))
    obj.add(-gp.quicksum((item_fee[i] * activation_vars[i] for i in items)))
    obj.add(-gp.quicksum((cat_fee[c] * cat_activation_vars[c] for c in categories)))
    obj.add(gp.quicksum((bundle_bonus[b] * bundle_vars[b] for b in bundles)))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()