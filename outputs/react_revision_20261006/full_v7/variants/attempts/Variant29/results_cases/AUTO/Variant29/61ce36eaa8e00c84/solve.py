import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
    df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
    df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
    df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
    df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
    df_incompatible = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
    df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
    df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
    df_item['authorized'] = df_item['authorized'].astype(int)
    authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].tolist()
    item_to_category = df_item.set_index('item_ref')['category'].to_dict()
    df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
    df_item['maximum_order'] = df_item['maximum_order'].astype(int)
    df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
    df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
    minimum_lot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
    maximum_order = df_item.set_index('item_ref')['maximum_order'].to_dict()
    unit_benefit_cents = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
    item_fee_cents = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
    categories = df_category['category'].tolist()
    df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
    df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
    df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
    minimum_quantity = df_category.set_index('category')['minimum_quantity'].to_dict()
    maximum_quantity = df_category.set_index('category')['maximum_quantity'].to_dict()
    activation_fee_cents = df_category.set_index('category')['activation_fee_cents'].to_dict()
    df_usage = df_usage[df_usage['item_ref'].isin(authorized_items)].copy()
    df_usage['amount'] = df_usage['amount'].astype(int)

    def normalize_usage_unit(row):
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if u == 'kwh':
            return (amt * 1000, 'wh')
        elif u == 'wh':
            return (amt, 'wh')
        elif u == 'hour':
            return (amt * 60, 'minute')
        elif u == 'minute':
            return (amt, 'minute')
        elif u == 'liter':
            return (amt * 1000, 'ml')
        elif u == 'ml':
            return (amt, 'ml')
        else:
            raise ValueError(f'Unknown unit: {u}')
    df_usage[['norm_amount', 'norm_unit']] = df_usage.apply(normalize_usage_unit, axis=1, result_type='expand')
    resource_set = sorted(df_usage['resource'].unique())
    resource_unit = {}
    for r in resource_set:
        units = df_usage[df_usage['resource'] == r]['norm_unit'].unique()
        if len(units) != 1:
            raise ValueError(f'Resource {r} has inconsistent units: {units}')
        resource_unit[r] = units[0]
    usage_dict = {}
    for (_, row) in df_usage.iterrows():
        usage_dict[row['item_ref'], row['resource']] = row['norm_amount']
    df_capacity['amount'] = df_capacity['amount'].astype(int)

    def normalize_capacity_unit(row):
        amt = row['amount']
        u = row['unit'].strip().casefold()
        if u == 'kwh':
            return (amt * 1000, 'wh')
        elif u == 'wh':
            return (amt, 'wh')
        elif u == 'hour':
            return (amt * 60, 'minute')
        elif u == 'minute':
            return (amt, 'minute')
        elif u == 'liter':
            return (amt * 1000, 'ml')
        elif u == 'ml':
            return (amt, 'ml')
        else:
            raise ValueError(f'Unknown unit: {u}')
    df_capacity[['norm_amount', 'norm_unit']] = df_capacity.apply(normalize_capacity_unit, axis=1, result_type='expand')
    capacity_dict = {}
    for r in resource_set:
        unit = resource_unit[r]
        total = df_capacity[(df_capacity['resource'] == r) & (df_capacity['norm_unit'] == unit)]['norm_amount'].sum()
        capacity_dict[r] = total
    incompatible_pairs = []
    for (_, row) in df_incompatible.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in authorized_items and b in authorized_items:
            incompatible_pairs.append((a, b))
    prerequisite_pairs = []
    for (_, row) in df_requires.iterrows():
        i = row['item_ref']
        j = row['prerequisite_ref']
        if i in authorized_items and j in authorized_items:
            prerequisite_pairs.append((i, j))
    df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
    bundle_tuples = []
    bundle_bonus = {}
    for (_, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in authorized_items and b in authorized_items:
            bundle_tuples.append((a, b))
            bundle_bonus[a, b] = row['bonus_cents']
    m = gp.Model('BakeryOrder')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(authorized_items, lb=0, ub={i: maximum_order[i] for i in authorized_items}, vtype=gp.GRB.INTEGER, name='')
    activation_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
    category_activation_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    bundle_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
    for i in authorized_items:
        m.addConstr(quantity_vars[i] >= minimum_lot[i] * activation_vars[i], name='minlot_' + i)
        m.addConstr(quantity_vars[i] <= maximum_order[i] * activation_vars[i], name='maxorder_' + i)
        m.addConstr(quantity_vars[i] >= 0, name='nonneg_' + i)
        m.addConstr(quantity_vars[i] <= maximum_order[i], name='ub_' + i)
    for c in categories:
        items_in_c = [i for i in authorized_items if item_to_category[i] == c]
        if items_in_c:
            m.addConstr(gp.quicksum((activation_vars[i] for i in items_in_c)) <= len(items_in_c) * category_activation_vars[c], name='catact1_' + c)
            m.addConstr(gp.quicksum((activation_vars[i] for i in items_in_c)) >= category_activation_vars[c], name='catact2_' + c)
        else:
            m.addConstr(category_activation_vars[c] == 0, name='catact0_' + c)
    for c in categories:
        items_in_c = [i for i in authorized_items if item_to_category[i] == c]
        if items_in_c:
            m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) >= minimum_quantity[c], name='catmin_' + c)
            m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) <= maximum_quantity[c], name='catmax_' + c)
        else:
            m.addConstr(gp.quicksum((quantity_vars[i] for i in authorized_items if item_to_category[i] == c)) == 0, name='catzero_' + c)
    for r in resource_set:
        expr = gp.LinExpr()
        for i in authorized_items:
            amt = usage_dict.get((i, r), 0)
            expr += amt * quantity_vars[i]
        m.addConstr(expr <= capacity_dict[r], name='res_' + r)
    for (i, j) in incompatible_pairs:
        m.addConstr(activation_vars[i] + activation_vars[j] <= 1, name='incomp_' + i + '_' + j)
    for (i, j) in prerequisite_pairs:
        m.addConstr(activation_vars[i] <= activation_vars[j], name='prereq_' + i + '_' + j)
    for (a, b) in bundle_tuples:
        m.addConstr(bundle_vars[a, b] <= activation_vars[a], name='bundle1_' + a + '_' + b)
        m.addConstr(bundle_vars[a, b] <= activation_vars[b], name='bundle2_' + a + '_' + b)
        m.addConstr(bundle_vars[a, b] >= activation_vars[a] + activation_vars[b] - 1, name='bundle3_' + a + '_' + b)
    obj = gp.LinExpr()
    obj += gp.quicksum((unit_benefit_cents[i] * quantity_vars[i] for i in authorized_items))
    obj -= gp.quicksum((item_fee_cents[i] * activation_vars[i] for i in authorized_items))
    obj -= gp.quicksum((activation_fee_cents[c] * category_activation_vars[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[a, b] * bundle_vars[a, b] for (a, b) in bundle_tuples))
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