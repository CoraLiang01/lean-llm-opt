import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def select_latest(df, tenant_col, table_col, date_col, action_col, key_cols, cutoff_date):
        df = df[df[tenant_col].str.casefold() == 'north']
        df = df[df[date_col] <= cutoff_date]
        df = df.sort_values(key_cols + ['revision'], ascending=[True] * len(key_cols) + [False])
        df = df[df[action_col].str.casefold() != 'delete']
        df = df.drop_duplicates(key_cols, keep='first')
        return df
    cutoff_date = '2026-03-12'
    table_map = {'requires': [0, 23], 'identity': [1, 21], 'item_fee': [2, 16], 'bundle': [3, 10], 'benefit': [4, 11, 14], 'usage': [5, 18, 22], 'incompatible': [6, 13], 'capacity_ledger': [7, 19], 'market': [8, 20], 'item': [9, 12], 'category': [15, 17]}

    def concat_and_select(table_name, key_cols, date_col='effective_date', action_col='action'):
        idxs = table_map[table_name]
        df = pd.concat([dfs[i] for i in idxs], ignore_index=True)
        return select_latest(df, 'tenant', 'table', date_col, action_col, key_cols, cutoff_date)
    item_df = concat_and_select('item', ['table', 'record_id'])
    item_df = item_df[item_df['item_ref'].notnull()]
    items = item_df['item_ref'].unique().tolist()
    item_to_cat = item_df.set_index('item_ref')['category'].to_dict()
    item_to_auth = item_df.set_index('item_ref')['authorized'].to_dict()
    item_to_minlot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
    item_to_maxorder = item_df.set_index('item_ref')['maximum_order'].to_dict()
    authorized_items = [i for i in items if float(item_to_auth[i]) > 0]
    cat_df = concat_and_select('category', ['table', 'record_id'])
    cat_df = cat_df[cat_df['category'].notnull()]
    categories = cat_df['category'].unique().tolist()
    cat_to_minqty = cat_df.set_index('category')['minimum_quantity'].to_dict()
    cat_to_maxqty = cat_df.set_index('category')['maximum_quantity'].to_dict()
    cat_to_fee = cat_df.set_index('category')['activation_fee_cents'].to_dict()
    cat_to_items = {g: [] for g in categories}
    for i in authorized_items:
        g = item_to_cat[i]
        if g in cat_to_items:
            cat_to_items[g].append(i)
        else:
            cat_to_items[g] = [i]
    item_fee_df = concat_and_select('item_fee', ['table', 'record_id'])
    item_fee_df = item_fee_df[item_fee_df['item_ref'].notnull()]
    item_to_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
    benefit_df = pd.concat([concat_and_select('benefit', ['table', 'record_id']) for _ in [0]], ignore_index=True)
    benefit_df = benefit_df[benefit_df['item_ref'].notnull()]
    benefit_df = benefit_df[benefit_df['item_ref'].isin(authorized_items)]
    benefit_by_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    for i in authorized_items:
        if i not in benefit_by_item:
            raise ValueError(f'Missing benefit for item {i}')
    bundle_df = concat_and_select('bundle', ['table', 'record_id'])
    bundle_df = bundle_df[bundle_df['item_a'].notnull() & bundle_df['item_b'].notnull()]
    bundle_df = bundle_df[bundle_df['item_a'].isin(authorized_items) & bundle_df['item_b'].isin(authorized_items)]
    bundle_keys = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
    bundle_bonus = {(row['item_a'], row['item_b']): float(row['bonus_cents']) for (_, row) in bundle_df.iterrows()}
    incomp_df = concat_and_select('incompatible', ['table', 'record_id'])
    incomp_df = incomp_df[incomp_df['item_a'].notnull() & incomp_df['item_b'].notnull()]
    incomp_pairs = [(row['item_a'], row['item_b']) for (_, row) in incomp_df.iterrows() if row['item_a'] in authorized_items and row['item_b'] in authorized_items]
    req_df = concat_and_select('requires', ['table', 'record_id'])
    req_df = req_df[req_df['item_ref'].notnull() & req_df['prerequisite_ref'].notnull()]
    req_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in req_df.iterrows() if row['item_ref'] in authorized_items and row['prerequisite_ref'] in authorized_items]
    usage_df = pd.concat([concat_and_select('usage', ['table', 'record_id']) for _ in [0]], ignore_index=True)
    usage_df = usage_df[usage_df['item_ref'].notnull() & usage_df['resource'].notnull() & usage_df['amount'].notnull() & usage_df['unit'].notnull()]
    usage_df = usage_df[usage_df['item_ref'].isin(authorized_items)]

    def usage_to_base(row):
        amt = float(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'liter':
            return amt * 1000.0
        elif unit == 'ml':
            return amt
        elif unit == 'hour':
            return amt * 60.0
        elif unit == 'minute':
            return amt
        elif unit == 'kwh':
            return amt * 1000.0
        elif unit == 'wh':
            return amt
        else:
            raise ValueError(f'Unknown unit in usage: {unit}')
    usage_df['amount_base'] = usage_df.apply(usage_to_base, axis=1)
    usage_by_item_resource = {}
    for (_, row) in usage_df.iterrows():
        key = (row['item_ref'], row['resource'])
        usage_by_item_resource[key] = float(row['amount_base'])
    resources = sorted(set((row['resource'] for (_, row) in usage_df.iterrows())))
    cap_df = pd.concat([concat_and_select('capacity_ledger', ['table', 'record_id']) for _ in [0]], ignore_index=True)
    cap_df = cap_df[cap_df['resource'].notnull() & cap_df['amount'].notnull() & cap_df['unit'].notnull()]
    cap_df = cap_df[cap_df['resource'].isin(resources)]

    def cap_to_base(row):
        amt = float(row['amount'])
        unit = str(row['unit']).strip().casefold()
        if unit == 'liter':
            return amt * 1000.0
        elif unit == 'ml':
            return amt
        elif unit == 'hour':
            return amt * 60.0
        elif unit == 'minute':
            return amt
        elif unit == 'kwh':
            return amt * 1000.0
        elif unit == 'wh':
            return amt
        else:
            raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
    cap_df['amount_base'] = cap_df.apply(cap_to_base, axis=1)
    cap_by_resource = cap_df.groupby('resource')['amount_base'].sum().to_dict()
    for r in resources:
        if r not in cap_by_resource:
            raise ValueError(f'Missing capacity for resource {r}')
    m = gp.Model('inventory_replenishment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
    bigM = max([int(item_to_maxorder[i]) for i in authorized_items] + [1])
    for i in authorized_items:
        m.addConstr(x[i] <= bigM * y[i], name=f'link_x_y_ub_{i}')
        m.addConstr(x[i] >= y[i], name=f'link_x_y_lb_{i}')
    for i in authorized_items:
        minlot = int(item_to_minlot[i])
        maxorder = int(item_to_maxorder[i])
        m.addConstr(x[i] >= minlot * y[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= maxorder * y[i], name=f'maxorder_{i}')
    for r in resources:
        expr = gp.LinExpr()
        for i in authorized_items:
            amt = usage_by_item_resource.get((i, r), 0.0)
            expr += amt * x[i]
        m.addConstr(expr <= cap_by_resource[r], name=f'resource_{r}')
    for g in categories:
        items_in_g = cat_to_items.get(g, [])
        if not items_in_g:
            m.addConstr(z[g] == 0, name=f'cat_empty_{g}')
            continue
        sum_x = gp.quicksum((x[i] for i in items_in_g))
        minq = int(cat_to_minqty[g])
        maxq = int(cat_to_maxqty[g])
        m.addConstr(sum_x >= minq * z[g], name=f'cat_min_{g}')
        m.addConstr(sum_x <= maxq * z[g], name=f'cat_max_{g}')
        m.addConstr(sum_x <= bigM * z[g], name=f'cat_link_{g}')
    for (i, j) in incomp_pairs:
        m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, j) in req_pairs:
        m.addConstr(y[i] <= y[j], name=f'requires_{i}_{j}')
    for (i, j) in bundle_keys:
        m.addConstr(w[i, j] <= y[i], name=f'bundle_w_yi_{i}_{j}')
        m.addConstr(w[i, j] <= y[j], name=f'bundle_w_yj_{i}_{j}')
        m.addConstr(w[i, j] >= y[i] + y[j] - 1, name=f'bundle_w_lb_{i}_{j}')
    obj = gp.LinExpr()
    for i in authorized_items:
        obj += benefit_by_item[i] * x[i]
    for i in authorized_items:
        fee = float(item_to_fee.get(i, 0.0))
        obj -= fee * y[i]
    for g in categories:
        fee = float(cat_to_fee.get(g, 0.0))
        obj -= fee * z[g]
    for b in bundle_keys:
        obj += bundle_bonus[b] * w[b]
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