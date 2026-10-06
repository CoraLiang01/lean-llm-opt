import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def select_latest(df, date_col='effective_date'):
        df = df[df[date_col] <= '2026-03-12']
        idx = df.groupby(['tenant', 'table', 'record_id'])['revision'].transform('max') == df['revision']
        df = df[idx]
        if 'action' in df.columns:
            df = df[df['action'].str.casefold() != 'delete']
        return df
    table_map = {'requires': [0, 23], 'identity': [1, 21], 'item_fee': [2, 16], 'bundle': [3, 10], 'benefit': [4, 11, 14], 'usage': [5, 18, 22], 'incompatible': [6, 13], 'capacity_ledger': [7, 19], 'market': [8, 20], 'item': [9, 12], 'category': [15, 17]}

    def get_table(table, date_col='effective_date'):
        idxs = table_map[table]
        df = pd.concat([dfs[i] for i in idxs], ignore_index=True)
        df = df[df['tenant'].str.casefold() == 'north']
        df = select_latest(df, date_col=date_col)
        return df
    item_df = get_table('item')
    item_df = item_df[item_df['item_ref'].notnull()]
    items = sorted(item_df['item_ref'].unique())
    item_category = item_df.set_index('item_ref')['category'].to_dict()
    item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
    item_minlot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
    item_maxorder = item_df.set_index('item_ref')['maximum_order'].to_dict()
    category_df = get_table('category')
    category_df = category_df[category_df['category'].notnull()]
    categories = sorted(category_df['category'].unique())
    cat_minqty = category_df.set_index('category')['minimum_quantity'].to_dict()
    cat_maxqty = category_df.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
    benefit_df = pd.concat([dfs[i] for i in table_map['benefit']], ignore_index=True)
    benefit_df = benefit_df[benefit_df['tenant'].str.casefold() == 'north']
    benefit_df = select_latest(benefit_df)
    benefit_df = benefit_df[benefit_df['item_ref'].notnull()]
    item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    for i in items:
        if i not in item_benefit:
            item_benefit[i] = 0
    item_fee_df = pd.concat([dfs[i] for i in table_map['item_fee']], ignore_index=True)
    item_fee_df = item_fee_df[item_fee_df['tenant'].str.casefold() == 'north']
    item_fee_df = select_latest(item_fee_df)
    item_fee_df = item_fee_df[item_fee_df['item_ref'].notnull()]
    item_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
    for i in items:
        if i not in item_fee:
            item_fee[i] = 0
    bundle_df = pd.concat([dfs[i] for i in table_map['bundle']], ignore_index=True)
    bundle_df = bundle_df[bundle_df['tenant'].str.casefold() == 'north']
    bundle_df = select_latest(bundle_df)
    bundle_df = bundle_df[bundle_df['item_a'].notnull() & bundle_df['item_b'].notnull()]
    bundle_df = bundle_df[bundle_df['item_a'].isin(items) & bundle_df['item_b'].isin(items)]
    bundles = list(bundle_df[['item_a', 'item_b']].itertuples(index=False, name=None))
    bundle_bonus = {(row.item_a, row.item_b): row.bonus_cents for row in bundle_df.itertuples(index=False)}
    incompatible_df = pd.concat([dfs[i] for i in table_map['incompatible']], ignore_index=True)
    incompatible_df = incompatible_df[incompatible_df['tenant'].str.casefold() == 'north']
    incompatible_df = select_latest(incompatible_df)
    incompatible_df = incompatible_df[incompatible_df['item_a'].notnull() & incompatible_df['item_b'].notnull()]
    incompatible_df = incompatible_df[incompatible_df['item_a'].isin(items) & incompatible_df['item_b'].isin(items)]
    incompatible_pairs = set()
    for row in incompatible_df.itertuples(index=False):
        incompatible_pairs.add((row.item_a, row.item_b))
        incompatible_pairs.add((row.item_b, row.item_a))
    requires_df = pd.concat([dfs[i] for i in table_map['requires']], ignore_index=True)
    requires_df = requires_df[requires_df['tenant'].str.casefold() == 'north']
    requires_df = select_latest(requires_df)
    requires_df = requires_df[requires_df['item_ref'].notnull() & requires_df['prerequisite_ref'].notnull()]
    requires_df = requires_df[requires_df['item_ref'].isin(items) & requires_df['prerequisite_ref'].isin(items)]
    requires_pairs = set(((row.item_ref, row.prerequisite_ref) for row in requires_df.itertuples(index=False)))
    usage_df = pd.concat([dfs[i] for i in table_map['usage']], ignore_index=True)
    usage_df = usage_df[usage_df['tenant'].str.casefold() == 'north']
    usage_df = select_latest(usage_df)
    usage_df = usage_df[usage_df['item_ref'].notnull() & usage_df['resource'].notnull() & usage_df['unit'].notnull()]
    usage_df = usage_df[usage_df['item_ref'].isin(items)]
    usage = {}
    for row in usage_df.itertuples(index=False):
        usage[row.item_ref, row.resource] = (row.amount, row.unit)
    cap_df = pd.concat([dfs[i] for i in table_map['capacity_ledger']], ignore_index=True)
    cap_df = cap_df[cap_df['tenant'].str.casefold() == 'north']
    cap_df = select_latest(cap_df)
    cap_df = cap_df[cap_df['resource'].notnull() & cap_df['unit'].notnull()]
    cap_df['amount'] = pd.to_numeric(cap_df['amount'], errors='coerce').fillna(0)
    resource_caps = {}
    for (resource, unit), group in cap_df.groupby(['resource', 'unit']):
        resource_caps[resource, unit] = group['amount'].sum()
    resource_units = {}
    for resource, unit in resource_caps:
        resource_units[resource] = unit
    category_items = {g: [] for g in categories}
    for i in items:
        g = item_category[i]
        if g in category_items:
            category_items[g].append(i)
        else:
            category_items[g] = [i]
    m = gp.Model('NORTH_Inventory_Replenishment')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        auth = item_authorized[i]
        minlot = int(item_minlot[i])
        maxorder = int(item_maxorder[i])
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'unauth_{i}')
            m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
        else:
            m.addConstr(x[i] >= minlot * y[i], name=f'minlot_{i}')
            m.addConstr(x[i] <= maxorder * y[i], name=f'maxorder_{i}')
            m.addConstr(x[i] <= maxorder * y[i], name=f'link_yx_{i}')
    for g in categories:
        minq = int(cat_minqty[g])
        maxq = int(cat_maxqty[g])
        items_in_g = category_items.get(g, [])
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= minq * z[g], name=f'cat_min_{g}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= maxq * z[g], name=f'cat_max_{g}')
        for i in items_in_g:
            m.addConstr(y[i] <= z[g], name=f'cat_link_{g}_{i}')
    for resource in resource_units:
        cap_unit = resource_units[resource]
        usage_terms = []
        for i in items:
            if (i, resource) in usage:
                amt, unit = usage[i, resource]
                factor = 1.0
                if unit == cap_unit:
                    factor = 1.0
                elif unit == 'liter' and cap_unit == 'ml':
                    factor = 1000.0
                elif unit == 'ml' and cap_unit == 'liter':
                    factor = 1.0 / 1000.0
                elif unit == 'hour' and cap_unit == 'minute':
                    factor = 60.0
                elif unit == 'minute' and cap_unit == 'hour':
                    factor = 1.0 / 60.0
                elif unit == 'kwh' and cap_unit == 'wh':
                    factor = 1000.0
                elif unit == 'wh' and cap_unit == 'kwh':
                    factor = 1.0 / 1000.0
                else:
                    raise ValueError(f'Unsupported unit conversion: {unit} to {cap_unit} for resource {resource}')
                usage_terms.append((float(amt) * factor, x[i]))
        if usage_terms:
            m.addConstr(gp.quicksum((coeff * var for coeff, var in usage_terms)) <= resource_caps[resource, cap_unit], name=f'rescap_{resource}')
    for i, j in incompatible_pairs:
        if i in items and j in items:
            m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for i, k in requires_pairs:
        if i in items and k in items:
            m.addConstr(y[i] <= y[k], name=f'req_{i}_{k}')
    for a, b in bundles:
        m.addConstr(w[a, b] <= y[a], name=f'bundle1_{a}_{b}')
        m.addConstr(w[a, b] <= y[b], name=f'bundle2_{a}_{b}')
        m.addConstr(w[a, b] >= y[a] + y[b] - 1, name=f'bundle3_{a}_{b}')
    obj = gp.quicksum((item_benefit[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
    obj -= gp.quicksum((cat_fee[g] * z[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[a, b] * w[a, b] for a, b in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()