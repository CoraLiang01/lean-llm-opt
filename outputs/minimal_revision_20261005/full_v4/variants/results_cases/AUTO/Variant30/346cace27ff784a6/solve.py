import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def filter_latest_valid(df, tenant, as_of_date):
        df = df[df['tenant'].str.casefold() == tenant.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df['_key'] = list(zip(df['table'], df['tenant'], df['record_id']))
        idx = df.groupby('_key')['revision'].idxmax()
        df_latest = df.loc[idx].copy()
        df_latest = df_latest[df_latest['action'].str.casefold() != 'delete']
        df_latest = df_latest.drop(columns=['_key'])
        return df_latest.reset_index(drop=True)
    as_of_date = '2026-03-12'
    tenant = 'NORTH'

    def get_table(table_name, dfs, tenant, as_of_date):
        result = []
        for df in dfs:
            if 'table' in df.columns:
                if (df['table'].str.casefold() == table_name.casefold()).any():
                    filtered = filter_latest_valid(df[df['table'].str.casefold() == table_name.casefold()], tenant, as_of_date)
                    if not filtered.empty:
                        result.append(filtered)
        if result:
            return pd.concat(result, ignore_index=True)
        else:
            return pd.DataFrame()
    item_df = pd.concat([get_table('item', dfs, tenant, as_of_date)], ignore_index=True)
    item_df = item_df[~item_df['item_ref'].isnull()]
    category_df = get_table('category', dfs, tenant, as_of_date)
    category_df = category_df[~category_df['category'].isnull()]
    item_fee_df = get_table('item_fee', dfs, tenant, as_of_date)
    item_fee_df = item_fee_df[~item_fee_df['item_ref'].isnull()]
    benefit_df = get_table('benefit', dfs, tenant, as_of_date)
    benefit_df = benefit_df[~benefit_df['item_ref'].isnull()]
    usage_df = get_table('usage', dfs, tenant, as_of_date)
    usage_df = usage_df[~usage_df['item_ref'].isnull()]
    capacity_ledger_df = get_table('capacity_ledger', dfs, tenant, as_of_date)
    bundle_df = get_table('bundle', dfs, tenant, as_of_date)
    bundle_df = bundle_df[~bundle_df['item_a'].isnull() & ~bundle_df['item_b'].isnull()]
    incompatible_df = get_table('incompatible', dfs, tenant, as_of_date)
    incompatible_df = incompatible_df[~incompatible_df['item_a'].isnull() & ~incompatible_df['item_b'].isnull()]
    requires_df = get_table('requires', dfs, tenant, as_of_date)
    requires_df = requires_df[~requires_df['item_ref'].isnull() & ~requires_df['prerequisite_ref'].isnull()]
    items = sorted(item_df['item_ref'].unique())
    categories = sorted(set(item_df['category'].dropna().unique()).union(set(category_df['category'].dropna().unique())))
    resources = sorted(set(usage_df['resource'].dropna().unique()).union(set(capacity_ledger_df['resource'].dropna().unique())))
    bundles = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
    incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompatible_df.iterrows()]
    requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
    benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    for i in items:
        if i not in benefit_per_item:
            benefit_per_item[i] = 0
    item_fee_map = {}
    for i in items:
        rows = item_fee_df[item_fee_df['item_ref'] == i]
        if not rows.empty:
            row = rows.loc[rows['revision'].idxmax()]
            item_fee_map[i] = float(row['activation_fee_cents'])
        else:
            item_fee_map[i] = 0.0
    category_fee_map = {}
    for g in categories:
        rows = category_df[category_df['category'] == g]
        if not rows.empty:
            row = rows.loc[rows['revision'].idxmax()]
            category_fee_map[g] = float(row['activation_fee_cents'])
        else:
            category_fee_map[g] = 0.0
    bundle_bonus_map = {}
    for (_, row) in bundle_df.iterrows():
        key = (row['item_a'], row['item_b'])
        bundle_bonus_map[key] = float(row['bonus_cents'])

    def convert_usage(amount, unit):
        if isinstance(unit, str):
            u = unit.strip().casefold()
            if u == 'ml':
                return amount / 1000.0
            elif u == 'liter':
                return amount
            elif u == 'minute':
                return amount / 60.0
            elif u == 'hour':
                return amount
            elif u == 'wh':
                return amount / 1000.0
            elif u == 'kwh':
                return amount
            else:
                return amount
        else:
            return amount
    usage_per_item_resource = {}
    for i in items:
        for r in resources:
            rows = usage_df[(usage_df['item_ref'] == i) & (usage_df['resource'] == r)]
            total = 0.0
            for (_, row) in rows.iterrows():
                amt = row['amount']
                unit = row['unit']
                total += convert_usage(amt, unit)
            usage_per_item_resource[i, r] = total
    capacity_per_resource = {}
    for r in resources:
        rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
        total = 0.0
        for (_, row) in rows.iterrows():
            amt = row['amount']
            unit = row['unit']
            total += convert_usage(amt, unit)
        capacity_per_resource[r] = total
    category_min_qty = {}
    category_max_qty = {}
    for g in categories:
        rows = category_df[category_df['category'] == g]
        if not rows.empty:
            row = rows.loc[rows['revision'].idxmax()]
            category_min_qty[g] = int(row['minimum_quantity'])
            category_max_qty[g] = int(row['maximum_quantity'])
        else:
            category_min_qty[g] = 0
            category_max_qty[g] = 0
    item_min_lot = {}
    item_max_order = {}
    item_authorized = {}
    item_category = {}
    for (_, row) in item_df.iterrows():
        i = row['item_ref']
        item_min_lot[i] = int(row['minimum_lot'])
        item_max_order[i] = int(row['maximum_order'])
        item_authorized[i] = int(row['authorized'])
        item_category[i] = row['category']
    m = gp.Model('vehicle_dealer_replenishment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if item_authorized[i] == 0:
            m.addConstr(x[i] == 0, name=f'auth_{i}')
            m.addConstr(y[i] == 0, name=f'authy_{i}')
        else:
            m.addConstr(x[i] >= item_min_lot[i] * y[i], name=f'minlot_{i}')
            m.addConstr(x[i] <= item_max_order[i] * y[i], name=f'maxorder_{i}')
            m.addConstr(y[i] <= 1, name=f'ybin_{i}')
            m.addConstr(x[i] >= 0, name=f'xnonneg_{i}')
            m.addGenConstrIndicator(y[i], True, x[i] >= 1, name=f'yxlink_{i}')
            m.addGenConstrIndicator(y[i], False, x[i] == 0, name=f'yxlink0_{i}')
    for g in categories:
        items_in_g = [i for i in items if item_category[i] == g]
        if items_in_g:
            m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= category_min_qty[g] * z[g], name=f'catmin_{g}')
            m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= category_max_qty[g] * z[g], name=f'catmax_{g}')
            for i in items_in_g:
                m.addConstr(z[g] >= y[i], name=f'zact_{g}_{i}')
            m.addConstr(z[g] <= gp.quicksum((y[i] for i in items_in_g)), name=f'zact2_{g}')
        else:
            m.addConstr(z[g] == 0, name=f'zempty_{g}')
    for r in resources:
        expr = gp.quicksum((usage_per_item_resource[i, r] * x[i] for i in items))
        m.addConstr(expr <= capacity_per_resource[r], name=f'res_{r}')
    for (i, j) in incompat_pairs:
        if i in items and j in items:
            m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
    for (i, k) in requires_pairs:
        if i in items and k in items:
            m.addConstr(y[i] <= y[k], name=f'reqbin_{i}_{k}')
            m.addConstr(x[k] >= y[i], name=f'reqqty_{i}_{k}')
    for (ia, ib) in bundles:
        if ia in items and ib in items:
            m.addConstr(w[ia, ib] <= y[ia], name=f'bundle1_{ia}_{ib}')
            m.addConstr(w[ia, ib] <= y[ib], name=f'bundle2_{ia}_{ib}')
            m.addConstr(w[ia, ib] >= y[ia] + y[ib] - 1, name=f'bundle3_{ia}_{ib}')
        else:
            m.addConstr(w[ia, ib] == 0, name=f'bundle0_{ia}_{ib}')
    obj = gp.quicksum((benefit_per_item[i] * x[i] for i in items)) - gp.quicksum((item_fee_map[i] * y[i] for i in items)) - gp.quicksum((category_fee_map[g] * z[g] for g in categories)) + gp.quicksum((bundle_bonus_map[b] * w[b] for b in bundles))
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