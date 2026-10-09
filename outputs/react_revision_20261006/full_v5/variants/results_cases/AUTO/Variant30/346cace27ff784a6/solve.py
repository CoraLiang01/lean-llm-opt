import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def select_latest(df, cutoff_date):
        df = df[df['effective_date'] <= cutoff_date].copy()
        df['revision'] = pd.to_numeric(df['revision'], errors='coerce')
        df = df.sort_values(['tenant', 'table', 'record_id', 'revision', 'effective_date'], ascending=[True, True, True, False, False])
        df = df.drop_duplicates(['tenant', 'table', 'record_id'], keep='first')
        if 'action' in df.columns:
            df = df[df['action'].str.casefold() != 'delete']
        return df
    cutoff_date = '2026-03-12'
    table_map = {}
    for (idx, df) in enumerate(dfs):
        if 'table' in df.columns:
            for t in df['table'].unique():
                table_map.setdefault(t.casefold(), []).append(idx)

    def get_table(table_name):
        idxs = table_map.get(table_name.casefold(), [])
        dfs_table = []
        for idx in idxs:
            df = dfs[idx]
            if 'tenant' in df.columns:
                df = df[df['tenant'].str.casefold() == 'north']
            if 'table' in df.columns:
                df = df[df['table'].str.casefold() == table_name.casefold()]
            dfs_table.append(df)
        if dfs_table:
            dfcat = pd.concat(dfs_table, axis=0, ignore_index=True)
            return select_latest(dfcat, cutoff_date)
        else:
            return pd.DataFrame()
    item_df = pd.concat([get_table('item')], axis=0, ignore_index=True)
    item_df['item_ref'] = item_df['item_ref'].astype(str)
    item_df['category'] = item_df['category'].astype(str)
    item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='coerce').fillna(0).astype(int)
    item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'], errors='coerce').fillna(0).astype(int)
    item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'], errors='coerce').fillna(0).astype(int)
    eligible_items = item_df[item_df['authorized'] > 0]['item_ref'].unique()
    all_items = item_df['item_ref'].unique()
    item_to_cat = dict(zip(item_df['item_ref'], item_df['category']))
    cat_df = pd.concat([get_table('category')], axis=0, ignore_index=True)
    cat_df['category'] = cat_df['category'].astype(str)
    cat_df['minimum_quantity'] = pd.to_numeric(cat_df['minimum_quantity'], errors='coerce').fillna(0).astype(int)
    cat_df['maximum_quantity'] = pd.to_numeric(cat_df['maximum_quantity'], errors='coerce').fillna(0).astype(int)
    cat_df['activation_fee_cents'] = pd.to_numeric(cat_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
    categories = cat_df['category'].unique()
    cat_min = dict(zip(cat_df['category'], cat_df['minimum_quantity']))
    cat_max = dict(zip(cat_df['category'], cat_df['maximum_quantity']))
    cat_fee = dict(zip(cat_df['category'], cat_df['activation_fee_cents']))
    item_fee_df = pd.concat([get_table('item_fee')], axis=0, ignore_index=True)
    item_fee_df['item_ref'] = item_fee_df['item_ref'].astype(str)
    item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
    item_fee = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
    benefit_df = pd.concat([get_table('benefit')], axis=0, ignore_index=True)
    benefit_df['item_ref'] = benefit_df['item_ref'].astype(str)
    benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='coerce').fillna(0).astype(int)
    benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    usage_df = pd.concat([get_table('usage')], axis=0, ignore_index=True)
    usage_df['item_ref'] = usage_df['item_ref'].astype(str)
    usage_df['resource'] = usage_df['resource'].astype(str)
    usage_df['amount'] = pd.to_numeric(usage_df['amount'], errors='coerce').fillna(0)
    usage_df['unit'] = usage_df['unit'].astype(str)
    cap_df = pd.concat([get_table('capacity_ledger')], axis=0, ignore_index=True)
    cap_df['resource'] = cap_df['resource'].astype(str)
    cap_df['amount'] = pd.to_numeric(cap_df['amount'], errors='coerce').fillna(0)
    cap_df['unit'] = cap_df['unit'].astype(str)
    cap_df = cap_df[cap_df['resource'].notnull() & cap_df['amount'].notnull()]
    incomp_df = pd.concat([get_table('incompatible')], axis=0, ignore_index=True)
    incomp_df['item_a'] = incomp_df['item_a'].astype(str)
    incomp_df['item_b'] = incomp_df['item_b'].astype(str)
    incomp_pairs = set()
    for (_, row) in incomp_df.iterrows():
        (a, b) = (row['item_a'], row['item_b'])
        if pd.notnull(a) and pd.notnull(b):
            incomp_pairs.add(tuple(sorted((a, b))))
    req_df = pd.concat([get_table('requires')], axis=0, ignore_index=True)
    req_df['item_ref'] = req_df['item_ref'].astype(str)
    req_df['prerequisite_ref'] = req_df['prerequisite_ref'].astype(str)
    req_pairs = set()
    for (_, row) in req_df.iterrows():
        (i, j) = (row['item_ref'], row['prerequisite_ref'])
        if pd.notnull(i) and pd.notnull(j):
            req_pairs.add((i, j))
    bundle_df = pd.concat([get_table('bundle')], axis=0, ignore_index=True)
    bundle_df['item_a'] = bundle_df['item_a'].astype(str)
    bundle_df['item_b'] = bundle_df['item_b'].astype(str)
    bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'], errors='coerce').fillna(0).astype(int)
    bundle_pairs = []
    bundle_bonus = {}
    for (_, row) in bundle_df.iterrows():
        (a, b) = (row['item_a'], row['item_b'])
        if pd.notnull(a) and pd.notnull(b):
            key = tuple(sorted((a, b)))
            bundle_pairs.append(key)
            bundle_bonus[key] = row['bonus_cents']
    I = sorted(set(eligible_items))
    I_all = sorted(set(item_df['item_ref']))
    G = sorted(set(item_df[item_df['item_ref'].isin(I)]['category']))
    resources_usage = set(usage_df['resource'].unique())
    resources_cap = set(cap_df['resource'].unique())
    R = sorted(resources_usage | resources_cap)
    B = [b for b in bundle_pairs if b[0] in I and b[1] in I]
    INCOMP = [p for p in incomp_pairs if p[0] in I and p[1] in I]
    REQ = [p for p in req_pairs if p[0] in I and p[1] in I]
    for i in I:
        if i not in benefit_per_item:
            raise ValueError(f'Missing per-unit benefit for item {i}')
    benefit = {i: benefit_per_item[i] for i in I}
    item_fee_param = {i: item_fee.get(i, 0) for i in I}
    for g in G:
        if g not in cat_fee:
            raise ValueError(f'Missing activation fee for category {g}')
    cat_fee_param = {g: cat_fee[g] for g in G}
    min_lot = {}
    max_order = {}
    for i in I:
        row = item_df[item_df['item_ref'] == i].iloc[0]
        min_lot[i] = int(row['minimum_lot'])
        max_order[i] = int(row['maximum_order'])
    cat_min_param = {g: cat_min[g] for g in G}
    cat_max_param = {g: cat_max[g] for g in G}

    def to_base_unit(amount, unit):
        if unit.casefold() == 'ml':
            return amount
        elif unit.casefold() == 'liter':
            return amount * 1000
        elif unit.casefold() == 'minute':
            return amount
        elif unit.casefold() == 'hour':
            return amount * 60
        elif unit.casefold() == 'wh':
            return amount
        elif unit.casefold() == 'kwh':
            return amount * 1000
        else:
            return amount
    usage_per_item_resource = {}
    for i in I:
        for r in R:
            rows = usage_df[(usage_df['item_ref'] == i) & (usage_df['resource'] == r)]
            total = 0.0
            for (_, row) in rows.iterrows():
                amt = row['amount']
                unit = row['unit']
                total += to_base_unit(amt, unit)
            usage_per_item_resource[i, r] = total
    cap_per_resource = {}
    for r in R:
        rows = cap_df[cap_df['resource'] == r]
        total = 0.0
        for (_, row) in rows.iterrows():
            amt = row['amount']
            unit = row['unit']
            total += to_base_unit(amt, unit)
        cap_per_resource[r] = total
    m = gp.Model('inventory_replenishment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(G, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(B, vtype=gp.GRB.BINARY, name='')
    obj = gp.LinExpr()
    obj += gp.quicksum((benefit[i] * x[i] for i in I))
    obj -= gp.quicksum((item_fee_param[i] * y[i] for i in I))
    obj -= gp.quicksum((cat_fee_param[g] * z[g] for g in G))
    obj += gp.quicksum((bundle_bonus[b] * w[b] for b in B))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    for i in set(I_all) - set(I):
        m.addVar(lb=0, ub=0, vtype=gp.GRB.INTEGER, name=f'x_{i}')
        m.addVar(lb=0, ub=0, vtype=gp.GRB.BINARY, name=f'y_{i}')
    for i in I:
        m.addConstr(x[i] >= min_lot[i] * y[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= max_order[i] * y[i], name=f'maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
        m.addConstr(x[i] <= max_order[i], name=f'ub_{i}')
        m.addGenConstrIndicator(y[i], True, x[i] >= 1, name=f'yxlink_{i}')
        m.addGenConstrIndicator(y[i], False, x[i] == 0, name=f'yxlink0_{i}')
    for g in G:
        items_in_g = [i for i in I if item_to_cat[i] == g]
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= cat_min_param[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= cat_max_param[g], name=f'catmax_{g}')
        for i in items_in_g:
            m.addConstr(z[g] >= y[i], name=f'zg_{g}_{i}')
        m.addConstr(z[g] <= gp.quicksum((y[i] for i in items_in_g)), name=f'zg_sum_{g}')
    for r in R:
        expr = gp.LinExpr()
        for i in I:
            expr += usage_per_item_resource.get((i, r), 0.0) * x[i]
        m.addConstr(expr <= cap_per_resource[r], name=f'res_cap_{r}')
    for (i, j) in INCOMP:
        m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, j) in REQ:
        if i in I and j in I:
            m.addConstr(y[i] <= y[j], name=f'req_y_{i}_{j}')
            m.addGenConstrIndicator(y[i], True, x[j] >= 1, name=f'req_x_{i}_{j}')
    for b in B:
        (i, j) = b
        m.addConstr(w[b] <= y[i], name=f'bundle1_{i}_{j}')
        m.addConstr(w[b] <= y[j], name=f'bundle2_{i}_{j}')
        m.addConstr(w[b] >= y[i] + y[j] - 1, name=f'bundle3_{i}_{j}')
    return m
m = solve_problem()
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')