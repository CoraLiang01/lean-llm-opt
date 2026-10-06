import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def select_latest(df, key_cols, date_col, action_col, cutoff_date):
        df = df[df[date_col] <= cutoff_date].copy()
        group_cols = key_cols
        df['_rev_rank'] = df.groupby(group_cols)['revision'].rank(method='first', ascending=False)
        df = df[df['_rev_rank'] == 1].copy()
        if action_col in df.columns:
            df = df[df[action_col].str.casefold() != 'delete']
        df = df.drop(columns=['_rev_rank'])
        df = df.drop_duplicates()
        return df
    cutoff_date = '2026-03-12'
    tenant = 'NORTH'
    table_map = {'requires': [0, 24], 'identity': [1, 22], 'item_fee': [2, 16], 'bundle': [3, 10], 'benefit': [4, 12, 14], 'usage': [5, 19, 23], 'incompatible': [6, 13], 'capacity_ledger': [7, 20], 'market': [8, 21], 'item': [9, 12], 'category': [15, 17]}

    def get_table_df(table_name):
        idxs = table_map.get(table_name, [])
        dfs_sel = []
        for idx in idxs:
            df = dfs[idx]
            if 'table' in df.columns:
                df = df[df['table'].str.casefold() == table_name.casefold()]
            if 'tenant' in df.columns:
                df = df[df['tenant'].str.casefold() == tenant.casefold()]
            dfs_sel.append(df)
        if dfs_sel:
            return pd.concat(dfs_sel, ignore_index=True)
        else:
            return pd.DataFrame()
    requires_df = get_table_df('requires')
    if not requires_df.empty:
        requires_df = select_latest(requires_df, ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date)
    identity_df = get_table_df('identity')
    if not identity_df.empty:
        identity_df = select_latest(identity_df, ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date)
    item_fee_df = get_table_df('item_fee')
    if not item_fee_df.empty:
        item_fee_df = select_latest(item_fee_df, ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date)
    bundle_df = get_table_df('bundle')
    if not bundle_df.empty:
        bundle_df = select_latest(bundle_df, ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date)
    benefit_df = pd.concat([select_latest(get_table_df('benefit'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[14], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'benefit' in dfs[14].get('table', []) else pd.DataFrame()], ignore_index=True)
    usage_df = pd.concat([select_latest(get_table_df('usage'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[19], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'usage' in dfs[19].get('table', []) else pd.DataFrame(), select_latest(dfs[23], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'usage' in dfs[23].get('table', []) else pd.DataFrame()], ignore_index=True)
    incompatible_df = pd.concat([select_latest(get_table_df('incompatible'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[13], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'incompatible' in dfs[13].get('table', []) else pd.DataFrame()], ignore_index=True)
    capacity_ledger_df = pd.concat([select_latest(get_table_df('capacity_ledger'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[20], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'capacity_ledger' in dfs[20].get('table', []) else pd.DataFrame()], ignore_index=True)
    item_df = pd.concat([select_latest(get_table_df('item'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[12], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'item' in dfs[12].get('table', []) else pd.DataFrame()], ignore_index=True)
    category_df = pd.concat([select_latest(get_table_df('category'), ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date), select_latest(dfs[17], ['tenant', 'table', 'record_id'], 'effective_date', 'action', cutoff_date) if 'category' in dfs[17].get('table', []) else pd.DataFrame()], ignore_index=True)
    item_df = item_df[item_df['authorized'].fillna(0) > 0].copy()
    item_df['item_ref'] = item_df['item_ref'].astype(str)
    items = sorted(item_df['item_ref'].unique())
    item_to_cat = dict(zip(item_df['item_ref'], item_df['category']))
    categories = sorted(item_df['category'].dropna().unique())
    category_df = category_df[category_df['category'].isin(categories)].copy()
    cat_param = {}
    for g in categories:
        cat_row = category_df[category_df['category'] == g]
        if cat_row.empty:
            raise ValueError(f'Missing category parameters for {g}')
        cat_row = cat_row.sort_values('revision', ascending=False).iloc[0]
        cat_param[g] = {'min_qty': int(cat_row['minimum_quantity']), 'max_qty': int(cat_row['maximum_quantity']), 'activation_fee': float(cat_row['activation_fee_cents'])}
    item_param = {}
    for i in items:
        row = item_df[item_df['item_ref'] == i]
        if row.empty:
            raise ValueError(f'Missing item parameters for {i}')
        row = row.sort_values('revision', ascending=False).iloc[0]
        item_param[i] = {'category': row['category'], 'authorized': int(row['authorized']), 'min_lot': int(row['minimum_lot']), 'max_order': int(row['maximum_order'])}
    item_fee_df = item_fee_df[item_fee_df['item_ref'].isin(items)]
    item_fee_map = {}
    for i in items:
        fee_row = item_fee_df[item_fee_df['item_ref'] == i]
        if not fee_row.empty:
            fee_row = fee_row.sort_values('revision', ascending=False).iloc[0]
            item_fee_map[i] = float(fee_row['activation_fee_cents'])
        else:
            item_fee_map[i] = 0.0
    benefit_df = benefit_df[benefit_df['item_ref'].isin(items)]
    benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='coerce').fillna(0)
    benefit_map = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    for i in items:
        if i not in benefit_map:
            benefit_map[i] = 0.0
    usage_df = usage_df[usage_df['item_ref'].isin(items)]
    usage_map = {}
    for _, row in usage_df.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        amt = float(row['amount'])
        unit = str(row['unit']).strip().lower()
        usage_map.setdefault(i, {})
        if r in usage_map[i]:
            prev_amt, prev_unit = usage_map[i][r]
            if prev_unit != unit:
                raise ValueError(f'Multiple units for usage of item {i}, resource {r}')
            usage_map[i][r] = (prev_amt + amt, unit)
        else:
            usage_map[i][r] = (amt, unit)
    capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['resource'].notnull()]
    resource_caps = {}
    for r in capacity_ledger_df['resource'].unique():
        rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
        total = 0.0
        base_unit = None
        for _, row in rows.iterrows():
            amt = float(row['amount'])
            unit = str(row['unit']).strip().lower()
            if base_unit is None:
                base_unit = unit
            elif base_unit != unit:
                if {base_unit, unit} == {'ml', 'liter'}:
                    if unit == 'ml':
                        amt = amt / 1000.0
                    else:
                        amt = amt * 1000.0
                    unit = base_unit
                elif {base_unit, unit} == {'wh', 'kwh'}:
                    if unit == 'wh':
                        amt = amt / 1000.0
                    else:
                        amt = amt * 1000.0
                    unit = base_unit
                elif {base_unit, unit} == {'minute', 'hour'}:
                    if unit == 'minute':
                        amt = amt / 60.0
                    else:
                        amt = amt * 60.0
                    unit = base_unit
                else:
                    raise ValueError(f'Unknown unit conversion: {base_unit} vs {unit}')
            total += amt
        resource_caps[r] = (total, base_unit)
    usage_per_unit = {}
    for i in items:
        usage_per_unit[i] = {}
        for r in resource_caps:
            if i in usage_map and r in usage_map[i]:
                amt, unit = usage_map[i][r]
                cap_amt, cap_unit = resource_caps[r]
                if unit == cap_unit:
                    amt_conv = amt
                elif {unit, cap_unit} == {'ml', 'liter'}:
                    if unit == 'ml' and cap_unit == 'liter':
                        amt_conv = amt / 1000.0
                    elif unit == 'liter' and cap_unit == 'ml':
                        amt_conv = amt * 1000.0
                    else:
                        raise ValueError(f'Unknown ml/liter conversion: {unit} to {cap_unit}')
                elif {unit, cap_unit} == {'wh', 'kwh'}:
                    if unit == 'wh' and cap_unit == 'kwh':
                        amt_conv = amt / 1000.0
                    elif unit == 'kwh' and cap_unit == 'wh':
                        amt_conv = amt * 1000.0
                    else:
                        raise ValueError(f'Unknown wh/kwh conversion: {unit} to {cap_unit}')
                elif {unit, cap_unit} == {'minute', 'hour'}:
                    if unit == 'minute' and cap_unit == 'hour':
                        amt_conv = amt / 60.0
                    elif unit == 'hour' and cap_unit == 'minute':
                        amt_conv = amt * 60.0
                    else:
                        raise ValueError(f'Unknown minute/hour conversion: {unit} to {cap_unit}')
                else:
                    raise ValueError(f'Unknown unit conversion: {unit} to {cap_unit}')
                usage_per_unit[i][r] = amt_conv
            else:
                usage_per_unit[i][r] = 0.0
    incompatible_pairs = set()
    for _, row in incompatible_df.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        if a in items and b in items:
            incompatible_pairs.add(tuple(sorted((a, b))))
    requires_pairs = set()
    for _, row in requires_df.iterrows():
        i = str(row['item_ref'])
        j = str(row['prerequisite_ref'])
        if i in items and j in items:
            requires_pairs.add((i, j))
    bundle_map = {}
    for _, row in bundle_df.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        if a in items and b in items:
            key = tuple(sorted((a, b)))
            bundle_map[key] = float(row['bonus_cents'])
    m = gp.Model('InventoryReplenishment')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundle_map.keys(), vtype=gp.GRB.BINARY, name='')
    for i in items:
        min_lot = item_param[i]['min_lot']
        max_order = item_param[i]['max_order']
        m.addConstr(x[i] >= min_lot * y[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= max_order * y[i], name=f'maxlot_{i}')
    for g in categories:
        minq = cat_param[g]['min_qty']
        maxq = cat_param[g]['max_qty']
        items_in_g = [i for i in items if item_param[i]['category'] == g]
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= minq * z[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= maxq * z[g], name=f'catmax_{g}')
    for r in resource_caps:
        cap_amt, _ = resource_caps[r]
        m.addConstr(gp.quicksum((usage_per_unit[i][r] * x[i] for i in items)) <= cap_amt, name=f'rescap_{r}')
    for g in categories:
        items_in_g = [i for i in items if item_param[i]['category'] == g]
        for i in items_in_g:
            m.addConstr(z[g] >= y[i], name=f'catlink_{g}_{i}')
    for i in items:
        m.addConstr(y[i] <= gp.quicksum([x[i] >= 1]), name=f'y_x_link_{i}')
    for a, b in incompatible_pairs:
        m.addConstr(y[a] + y[b] <= 1, name=f'incomp_{a}_{b}')
    for i, j in requires_pairs:
        max_order = item_param[i]['max_order']
        m.addConstr(x[i] <= max_order * y[j], name=f'req_{i}_{j}')
    for a, b in bundle_map:
        m.addConstr(w[a, b] <= y[a], name=f'wleya_{a}_{b}')
        m.addConstr(w[a, b] <= y[b], name=f'wleyb_{a}_{b}')
        m.addConstr(w[a, b] >= y[a] + y[b] - 1, name=f'wgeya_yb_{a}_{b}')
    obj = gp.quicksum((benefit_map[i] * x[i] for i in items))
    obj -= gp.quicksum((item_fee_map[i] * y[i] for i in items))
    obj -= gp.quicksum((cat_param[g]['activation_fee'] * z[g] for g in categories))
    obj += gp.quicksum((bundle_map[b] * w[b] for b in bundle_map))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()