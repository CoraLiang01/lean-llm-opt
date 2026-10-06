import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]
    cutoff = '2026-05-07'

    def filter_latest(df, table_name, key_cols, extra_pred=None):
        df = df[df['dealership_id'].astype(str).str.casefold() == 'oslo_new_cars']
        df = df[df['table'].astype(str).str.casefold() == table_name.casefold()]
        df = df[df['effective_date'] <= cutoff]
        if extra_pred is not None:
            df = df[extra_pred(df)]
        idx = df.groupby(['dealership_id', 'table', 'record_id'])['revision'].transform('max') == df['revision']
        df = df[idx]
        if 'action' in df.columns:
            df = df[df['action'].astype(str).str.casefold() != 'delete']
        df = df.drop_duplicates(subset=['dealership_id', 'table', 'record_id'] + [c for c in df.columns if c not in ['dealership_id', 'table', 'record_id', 'revision', 'effective_date', 'action']])
        return df.copy()
    table_map = {'benefit': [0, ['item_ref', 'component']], 'bundle': [3, ['item_a', 'item_b']], 'capacity_ledger': [5, ['resource', 'entry']], 'category': [7, ['category']], 'fx': [9, ['currency']], 'identity': [10, ['ref']], 'incompatible': [12, ['item_a', 'item_b']], 'item': [14, ['item_ref']], 'item_fee': [16, ['item_ref']], 'market': [18, ['item_ref']], 'requires': [20, ['item_ref', 'prerequisite_ref']], 'usage': [22, ['item_ref', 'resource']]}
    benefit_df = pd.concat([filter_latest(dfs[0], 'benefit', ['item_ref', 'component']), filter_latest(dfs[1], 'benefit', ['item_ref', 'component']), filter_latest(dfs[2], 'benefit', ['item_ref', 'component'])], ignore_index=True)
    bundle_df = pd.concat([filter_latest(dfs[3], 'bundle', ['item_a', 'item_b']), filter_latest(dfs[4], 'bundle', ['item_a', 'item_b'])], ignore_index=True)
    capacity_ledger_df = pd.concat([filter_latest(dfs[5], 'capacity_ledger', ['resource', 'entry']), filter_latest(dfs[6], 'capacity_ledger', ['resource', 'entry'])], ignore_index=True)
    category_df = pd.concat([filter_latest(dfs[7], 'category', ['category']), filter_latest(dfs[8], 'category', ['category'])], ignore_index=True)
    fx_df = filter_latest(dfs[9], 'fx', ['currency'])
    identity_df = pd.concat([filter_latest(dfs[10], 'identity', ['ref']), filter_latest(dfs[11], 'identity', ['ref'])], ignore_index=True)
    incompatible_df = pd.concat([filter_latest(dfs[12], 'incompatible', ['item_a', 'item_b']), filter_latest(dfs[13], 'incompatible', ['item_a', 'item_b'])], ignore_index=True)
    item_df = pd.concat([filter_latest(dfs[14], 'item', ['item_ref']), filter_latest(dfs[15], 'item', ['item_ref'])], ignore_index=True)
    item_fee_df = pd.concat([filter_latest(dfs[16], 'item_fee', ['item_ref']), filter_latest(dfs[17], 'item_fee', ['item_ref'])], ignore_index=True)
    market_df = pd.concat([filter_latest(dfs[18], 'market', ['item_ref']), filter_latest(dfs[19], 'market', ['item_ref'])], ignore_index=True)
    requires_df = pd.concat([filter_latest(dfs[20], 'requires', ['item_ref', 'prerequisite_ref']), filter_latest(dfs[21], 'requires', ['item_ref', 'prerequisite_ref'])], ignore_index=True)
    usage_df = pd.concat([filter_latest(dfs[22], 'usage', ['item_ref', 'resource']), filter_latest(dfs[23], 'usage', ['item_ref', 'resource']), filter_latest(dfs[24], 'usage', ['item_ref', 'resource'])], ignore_index=True)
    item_df['authorized'] = item_df['authorized'].astype(float)
    items_all = item_df['item_ref'].astype(str).unique()
    items_auth = item_df[item_df['authorized'] > 0]['item_ref'].astype(str).unique()
    items_unauth = item_df[item_df['authorized'] <= 0]['item_ref'].astype(str).unique()
    items = item_df['item_ref'].astype(str).unique()
    categories = category_df['category'].astype(str).unique()
    resources = pd.concat([usage_df['resource'].dropna().astype(str), capacity_ledger_df['resource'].dropna().astype(str)]).unique()
    bundle_df = bundle_df.dropna(subset=['item_a', 'item_b'])
    bundles = [(str(row['item_a']), str(row['item_b'])) for _, row in bundle_df.iterrows()]
    incompatible_df = incompatible_df.dropna(subset=['item_a', 'item_b'])
    incompatible_pairs = set()
    for _, row in incompatible_df.iterrows():
        a, b = (str(row['item_a']), str(row['item_b']))
        if a != b:
            incompatible_pairs.add(tuple(sorted((a, b))))
    incompatible_pairs = list(incompatible_pairs)
    requires_df = requires_df.dropna(subset=['item_ref', 'prerequisite_ref'])
    requires_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for _, row in requires_df.iterrows()]
    fx_df = fx_df.dropna(subset=['currency', 'usd_cents_numerator', 'denominator'])
    fx_map = {}
    for _, row in fx_df.iterrows():
        c = str(row['currency'])
        fx_map[c] = (float(row['usd_cents_numerator']), float(row['denominator']))
    benefit_per_unit = {}
    for i in items:
        df = benefit_df[benefit_df['item_ref'].astype(str) == i]
        total = 0.0
        for _, row in df.iterrows():
            amt = float(row['amount'])
            curr = str(row['currency'])
            if curr not in fx_map:
                raise ValueError(f'Missing FX rate for currency {curr}')
            num, denom = fx_map[curr]
            total += amt * num / denom
        benefit_per_unit[i] = total
    item_fee_map = {}
    for i in items:
        df = item_fee_df[item_fee_df['item_ref'].astype(str) == i]
        if not df.empty:
            fee = float(df.iloc[0]['activation_fee_cents'])
            item_fee_map[i] = fee
        else:
            item_fee_map[i] = 0.0
    category_param = {}
    for g in categories:
        df = category_df[category_df['category'].astype(str) == g]
        if not df.empty:
            minq = int(df.iloc[0]['minimum_quantity'])
            maxq = int(df.iloc[0]['maximum_quantity'])
            fee = float(df.iloc[0]['activation_fee_cents'])
            category_param[g] = (minq, maxq, fee)
        else:
            raise ValueError(f'Missing category info for {g}')
    item_param = {}
    for _, row in item_df.iterrows():
        i = str(row['item_ref'])
        authorized = int(row['authorized'])
        min_lot = int(row['minimum_lot'])
        max_order = int(row['maximum_order'])
        category = str(row['category'])
        item_param[i] = (authorized, min_lot, max_order, category)
    unit_map = {'liter': 1000.0, 'ml': 1.0, 'hour': 60.0, 'minute': 1.0, 'kwh': 1000.0, 'wh': 1.0}
    resource_usage = {}
    for _, row in usage_df.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        amt = float(row['amount'])
        unit = str(row['unit']).strip().lower()
        if unit not in unit_map:
            raise ValueError(f'Unknown unit {unit} for usage')
        amt_base = amt * unit_map[unit]
        resource_usage[i, r] = amt_base
    cap_entries = capacity_ledger_df.dropna(subset=['resource', 'entry', 'amount', 'unit'])
    resource_capacity = {}
    for r in resources:
        df = cap_entries[cap_entries['resource'].astype(str) == r]
        total = 0.0
        for _, row in df.iterrows():
            amt = float(row['amount'])
            unit = str(row['unit']).strip().lower()
            if unit not in unit_map:
                raise ValueError(f'Unknown unit {unit} for capacity_ledger')
            amt_base = amt * unit_map[unit]
            total += amt_base
        resource_capacity[r] = total
    bundle_bonus = {}
    for _, row in bundle_df.iterrows():
        a, b = (str(row['item_a']), str(row['item_b']))
        bonus = float(row['bonus_cents'])
        bundle_bonus[a, b] = bonus
    item_to_category = {i: item_param[i][3] for i in items}
    category_items = {g: [] for g in categories}
    for i in items:
        g = item_to_category[i]
        if g in category_items:
            category_items[g].append(i)
    m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
    q = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        authorized, min_lot, max_order, category = item_param[i]
        if authorized <= 0:
            m.addConstr(q[i] == 0)
            m.addConstr(z[i] == 0)
        else:
            m.addConstr(q[i] >= min_lot * z[i])
            m.addConstr(q[i] <= max_order * z[i])
    for g in categories:
        items_in_g = category_items[g]
        minq, maxq, fee = category_param[g]
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) >= minq * y[g])
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) <= maxq * y[g])
        for i in items_in_g:
            m.addConstr(z[i] <= y[g])
    for r in resources:
        usage_terms = []
        for i in items:
            if (i, r) in resource_usage:
                usage_terms.append(resource_usage[i, r] * q[i])
        if usage_terms:
            m.addConstr(gp.quicksum(usage_terms) <= resource_capacity[r])
    for i, j in incompatible_pairs:
        if i in items and j in items:
            m.addConstr(z[i] + z[j] <= 1)
    for i, k in requires_pairs:
        if i in items and k in items:
            _, _, max_order_i, _ = item_param[i]
            m.addConstr(q[i] <= max_order_i * z[k])
    for a, b in bundles:
        if a in items and b in items:
            m.addConstr(w[a, b] <= z[a])
            m.addConstr(w[a, b] <= z[b])
            m.addConstr(w[a, b] >= z[a] + z[b] - 1)
        else:
            m.addConstr(w[a, b] == 0)
    obj = gp.quicksum((benefit_per_unit[i] * q[i] for i in items))
    obj -= gp.quicksum((item_fee_map[i] * z[i] for i in items))
    obj -= gp.quicksum((category_param[g][2] * y[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[a, b] * w[a, b] for a, b in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()