import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]
    as_of_date = '2026-05-07'

    def filter_valid(df, dealership_id, table_name):
        mask = (df['dealership_id'].astype(str).str.casefold() == dealership_id.casefold()) & (df['table'].astype(str).str.casefold() == table_name.casefold())
        df = df[mask].copy()
        df = df[df['effective_date'] <= as_of_date]
        df['revision'] = pd.to_numeric(df['revision'], errors='coerce')
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(['record_id'], keep='first')
        if 'action' in df.columns:
            df = df[df['action'].str.casefold() != 'delete']
        return df
    table_map = {'benefit': [0, 1, 2], 'bundle': [3, 4], 'capacity_ledger': [5, 6], 'category': [7, 8], 'fx': [9], 'identity': [10, 11], 'incompatible': [12, 13], 'item': [14, 15], 'item_fee': [16, 17], 'market': [18, 19], 'requires': [20, 21], 'usage': [22, 23, 24]}
    benefit_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'benefit') for i in table_map['benefit']], ignore_index=True)
    bundle_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'bundle') for i in table_map['bundle']], ignore_index=True)
    capacity_ledger_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'capacity_ledger') for i in table_map['capacity_ledger']], ignore_index=True)
    category_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'category') for i in table_map['category']], ignore_index=True)
    fx_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'fx') for i in table_map['fx']], ignore_index=True)
    identity_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'identity') for i in table_map['identity']], ignore_index=True)
    incompatible_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'incompatible') for i in table_map['incompatible']], ignore_index=True)
    item_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'item') for i in table_map['item']], ignore_index=True)
    item_fee_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'item_fee') for i in table_map['item_fee']], ignore_index=True)
    requires_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'requires') for i in table_map['requires']], ignore_index=True)
    usage_df = pd.concat([filter_valid(dfs[i], 'OSLO_NEW_CARS', 'usage') for i in table_map['usage']], ignore_index=True)
    items = sorted(item_df['item_ref'].dropna().unique())
    categories = sorted(category_df['category'].dropna().unique())
    resources = sorted(pd.concat([usage_df['resource'].dropna(), capacity_ledger_df['resource'].dropna()]).unique())
    bundle_pairs = []
    for (_, row) in bundle_df.iterrows():
        if pd.notna(row['item_a']) and pd.notna(row['item_b']):
            bundle_pairs.append((str(row['item_a']), str(row['item_b'])))
    incompatible_pairs = []
    for (_, row) in incompatible_df.iterrows():
        if pd.notna(row['item_a']) and pd.notna(row['item_b']):
            incompatible_pairs.append((str(row['item_a']), str(row['item_b'])))
    requires_pairs = []
    for (_, row) in requires_df.iterrows():
        if pd.notna(row['item_ref']) and pd.notna(row['prerequisite_ref']):
            requires_pairs.append((str(row['item_ref']), str(row['prerequisite_ref'])))
    fx_df = fx_df.dropna(subset=['currency'])
    fx_df = fx_df.sort_values(['currency', 'revision'], ascending=[True, False])
    fx_df = fx_df.drop_duplicates(['currency'], keep='first')
    fx_rate = {}
    for (_, row) in fx_df.iterrows():
        c = str(row['currency'])
        fx_rate[c] = (float(row['usd_cents_numerator']), float(row['denominator']))
    benefit_per_unit = {}
    for i in items:
        rows = benefit_df[benefit_df['item_ref'] == i]
        total = 0.0
        for (_, r) in rows.iterrows():
            amt = float(r['amount'])
            curr = str(r['currency'])
            if curr not in fx_rate:
                raise ValueError(f'Missing FX rate for currency {curr}')
            (num, denom) = fx_rate[curr]
            total += amt * num / denom
        benefit_per_unit[i] = total
    item_fee_df = item_fee_df.dropna(subset=['item_ref'])
    item_fee_df = item_fee_df.sort_values(['item_ref', 'revision'], ascending=[True, False])
    item_fee_df = item_fee_df.drop_duplicates(['item_ref'], keep='first')
    item_fee = {}
    for (_, row) in item_fee_df.iterrows():
        item_fee[str(row['item_ref'])] = float(row['activation_fee_cents'])
    category_df = category_df.dropna(subset=['category'])
    category_df = category_df.sort_values(['category', 'revision'], ascending=[True, False])
    category_df = category_df.drop_duplicates(['category'], keep='first')
    category_min = {}
    category_max = {}
    category_fee = {}
    for (_, row) in category_df.iterrows():
        g = str(row['category'])
        category_min[g] = int(row['minimum_quantity'])
        category_max[g] = int(row['maximum_quantity'])
        category_fee[g] = float(row['activation_fee_cents'])

    def to_base_unit(amount, unit):
        if unit == 'liter':
            return float(amount) * 1000.0
        elif unit == 'ml':
            return float(amount)
        elif unit == 'hour':
            return float(amount) * 60.0
        elif unit == 'minute':
            return float(amount)
        elif unit == 'kwh':
            return float(amount) * 1000.0
        elif unit == 'wh':
            return float(amount)
        else:
            raise ValueError(f'Unknown unit: {unit}')
    usage_dict = {}
    for (_, row) in usage_df.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        amt = row['amount']
        unit = str(row['unit'])
        usage_dict[i, r] = to_base_unit(amt, unit)
    capacity_ledger_df = capacity_ledger_df.dropna(subset=['resource'])
    capacity = {}
    for r in resources:
        rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
        total = 0.0
        for (_, row) in rows.iterrows():
            amt = row['amount']
            unit = str(row['unit'])
            total += to_base_unit(amt, unit)
        capacity[r] = total
    item_df = item_df.dropna(subset=['item_ref'])
    item_df = item_df.sort_values(['item_ref', 'revision'], ascending=[True, False])
    item_df = item_df.drop_duplicates(['item_ref'], keep='first')
    authorized = {}
    minimum_lot = {}
    maximum_order = {}
    item_category = {}
    for (_, row) in item_df.iterrows():
        i = str(row['item_ref'])
        authorized[i] = int(row['authorized'])
        minimum_lot[i] = int(row['minimum_lot'])
        maximum_order[i] = int(row['maximum_order'])
        item_category[i] = str(row['category'])
    category_items = {g: [] for g in categories}
    for i in items:
        g = item_category[i]
        if g in category_items:
            category_items[g].append(i)
    bundle_bonus = {}
    for (_, row) in bundle_df.iterrows():
        if pd.notna(row['item_a']) and pd.notna(row['item_b']):
            a = str(row['item_a'])
            b = str(row['item_b'])
            bundle_bonus[a, b] = float(row['bonus_cents'])
    m = gp.Model('OsloNewCarsNetBenefit')
    m.setParam('MIPGap', 0.0001)
    q = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if authorized[i] == 0:
            m.addConstr(q[i] == 0)
            m.addConstr(z[i] == 0)
        else:
            m.addConstr(q[i] >= minimum_lot[i] * z[i])
            m.addConstr(q[i] <= maximum_order[i] * z[i])
            m.addConstr(q[i] >= 0)
            m.addConstr(z[i] <= 1)
    for g in categories:
        items_in_g = category_items[g]
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) >= category_min[g] * y[g])
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) <= category_max[g] * y[g])
        for i in items_in_g:
            m.addConstr(z[i] <= y[g])
        m.addConstr(y[g] <= gp.quicksum((z[i] for i in items_in_g)))
    for r in resources:
        m.addConstr(gp.quicksum((usage_dict.get((i, r), 0.0) * q[i] for i in items)) <= capacity[r])
    for (i, j) in incompatible_pairs:
        if i in items and j in items:
            m.addConstr(z[i] + z[j] <= 1)
    M = max(maximum_order.values()) if maximum_order else 1000
    for (i, k) in requires_pairs:
        if i in items and k in items:
            m.addConstr(q[i] <= M * z[k])
    for (a, b) in bundle_pairs:
        if a in items and b in items:
            m.addConstr(w[a, b] <= z[a])
            m.addConstr(w[a, b] <= z[b])
            m.addConstr(w[a, b] >= z[a] + z[b] - 1)
        else:
            m.addConstr(w[a, b] == 0)
    obj_benefit = gp.quicksum((benefit_per_unit[i] * q[i] for i in items))
    obj_item_fee = gp.quicksum((item_fee.get(i, 0.0) * z[i] for i in items))
    obj_category_fee = gp.quicksum((category_fee[g] * y[g] for g in categories))
    obj_bundle_bonus = gp.quicksum((bundle_bonus.get((a, b), 0.0) * w[a, b] for (a, b) in bundle_pairs))
    m.setObjective(obj_benefit - obj_item_fee - obj_category_fee + obj_bundle_bonus, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')