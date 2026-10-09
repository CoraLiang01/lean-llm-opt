import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in csv_paths]
    as_of_date = '2026-05-07'

    def select_latest_valid(df, table_name, dealership_id):
        df = df[df['table'].str.casefold() == table_name.casefold()]
        df = df[df['dealership_id'].str.casefold() == dealership_id.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        df['revision'] = df['revision'].astype(int)
        df = df.sort_values(['record_id', 'revision'], ascending=[True, False])
        df = df.drop_duplicates(subset=['record_id'], keep='first')
        if 'action' in df.columns:
            df = df[df['action'].str.casefold() != 'delete']
        return df.reset_index(drop=True)
    benefit_df = pd.concat([select_latest_valid(dfs[0], 'benefit', 'OSLO_NEW_CARS'), select_latest_valid(dfs[1], 'benefit', 'OSLO_NEW_CARS'), select_latest_valid(dfs[2], 'benefit', 'OSLO_NEW_CARS')], ignore_index=True)
    bundle_df = pd.concat([select_latest_valid(dfs[3], 'bundle', 'OSLO_NEW_CARS'), select_latest_valid(dfs[4], 'bundle', 'OSLO_NEW_CARS')], ignore_index=True)
    capacity_ledger_df = pd.concat([select_latest_valid(dfs[5], 'capacity_ledger', 'OSLO_NEW_CARS'), select_latest_valid(dfs[6], 'capacity_ledger', 'OSLO_NEW_CARS')], ignore_index=True)
    category_df = pd.concat([select_latest_valid(dfs[7], 'category', 'OSLO_NEW_CARS'), select_latest_valid(dfs[8], 'category', 'OSLO_NEW_CARS')], ignore_index=True)
    fx_df = select_latest_valid(dfs[9], 'fx', 'OSLO_NEW_CARS')
    item_fee_df = pd.concat([select_latest_valid(dfs[16], 'item_fee', 'OSLO_NEW_CARS'), select_latest_valid(dfs[17], 'item_fee', 'OSLO_NEW_CARS')], ignore_index=True)
    item_df = pd.concat([select_latest_valid(dfs[14], 'item', 'OSLO_NEW_CARS'), select_latest_valid(dfs[15], 'item', 'OSLO_NEW_CARS')], ignore_index=True)
    incompatible_df = pd.concat([select_latest_valid(dfs[12], 'incompatible', 'OSLO_NEW_CARS'), select_latest_valid(dfs[13], 'incompatible', 'OSLO_NEW_CARS')], ignore_index=True)
    requires_df = pd.concat([select_latest_valid(dfs[20], 'requires', 'OSLO_NEW_CARS'), select_latest_valid(dfs[21], 'requires', 'OSLO_NEW_CARS')], ignore_index=True)
    usage_df = pd.concat([select_latest_valid(dfs[22], 'usage', 'OSLO_NEW_CARS'), select_latest_valid(dfs[23], 'usage', 'OSLO_NEW_CARS'), select_latest_valid(dfs[24], 'usage', 'OSLO_NEW_CARS')], ignore_index=True)
    item_df = item_df[item_df['item_ref'] != '']
    items = sorted(item_df['item_ref'].unique())
    category_df = category_df[category_df['category'] != '']
    categories = sorted(category_df['category'].unique())
    item_params = {}
    for (_, row) in item_df.iterrows():
        i = row['item_ref']
        g = row['category']
        authorized = int(float(row['authorized']))
        min_lot = int(float(row['minimum_lot']))
        max_order = int(float(row['maximum_order']))
        item_params[i] = {'category': g, 'authorized': authorized, 'minimum_lot': min_lot, 'maximum_order': max_order}
    category_params = {}
    for (_, row) in category_df.iterrows():
        g = row['category']
        min_q = int(float(row['minimum_quantity']))
        max_q = int(float(row['maximum_quantity']))
        fee = int(float(row['activation_fee_cents']))
        category_params[g] = {'minimum_quantity': min_q, 'maximum_quantity': max_q, 'activation_fee_cents': fee}
    fx_df = fx_df[fx_df['currency'] != '']
    fx_rates = {}
    for (_, row) in fx_df.iterrows():
        c = row['currency']
        num = float(row['usd_cents_numerator'])
        denom = float(row['denominator'])
        fx_rates[c] = (num, denom)
    benefit_per_unit = {}
    for i in items:
        df = benefit_df[benefit_df['item_ref'] == i]
        total = 0.0
        for (_, row) in df.iterrows():
            amt = float(row['amount'])
            curr = row['currency']
            if curr not in fx_rates:
                raise ValueError(f'Missing FX rate for currency {curr}')
            (num, denom) = fx_rates[curr]
            total += amt * num / denom
        benefit_per_unit[i] = total
    item_fee_map = {}
    for (_, row) in item_fee_df.iterrows():
        i = row['item_ref']
        if i == '':
            continue
        fee = float(row['activation_fee_cents'])
        item_fee_map[i] = fee
    bundle_bonus = {}
    for (_, row) in bundle_df.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a == '' or b == '':
            continue
        bonus = float(row['bonus_cents'])
        bundle_bonus[a, b] = bonus
    usage_map = {}
    for (_, row) in usage_df.iterrows():
        i = row['item_ref']
        r = row['resource']
        if i == '' or r == '':
            continue
        amt = float(row['amount'])
        unit = row['unit'].casefold()
        if r.casefold() == 'space':
            if unit == 'liter':
                amt = amt * 1000.0
            elif unit == 'ml':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for space: {unit}')
        elif r.casefold() == 'power':
            if unit == 'kwh':
                amt = amt * 1000.0
            elif unit == 'wh':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for power: {unit}')
        elif r.casefold() == 'labor':
            if unit == 'hour':
                amt = amt * 60.0
            elif unit == 'minute':
                amt = amt
            else:
                raise ValueError(f'Unknown unit for labor: {unit}')
        else:
            raise ValueError(f'Unknown resource: {r}')
        usage_map[i, r] = amt
    resource_capacity = {}
    for r in ['space', 'power', 'labor']:
        df = capacity_ledger_df[capacity_ledger_df['resource'].str.casefold() == r.casefold()]
        total = 0.0
        for (_, row) in df.iterrows():
            amt = float(row['amount'])
            unit = row['unit'].casefold()
            if r == 'space':
                if unit == 'liter':
                    amt = amt * 1000.0
                elif unit == 'ml':
                    amt = amt
                else:
                    raise ValueError(f'Unknown unit for space: {unit}')
            elif r == 'power':
                if unit == 'kwh':
                    amt = amt * 1000.0
                elif unit == 'wh':
                    amt = amt
                else:
                    raise ValueError(f'Unknown unit for power: {unit}')
            elif r == 'labor':
                if unit == 'hour':
                    amt = amt * 60.0
                elif unit == 'minute':
                    amt = amt
                else:
                    raise ValueError(f'Unknown unit for labor: {unit}')
            total += amt
        resource_capacity[r] = total
    incompatible_pairs = set()
    for (_, row) in incompatible_df.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a == '' or b == '':
            continue
        if a in items and b in items:
            pair = tuple(sorted((a, b)))
            incompatible_pairs.add(pair)
    requires_pairs = set()
    for (_, row) in requires_df.iterrows():
        i = row['item_ref']
        k = row['prerequisite_ref']
        if i == '' or k == '':
            continue
        if i in items and k in items:
            requires_pairs.add((i, k))
    bundle_pairs = list(bundle_bonus.keys())
    category_items = {g: [] for g in categories}
    for i in items:
        g = item_params[i]['category']
        if g in categories:
            category_items[g].append(i)
    m = gp.Model('oslo_vehicle_order')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
    for i in items:
        p = item_params[i]
        min_lot = p['minimum_lot']
        max_order = p['maximum_order']
        authorized = p['authorized']
        if authorized > 0:
            m.addConstr(quantity_vars[i] >= z_vars[i] * min_lot, name=f'minlot_{i}')
            m.addConstr(quantity_vars[i] <= z_vars[i] * max_order, name=f'maxorder_{i}')
        else:
            m.addConstr(quantity_vars[i] == 0, name=f'unauth_{i}')
            m.addConstr(z_vars[i] == 0, name=f'unauthz_{i}')
    for g in categories:
        for i in category_items[g]:
            m.addConstr(y_vars[g] >= z_vars[i], name=f'catact_{g}_{i}')
    for g in categories:
        min_q = category_params[g]['minimum_quantity']
        max_q = category_params[g]['maximum_quantity']
        m.addConstr(gp.quicksum((quantity_vars[i] for i in category_items[g])) >= min_q, name=f'catmin_{g}')
        m.addConstr(gp.quicksum((quantity_vars[i] for i in category_items[g])) <= max_q, name=f'catmax_{g}')
    for r in resource_capacity:
        usage_terms = []
        for i in items:
            if (i, r) in usage_map:
                usage_terms.append(usage_map[i, r] * quantity_vars[i])
        if usage_terms:
            m.addConstr(gp.quicksum(usage_terms) <= resource_capacity[r], name=f'rescap_{r}')
    for (i, j) in incompatible_pairs:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, k) in requires_pairs:
        max_order = item_params[i]['maximum_order']
        m.addConstr(quantity_vars[i] <= max_order * z_vars[k], name=f'req_{i}_{k}')
    for (a, b) in bundle_pairs:
        m.addConstr(w_vars[a, b] <= z_vars[a], name=f'wleza_{a}_{b}')
        m.addConstr(w_vars[a, b] <= z_vars[b], name=f'wlezb_{a}_{b}')
        m.addConstr(w_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'wge_{a}_{b}')
    obj = gp.LinExpr()
    for i in items:
        obj += benefit_per_unit.get(i, 0.0) * quantity_vars[i]
        if i in item_fee_map:
            obj += -item_fee_map[i] * z_vars[i]
    for g in categories:
        obj += -category_params[g]['activation_fee_cents'] * y_vars[g]
    for b in bundle_pairs:
        obj += bundle_bonus[b] * w_vars[b]
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