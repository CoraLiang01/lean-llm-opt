import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def filter_latest(df, table_name, key_cols, date_col='effective_date', dealership_id='OSLO_NEW_CARS', cutoff='2026-05-07'):
        mask = (df['dealership_id'].str.casefold() == dealership_id.casefold()) & (df['table'].str.casefold() == table_name.casefold()) & (pd.to_datetime(df[date_col]) <= pd.to_datetime(cutoff))
        df = df[mask].copy()
        df['revision'] = pd.to_numeric(df['revision'], errors='coerce')
        df = df.dropna(subset=key_cols + ['revision'])
        df = df.sort_values(key_cols + ['revision'])
        idx = df.groupby(key_cols)['revision'].transform('max') == df['revision']
        df = df[idx]
        if 'action' in df.columns:
            df = df[df['action'].str.casefold() != 'delete']
        df = df.drop_duplicates(subset=key_cols)
        return df
    benefit = pd.concat([filter_latest(dfs[0], 'benefit', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[1], 'benefit', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[2], 'benefit', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    benefit = benefit.dropna(subset=['item_ref', 'amount', 'currency'])
    fx = filter_latest(dfs[9], 'fx', ['dealership_id', 'table', 'record_id'])
    fx = fx.dropna(subset=['currency', 'usd_cents_numerator', 'denominator'])
    fx_map = {}
    for (_, row) in fx.iterrows():
        c = str(row['currency']).casefold().strip()
        fx_map[c] = (float(row['usd_cents_numerator']), float(row['denominator']))
    item_fee = pd.concat([filter_latest(dfs[16], 'item_fee', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[17], 'item_fee', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    item_fee = item_fee.dropna(subset=['item_ref', 'activation_fee_cents'])
    item_fee_map = {}
    for (_, row) in item_fee.iterrows():
        item_fee_map[str(row['item_ref'])] = float(row['activation_fee_cents'])
    category = pd.concat([filter_latest(dfs[7], 'category', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[8], 'category', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    category = category.dropna(subset=['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'])
    category_map = {}
    for (_, row) in category.iterrows():
        g = str(row['category'])
        category_map[g] = {'min': int(row['minimum_quantity']), 'max': int(row['maximum_quantity']), 'fee': float(row['activation_fee_cents'])}
    item = pd.concat([filter_latest(dfs[14], 'item', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[15], 'item', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    item = item.dropna(subset=['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order'])
    item_map = {}
    for (_, row) in item.iterrows():
        i = str(row['item_ref'])
        g = str(row['category'])
        item_map[i] = {'category': g, 'authorized': int(row['authorized']), 'min_lot': int(row['minimum_lot']), 'max_order': int(row['maximum_order'])}
    usage = pd.concat([filter_latest(dfs[22], 'usage', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[23], 'usage', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[24], 'usage', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    usage = usage.dropna(subset=['item_ref', 'resource', 'amount', 'unit'])

    def convert_usage_unit(amount, unit):
        u = str(unit).casefold().strip()
        if u == 'liter':
            return (float(amount) * 1000.0, 'ml')
        elif u == 'ml':
            return (float(amount), 'ml')
        elif u == 'hour':
            return (float(amount) * 60.0, 'minute')
        elif u == 'minute':
            return (float(amount), 'minute')
        elif u == 'kwh':
            return (float(amount) * 1000.0, 'wh')
        elif u == 'wh':
            return (float(amount), 'wh')
        else:
            raise ValueError(f'Unknown usage unit: {unit}')
    usage_map = {}
    for (_, row) in usage.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        (amt, base_unit) = convert_usage_unit(row['amount'], row['unit'])
        usage_map.setdefault(i, {})[r] = (amt, base_unit)
    capacity = pd.concat([filter_latest(dfs[5], 'capacity_ledger', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[6], 'capacity_ledger', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    capacity = capacity.dropna(subset=['resource', 'amount', 'unit'])

    def convert_capacity_unit(amount, unit):
        u = str(unit).casefold().strip()
        if u == 'liter':
            return (float(amount) * 1000.0, 'ml')
        elif u == 'ml':
            return (float(amount), 'ml')
        elif u == 'hour':
            return (float(amount) * 60.0, 'minute')
        elif u == 'minute':
            return (float(amount), 'minute')
        elif u == 'kwh':
            return (float(amount) * 1000.0, 'wh')
        elif u == 'wh':
            return (float(amount), 'wh')
        else:
            raise ValueError(f'Unknown capacity unit: {unit}')
    capacity_map = {}
    for (_, row) in capacity.iterrows():
        r = str(row['resource'])
        (amt, base_unit) = convert_capacity_unit(row['amount'], row['unit'])
        if r not in capacity_map:
            capacity_map[r] = {'amount': 0.0, 'unit': base_unit}
        if capacity_map[r]['unit'] != base_unit:
            raise ValueError(f'Mixed units for resource {r}')
        capacity_map[r]['amount'] += amt
    incompatible = pd.concat([filter_latest(dfs[12], 'incompatible', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[13], 'incompatible', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    incompatible = incompatible.dropna(subset=['item_a', 'item_b'])
    incompatible_pairs = set()
    for (_, row) in incompatible.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        if a != b:
            incompatible_pairs.add(tuple(sorted((a, b))))
    requires = pd.concat([filter_latest(dfs[20], 'requires', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[21], 'requires', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    requires = requires.dropna(subset=['item_ref', 'prerequisite_ref'])
    requires_pairs = set()
    for (_, row) in requires.iterrows():
        i = str(row['item_ref'])
        k = str(row['prerequisite_ref'])
        if i != k:
            requires_pairs.add((i, k))
    bundle = pd.concat([filter_latest(dfs[3], 'bundle', ['dealership_id', 'table', 'record_id']), filter_latest(dfs[4], 'bundle', ['dealership_id', 'table', 'record_id'])], ignore_index=True)
    bundle = bundle.dropna(subset=['item_a', 'item_b', 'bonus_cents'])
    bundle_pairs = []
    bundle_bonus = {}
    for (_, row) in bundle.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        key = tuple(sorted((a, b)))
        bundle_pairs.append(key)
        bundle_bonus[key] = float(row['bonus_cents'])
    items = sorted(item_map.keys())
    categories = sorted(category_map.keys())
    resources = sorted(capacity_map.keys())
    bundles = bundle_pairs
    benefit_per_item = {i: 0.0 for i in items}
    for i in items:
        rows = benefit[benefit['item_ref'] == i]
        total = 0.0
        for (_, row) in rows.iterrows():
            amt = float(row['amount'])
            c = str(row['currency']).casefold().strip()
            if c not in fx_map:
                raise ValueError(f'Missing FX rate for currency {c}')
            (num, denom) = fx_map[c]
            usd_cents = amt * num / denom
            total += usd_cents
        benefit_per_item[i] = total
    usage_per_item_resource = {}
    for i in items:
        usage_per_item_resource[i] = {}
        if i in usage_map:
            for r in usage_map[i]:
                (amt, unit) = usage_map[i][r]
                if r not in capacity_map:
                    raise ValueError(f'Resource {r} used by item {i} not in capacity_map')
                if unit != capacity_map[r]['unit']:
                    raise ValueError(f"Unit mismatch for resource {r}: {unit} vs {capacity_map[r]['unit']}")
                usage_per_item_resource[i][r] = amt
    item_to_category = {i: item_map[i]['category'] for i in items}
    category_to_items = {g: [] for g in categories}
    for i in items:
        g = item_to_category[i]
        if g in category_to_items:
            category_to_items[g].append(i)
    authorized = {i: item_map[i]['authorized'] for i in items}
    min_lot = {i: item_map[i]['min_lot'] for i in items}
    max_order = {i: item_map[i]['max_order'] for i in items}
    item_fee_val = {i: item_fee_map.get(i, 0.0) for i in items}
    min_cat_qty = {g: category_map[g]['min'] for g in categories}
    max_cat_qty = {g: category_map[g]['max'] for g in categories}
    cat_fee = {g: category_map[g]['fee'] for g in categories}
    m = gp.Model('Oslo_New_Cars_Replenishment')
    m.setParam('MIPGap', 0.0001)
    q = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if authorized[i] == 0:
            m.addConstr(q[i] == 0, name=f'unauth_{i}')
            m.addConstr(z[i] == 0, name=f'unauth_z_{i}')
        else:
            m.addConstr(q[i] >= min_lot[i] * z[i], name=f'minlot_{i}')
            m.addConstr(q[i] <= max_order[i] * z[i], name=f'maxorder_{i}')
            m.addConstr(q[i] <= max_order[i] * z[i], name=f'link1_{i}')
            m.addConstr(q[i] >= z[i], name=f'link2_{i}')
    for g in categories:
        items_in_g = category_to_items[g]
        if not items_in_g:
            continue
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) >= min_cat_qty[g] * y[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((q[i] for i in items_in_g)) <= max_cat_qty[g] * y[g], name=f'catmax_{g}')
        for i in items_in_g:
            m.addConstr(z[i] <= y[g], name=f'catlink_{g}_{i}')
        m.addConstr(y[g] <= gp.quicksum((z[i] for i in items_in_g)), name=f'caty_{g}')
    for r in resources:
        expr = gp.LinExpr()
        for i in items:
            amt = usage_per_item_resource[i].get(r, 0.0)
            expr += amt * q[i]
        m.addConstr(expr <= capacity_map[r]['amount'], name=f'res_{r}')
    for (a, b) in incompatible_pairs:
        if a in items and b in items:
            m.addConstr(z[a] + z[b] <= 1, name=f'incomp_{a}_{b}')
    for (i, k) in requires_pairs:
        if i in items and k in items:
            m.addConstr(z[i] <= z[k], name=f'req_{i}_{k}')
    for b in bundles:
        (i_a, i_b) = b
        if i_a in items and i_b in items and authorized[i_a] and authorized[i_b]:
            m.addConstr(w[b] <= z[i_a], name=f'bundle1_{b}')
            m.addConstr(w[b] <= z[i_b], name=f'bundle2_{b}')
            m.addConstr(w[b] >= z[i_a] + z[i_b] - 1, name=f'bundle3_{b}')
        else:
            m.addConstr(w[b] == 0, name=f'bundle0_{b}')
    obj = gp.LinExpr()
    for i in items:
        obj += benefit_per_item[i] * q[i]
        obj -= item_fee_val[i] * z[i]
    for g in categories:
        obj -= cat_fee[g] * y[g]
    for b in bundles:
        obj += bundle_bonus[b] * w[b]
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')