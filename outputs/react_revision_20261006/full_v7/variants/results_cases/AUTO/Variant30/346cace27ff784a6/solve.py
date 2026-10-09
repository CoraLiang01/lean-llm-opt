import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
    dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]

    def select_latest(df, key_cols, date_col, revision_col, action_col, cutoff_date):
        df = df[df[date_col] <= cutoff_date].copy()
        df[revision_col] = df[revision_col].astype(int)
        df = df.sort_values(key_cols + [revision_col], ascending=[True] * len(key_cols) + [False])
        df = df.drop_duplicates(key_cols, keep='first')
        df = df[df[action_col].str.casefold() != 'delete']
        return df
    cutoff_date = '2026-03-12'
    table_map = {}
    for df in dfs:
        if 'table' in df.columns:
            for t in df['table'].unique():
                if t not in table_map:
                    table_map[t] = []
                table_map[t].append(df[df['table'] == t])
    for t in table_map:
        table_map[t] = pd.concat(table_map[t], ignore_index=True)

    def filter_north_latest(table, key_cols, date_col='effective_date', revision_col='revision', action_col='action'):
        df = table_map[table]
        df = df[df['tenant'].str.casefold() == 'north']
        return select_latest(df, key_cols, date_col, revision_col, action_col, cutoff_date)
    item_df = pd.concat([filter_north_latest('item', ['tenant', 'table', 'record_id'])], ignore_index=True)
    item_df = item_df[item_df['item_ref'] != '']
    category_df = pd.concat([filter_north_latest('category', ['tenant', 'table', 'record_id'])], ignore_index=True)
    category_df = category_df[category_df['category'] != '']
    item_fee_df = pd.concat([filter_north_latest('item_fee', ['tenant', 'table', 'record_id'])], ignore_index=True)
    item_fee_df = item_fee_df[item_fee_df['item_ref'] != '']
    benefit_df = pd.concat([filter_north_latest('benefit', ['tenant', 'table', 'record_id'])], ignore_index=True)
    benefit_df = benefit_df[benefit_df['item_ref'] != '']
    usage_df = pd.concat([filter_north_latest('usage', ['tenant', 'table', 'record_id'])], ignore_index=True)
    usage_df = usage_df[usage_df['item_ref'] != '']
    capacity_ledger_df = pd.concat([filter_north_latest('capacity_ledger', ['tenant', 'table', 'record_id'])], ignore_index=True)
    capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['resource'] != '']
    incompatible_df = pd.concat([filter_north_latest('incompatible', ['tenant', 'table', 'record_id'])], ignore_index=True)
    incompatible_df = incompatible_df[(incompatible_df['item_a'] != '') & (incompatible_df['item_b'] != '')]
    requires_df = pd.concat([filter_north_latest('requires', ['tenant', 'table', 'record_id'])], ignore_index=True)
    requires_df = requires_df[(requires_df['item_ref'] != '') & (requires_df['prerequisite_ref'] != '')]
    bundle_df = pd.concat([filter_north_latest('bundle', ['tenant', 'table', 'record_id'])], ignore_index=True)
    bundle_df = bundle_df[(bundle_df['item_a'] != '') & (bundle_df['item_b'] != '')]
    identity_df = pd.concat([filter_north_latest('identity', ['tenant', 'table', 'record_id'])], ignore_index=True)
    identity_df = identity_df[identity_df['ref'] != '']
    items = sorted(item_df['item_ref'].unique())
    categories = sorted(category_df['category'].unique())
    item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
    category_to_items = {g: [] for g in categories}
    for (i, g) in item_to_category.items():
        if g in category_to_items:
            category_to_items[g].append(i)
    resource_names = sorted(set(usage_df['resource'].unique()) | set(capacity_ledger_df['resource'].unique()))
    bundles = []
    bundle_bonus = {}
    for (_, row) in bundle_df.iterrows():
        (a, b) = (row['item_a'], row['item_b'])
        if a in items and b in items:
            bundles.append((a, b))
            bonus = float(row['bonus_cents'])
            bundle_bonus[a, b] = bonus
    incompatible_pairs = []
    for (_, row) in incompatible_df.iterrows():
        (a, b) = (row['item_a'], row['item_b'])
        if a in items and b in items:
            incompatible_pairs.append((a, b))
    requires_pairs = []
    for (_, row) in requires_df.iterrows():
        (i, j) = (row['item_ref'], row['prerequisite_ref'])
        if i in items and j in items:
            requires_pairs.append((i, j))
    benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(float)
    per_unit_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    for i in items:
        if i not in per_unit_benefit:
            per_unit_benefit[i] = 0.0
    item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(float)
    item_activation_fee = {}
    for i in items:
        fee_rows = item_fee_df[item_fee_df['item_ref'] == i]
        if not fee_rows.empty:
            item_activation_fee[i] = float(fee_rows.iloc[0]['activation_fee_cents'])
        else:
            item_activation_fee[i] = 0.0
    category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(float)
    category_activation_fee = {}
    for g in categories:
        fee_rows = category_df[category_df['category'] == g]
        if not fee_rows.empty:
            category_activation_fee[g] = float(fee_rows.iloc[0]['activation_fee_cents'])
        else:
            category_activation_fee[g] = 0.0
    item_df['minimum_lot'] = item_df['minimum_lot'].astype(float)
    item_df['maximum_order'] = item_df['maximum_order'].astype(float)
    item_df['authorized'] = item_df['authorized'].astype(float)
    minimum_lot = {}
    maximum_order = {}
    authorized = {}
    for i in items:
        row = item_df[item_df['item_ref'] == i]
        if not row.empty:
            minimum_lot[i] = float(row.iloc[0]['minimum_lot'])
            maximum_order[i] = float(row.iloc[0]['maximum_order'])
            authorized[i] = float(row.iloc[0]['authorized'])
        else:
            raise ValueError(f'Missing item row for {i}')
    category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(float)
    category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(float)
    minimum_quantity = {}
    maximum_quantity = {}
    for g in categories:
        row = category_df[category_df['category'] == g]
        if not row.empty:
            minimum_quantity[g] = float(row.iloc[0]['minimum_quantity'])
            maximum_quantity[g] = float(row.iloc[0]['maximum_quantity'])
        else:
            raise ValueError(f'Missing category row for {g}')
    usage_df['amount'] = usage_df['amount'].astype(float)

    def to_base_unit(amount, unit):
        if unit == 'liter':
            return amount * 1000.0
        elif unit == 'ml':
            return amount
        elif unit == 'hour':
            return amount * 60.0
        elif unit == 'minute':
            return amount
        elif unit == 'kwh':
            return amount * 1000.0
        elif unit == 'wh':
            return amount
        else:
            raise ValueError(f'Unknown unit: {unit}')
    usage_per_unit = {}
    for (_, row) in usage_df.iterrows():
        i = row['item_ref']
        r = row['resource']
        amt = to_base_unit(float(row['amount']), row['unit'])
        key = (i, r)
        usage_per_unit[key] = usage_per_unit.get(key, 0.0) + amt
    capacity_ledger_df['amount'] = capacity_ledger_df['amount'].replace('', '0').astype(float)
    capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['resource'] != '']

    def ledger_to_base(row):
        return to_base_unit(float(row['amount']), row['unit'])
    capacity_ledger_df['amount_base'] = capacity_ledger_df.apply(ledger_to_base, axis=1)
    resource_capacity = capacity_ledger_df.groupby('resource')['amount_base'].sum().to_dict()
    for r in resource_names:
        if r not in resource_capacity:
            resource_capacity[r] = 0.0
    m = gp.Model('inventory_replenishment')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    activation_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    category_activation_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    bundle_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if authorized[i] == 0:
            m.addConstr(quantity_vars[i] == 0, name=f'auth0_{i}')
            m.addConstr(activation_vars[i] == 0, name=f'authz0_{i}')
        else:
            m.addConstr(quantity_vars[i] >= minimum_lot[i] * activation_vars[i], name=f'minlot_{i}')
            m.addConstr(quantity_vars[i] <= maximum_order[i] * activation_vars[i], name=f'maxord_{i}')
            m.addConstr(quantity_vars[i] >= 0, name=f'nonneg_{i}')
    for g in categories:
        items_in_g = category_to_items[g]
        if not items_in_g:
            m.addConstr(category_activation_vars[g] == 0, name=f'cat_empty_{g}')
            continue
        sum_q = gp.quicksum((quantity_vars[i] for i in items_in_g))
        m.addConstr(sum_q >= minimum_quantity[g] * category_activation_vars[g], name=f'catmin_{g}')
        m.addConstr(sum_q <= maximum_quantity[g] * category_activation_vars[g], name=f'catmax_{g}')
        for i in items_in_g:
            m.addConstr(activation_vars[i] <= category_activation_vars[g], name=f'catlink_{i}_{g}')
    for r in resource_names:
        expr = gp.LinExpr()
        for i in items:
            amt = usage_per_unit.get((i, r), 0.0)
            expr += amt * quantity_vars[i]
        cap = resource_capacity.get(r, 0.0)
        m.addConstr(expr <= cap, name=f'rescap_{r}')
    for (i, j) in incompatible_pairs:
        m.addConstr(activation_vars[i] + activation_vars[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, j) in requires_pairs:
        m.addConstr(activation_vars[i] <= activation_vars[j], name=f'req_{i}_{j}')
    for (i, j) in bundles:
        m.addConstr(bundle_vars[i, j] <= activation_vars[i], name=f'bundle1_{i}_{j}')
        m.addConstr(bundle_vars[i, j] <= activation_vars[j], name=f'bundle2_{i}_{j}')
        m.addConstr(bundle_vars[i, j] >= activation_vars[i] + activation_vars[j] - 1, name=f'bundle3_{i}_{j}')
    obj = gp.quicksum((per_unit_benefit[i] * quantity_vars[i] for i in items))
    obj -= gp.quicksum((item_activation_fee[i] * activation_vars[i] for i in items))
    obj -= gp.quicksum((category_activation_fee[g] * category_activation_vars[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[i, j] * bundle_vars[i, j] for (i, j) in bundles))
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