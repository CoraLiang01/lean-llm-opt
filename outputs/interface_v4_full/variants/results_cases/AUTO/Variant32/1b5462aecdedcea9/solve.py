import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant32/inputs/batch_01/export_25.csv']
    dfs = [pd.read_csv(p, sep=',') for p in paths]

    def filter_latest_valid(df, table_name, dealership_id='OSLO_NEW_CARS', as_of_date='2026-05-07'):
        df = df[df['dealership_id'].astype(str).str.casefold() == dealership_id.casefold()]
        df = df[df['table'].astype(str).str.casefold() == table_name.casefold()]
        df = df[df['effective_date'] <= as_of_date]
        group_cols = ['dealership_id', 'table', 'record_id']
        df['revision'] = pd.to_numeric(df['revision'], errors='coerce')
        idx = df.groupby(group_cols)['revision'].transform('max') == df['revision']
        df = df[idx]
        if 'action' in df.columns:
            df = df[df['action'].astype(str).str.casefold() != 'delete']
        df = df.drop_duplicates(subset=group_cols + [c for c in df.columns if c not in group_cols])
        return df
    benefit_df = pd.concat([filter_latest_valid(dfs[0], 'benefit'), filter_latest_valid(dfs[1], 'benefit'), filter_latest_valid(dfs[2], 'benefit')], ignore_index=True)
    bundle_df = pd.concat([filter_latest_valid(dfs[3], 'bundle'), filter_latest_valid(dfs[4], 'bundle')], ignore_index=True)
    capacity_ledger_df = pd.concat([filter_latest_valid(dfs[5], 'capacity_ledger'), filter_latest_valid(dfs[6], 'capacity_ledger')], ignore_index=True)
    category_df = pd.concat([filter_latest_valid(dfs[7], 'category'), filter_latest_valid(dfs[8], 'category')], ignore_index=True)
    fx_df = filter_latest_valid(dfs[9], 'fx')
    identity_df = pd.concat([filter_latest_valid(dfs[10], 'identity'), filter_latest_valid(dfs[11], 'identity')], ignore_index=True)
    incompatible_df = pd.concat([filter_latest_valid(dfs[12], 'incompatible'), filter_latest_valid(dfs[13], 'incompatible')], ignore_index=True)
    item_df = pd.concat([filter_latest_valid(dfs[14], 'item'), filter_latest_valid(dfs[15], 'item')], ignore_index=True)
    item_fee_df = pd.concat([filter_latest_valid(dfs[16], 'item_fee'), filter_latest_valid(dfs[17], 'item_fee')], ignore_index=True)
    requires_df = pd.concat([filter_latest_valid(dfs[20], 'requires'), filter_latest_valid(dfs[21], 'requires')], ignore_index=True)
    usage_df = pd.concat([filter_latest_valid(dfs[22], 'usage'), filter_latest_valid(dfs[23], 'usage'), filter_latest_valid(dfs[24], 'usage')], ignore_index=True)
    item_df = item_df.dropna(subset=['item_ref'])
    items = sorted(item_df['item_ref'].astype(str).unique())
    category_df = category_df.dropna(subset=['category'])
    categories = sorted(category_df['category'].astype(str).unique())
    usage_resources = usage_df['resource'].dropna().astype(str).unique()
    cap_resources = capacity_ledger_df['resource'].dropna().astype(str).unique()
    resources = sorted(set(usage_resources).union(set(cap_resources)))
    bundle_df = bundle_df.dropna(subset=['item_a', 'item_b'])
    bundles = [(str(row['item_a']), str(row['item_b'])) for _, row in bundle_df.iterrows()]
    incompatible_df = incompatible_df.dropna(subset=['item_a', 'item_b'])
    incompatible_pairs = set()
    for _, row in incompatible_df.iterrows():
        a, b = (str(row['item_a']), str(row['item_b']))
        if a != b:
            incompatible_pairs.add(tuple(sorted((a, b))))
    requires_df = requires_df.dropna(subset=['item_ref', 'prerequisite_ref'])
    requires_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for _, row in requires_df.iterrows()]
    fx_df = fx_df.dropna(subset=['currency', 'usd_cents_numerator', 'denominator'])
    fx_df = fx_df.sort_values(['currency', 'revision'], ascending=[True, False])
    fx_rates = {}
    for currency in fx_df['currency'].unique():
        sub = fx_df[fx_df['currency'] == currency]
        if not sub.empty:
            row = sub.iloc[0]
            fx_rates[currency] = (float(row['usd_cents_numerator']), float(row['denominator']))
    benefit_per_unit = {i: 0.0 for i in items}
    for i in items:
        sub = benefit_df[benefit_df['item_ref'] == i]
        total = 0.0
        for _, row in sub.iterrows():
            amt = float(row['amount'])
            currency = str(row['currency'])
            if currency not in fx_rates:
                raise ValueError(f'Missing FX rate for currency {currency}')
            num, denom = fx_rates[currency]
            total += amt * num / denom
        benefit_per_unit[i] = total
    item_fee_df = item_fee_df.dropna(subset=['item_ref', 'activation_fee_cents'])
    item_activation_fee = {}
    for i in items:
        sub = item_fee_df[item_fee_df['item_ref'] == i]
        if not sub.empty:
            row = sub.sort_values('revision', ascending=False).iloc[0]
            item_activation_fee[i] = float(row['activation_fee_cents'])
        else:
            item_activation_fee[i] = 0.0
    category_activation_fee = {}
    for g in categories:
        sub = category_df[category_df['category'] == g]
        if not sub.empty:
            row = sub.sort_values('revision', ascending=False).iloc[0]
            category_activation_fee[g] = float(row['activation_fee_cents'])
        else:
            category_activation_fee[g] = 0.0
    bundle_bonus = {}
    for a, b in bundles:
        sub = bundle_df[(bundle_df['item_a'] == a) & (bundle_df['item_b'] == b)]
        if not sub.empty:
            row = sub.sort_values('revision', ascending=False).iloc[0]
            bundle_bonus[a, b] = float(row['bonus_cents'])
        else:
            bundle_bonus[a, b] = 0.0
    unit_map = {'liter': 1000, 'ml': 1, 'hour': 60, 'minute': 1, 'kwh': 1000, 'wh': 1}
    usage_per_unit = {(i, r): 0.0 for i in items for r in resources}
    usage_df = usage_df.dropna(subset=['item_ref', 'resource', 'amount', 'unit'])
    for _, row in usage_df.iterrows():
        i = str(row['item_ref'])
        r = str(row['resource'])
        amt = float(row['amount'])
        unit = str(row['unit']).strip().lower()
        if unit not in unit_map:
            raise ValueError(f'Unknown unit {unit} for usage')
        amt_base = amt * unit_map[unit]
        if (i, r) in usage_per_unit:
            usage_per_unit[i, r] += amt_base
        else:
            usage_per_unit[i, r] = amt_base
    capacity_ledger_df = capacity_ledger_df.dropna(subset=['resource', 'amount', 'unit'])
    resource_capacity = {r: 0.0 for r in resources}
    for r in resources:
        sub = capacity_ledger_df[capacity_ledger_df['resource'] == r]
        total = 0.0
        for _, row in sub.iterrows():
            amt = float(row['amount'])
            unit = str(row['unit']).strip().lower()
            if unit not in unit_map:
                raise ValueError(f'Unknown unit {unit} for capacity_ledger')
            amt_base = amt * unit_map[unit]
            total += amt_base
        resource_capacity[r] = total
    item_df = item_df.dropna(subset=['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order'])
    item_category = {}
    item_authorized = {}
    item_min_lot = {}
    item_max_order = {}
    for _, row in item_df.iterrows():
        i = str(row['item_ref'])
        g = str(row['category'])
        item_category[i] = g
        item_authorized[i] = int(row['authorized'])
        item_min_lot[i] = int(row['minimum_lot'])
        item_max_order[i] = int(row['maximum_order'])
    category_min_qty = {}
    category_max_qty = {}
    for g in categories:
        sub = category_df[category_df['category'] == g]
        if not sub.empty:
            row = sub.sort_values('revision', ascending=False).iloc[0]
            category_min_qty[g] = int(row['minimum_quantity'])
            category_max_qty[g] = int(row['maximum_quantity'])
        else:
            category_min_qty[g] = 0
            category_max_qty[g] = 999999
    m = gp.Model('Oslo_New_Cars_Max_Net_Benefit')
    q = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        if item_authorized[i] == 0:
            m.addConstr(q[i] == 0, name=f'unauth_{i}')
            m.addConstr(z[i] == 0, name=f'unauth_z_{i}')
        else:
            m.addConstr(q[i] >= item_min_lot[i] * z[i], name=f'minlot_{i}')
            m.addConstr(q[i] <= item_max_order[i] * z[i], name=f'maxlot_{i}')
    for g in categories:
        items_in_g = [i for i in items if item_category[i] == g]
        if items_in_g:
            m.addConstr(gp.quicksum((q[i] for i in items_in_g)) >= category_min_qty[g] * y[g], name=f'catmin_{g}')
            m.addConstr(gp.quicksum((q[i] for i in items_in_g)) <= category_max_qty[g] * y[g], name=f'catmax_{g}')
            for i in items_in_g:
                m.addConstr(z[i] <= y[g], name=f'catlink_{i}_{g}')
            m.addConstr(gp.quicksum((z[i] for i in items_in_g)) >= y[g], name=f'catuse_{g}')
    for r in resources:
        m.addConstr(gp.quicksum((usage_per_unit.get((i, r), 0.0) * q[i] for i in items)) <= resource_capacity[r], name=f'rescap_{r}')
    for a, b in incompatible_pairs:
        if a in items and b in items:
            m.addConstr(z[a] + z[b] <= 1, name=f'incomp_{a}_{b}')
    bigM = sum((item_max_order[i] for i in items))
    for i, k in requires_pairs:
        if i in items and k in items:
            m.addConstr(q[i] <= bigM * z[k], name=f'req_{i}_{k}')
    for a, b in bundles:
        if a in items and b in items:
            m.addConstr(w[a, b] <= z[a], name=f'bundle1_{a}_{b}')
            m.addConstr(w[a, b] <= z[b], name=f'bundle2_{a}_{b}')
            m.addConstr(w[a, b] >= z[a] + z[b] - 1, name=f'bundle3_{a}_{b}')
        else:
            m.addConstr(w[a, b] == 0, name=f'bundle0_{a}_{b}')
    obj = gp.quicksum((benefit_per_unit[i] * q[i] for i in items))
    obj -= gp.quicksum((item_activation_fee[i] * z[i] for i in items))
    obj -= gp.quicksum((category_activation_fee[g] * y[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[a, b] * w[a, b] for a, b in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()