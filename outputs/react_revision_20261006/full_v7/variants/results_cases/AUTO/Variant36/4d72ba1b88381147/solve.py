import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
    df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
    df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
    df_capacity['amount'] = df_capacity['amount'].astype(int)
    df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
    df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
    df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
    df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
    df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
    df_incompatible = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
    df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
    df_item_6 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
    df_item_6['authorized'] = df_item_6['authorized'].astype(int)
    df_item_6['minimum_lot'] = df_item_6['minimum_lot'].astype(int)
    df_item_6['maximum_order'] = df_item_6['maximum_order'].astype(int)
    df_item_6['unit_benefit_cents'] = df_item_6['unit_benefit_cents'].astype(int)
    df_item_6['item_fee_cents'] = df_item_6['item_fee_cents'].astype(int)
    df_item_7 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
    df_item_7['authorized'] = df_item_7['authorized'].astype(int)
    df_item_7['minimum_lot'] = df_item_7['minimum_lot'].astype(int)
    df_item_7['maximum_order'] = df_item_7['maximum_order'].astype(int)
    df_item_7['unit_benefit_cents'] = df_item_7['unit_benefit_cents'].astype(int)
    df_item_7['item_fee_cents'] = df_item_7['item_fee_cents'].astype(int)
    df_usage_10 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
    df_usage_10['amount'] = df_usage_10['amount'].astype(int)
    df_usage_11 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
    df_usage_11['amount'] = df_usage_11['amount'].astype(int)
    items_6 = df_item_6[df_item_6['authorized'] == 1]['item_ref'].unique().tolist()
    items_7 = df_item_7[df_item_7['authorized'] == 1]['item_ref'].unique().tolist()
    item_refs = sorted(set(items_6) | set(items_7))
    item_param_rows = []
    for i in item_refs:
        rows = []
        if i in df_item_6['item_ref'].values:
            rows.append(df_item_6[df_item_6['item_ref'] == i].iloc[0])
        if i in df_item_7['item_ref'].values:
            rows.append(df_item_7[df_item_7['item_ref'] == i].iloc[0])
        if len(rows) == 1:
            item_param_rows.append(rows[0])
        elif len(rows) == 2:
            if rows[0]['unit_benefit_cents'] >= rows[1]['unit_benefit_cents']:
                item_param_rows.append(rows[0])
            else:
                item_param_rows.append(rows[1])
        else:
            raise ValueError(f'Item {i} not found in any authorized item table.')
    item_category = {}
    item_min_lot = {}
    item_max_order = {}
    item_location = {}
    item_unit_benefit = {}
    item_fee = {}
    for row in item_param_rows:
        i = row['item_ref']
        item_category[i] = row['category']
        item_min_lot[i] = row['minimum_lot']
        item_max_order[i] = row['maximum_order']
        item_location[i] = row['location_id']
        item_unit_benefit[i] = row['unit_benefit_cents']
        item_fee[i] = row['item_fee_cents']
    categories = df_category['category'].unique().tolist()
    category_min_qty = dict(zip(df_category['category'], df_category['minimum_quantity']))
    category_max_qty = dict(zip(df_category['category'], df_category['maximum_quantity']))
    category_activation_fee = dict(zip(df_category['category'], df_category['activation_fee_cents']))
    df_capacity_ledger = df_capacity[df_capacity['table'].str.casefold() == 'capacity_ledger']
    resources = df_capacity_ledger['resource'].unique().tolist()
    resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
    df_usage = pd.concat([df_usage_10, df_usage_11], ignore_index=True)
    df_usage = df_usage[df_usage['table'].str.casefold() == 'usage']
    df_usage = df_usage[df_usage['item_ref'].isin(item_refs)]
    usage = {}
    for (_, row) in df_usage.iterrows():
        i = row['item_ref']
        r = row['resource']
        amt = row['amount']
        if r not in usage:
            usage[r] = {}
        usage[r][i] = amt
    df_bundle = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
    bundle_keys = []
    bundle_bonus = {}
    for (_, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in item_refs and b in item_refs:
            bundle_keys.append((a, b))
            bundle_bonus[a, b] = row['bonus_cents']
    df_incompatible = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
    incompatible_pairs = []
    for (_, row) in df_incompatible.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a in item_refs and b in item_refs:
            incompatible_pairs.append((a, b))
    df_requires = df_requires[df_requires['table'].str.casefold() == 'requires']
    requires_pairs = []
    for (_, row) in df_requires.iterrows():
        i = row['item_ref']
        prereq = row['prerequisite_ref']
        if i in item_refs and prereq in item_refs:
            requires_pairs.append((i, prereq))
    category_items = {g: [] for g in categories}
    for i in item_refs:
        g = item_category[i]
        category_items[g].append(i)
    m = gp.Model('FC_EAST_HVAC_Placement')
    quantity_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, ub={i: item_max_order[i] for i in item_refs}, name='')
    activation_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
    category_activation_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    bundle_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
    for i in item_refs:
        m.addConstr(quantity_vars[i] >= item_min_lot[i] * activation_vars[i], name='')
        m.addConstr(quantity_vars[i] <= item_max_order[i] * activation_vars[i], name='')
    for r in resources:
        expr = gp.LinExpr()
        for i in item_refs:
            amt = usage.get(r, {}).get(i, 0)
            if amt != 0:
                expr += amt * quantity_vars[i]
        m.addConstr(expr <= resource_capacity[r], name='')
    for g in categories:
        items_in_g = category_items[g]
        sum_expr = gp.quicksum((quantity_vars[i] for i in items_in_g))
        m.addConstr(sum_expr >= category_min_qty[g], name='')
        m.addConstr(sum_expr <= category_max_qty[g], name='')
        bigM = sum((item_max_order[i] for i in items_in_g))
        m.addConstr(sum_expr <= bigM * category_activation_vars[g], name='')
    for (i, j) in incompatible_pairs:
        m.addConstr(activation_vars[i] + activation_vars[j] <= 1, name='')
    for (i, prereq) in requires_pairs:
        m.addConstr(activation_vars[i] <= activation_vars[prereq], name='')
    for (a, b) in bundle_keys:
        m.addConstr(bundle_vars[a, b] <= activation_vars[a], name='')
        m.addConstr(bundle_vars[a, b] <= activation_vars[b], name='')
        m.addConstr(bundle_vars[a, b] >= activation_vars[a] + activation_vars[b] - 1, name='')
    obj = gp.LinExpr()
    obj += gp.quicksum((item_unit_benefit[i] * quantity_vars[i] for i in item_refs))
    obj -= gp.quicksum((item_fee[i] * activation_vars[i] for i in item_refs))
    obj -= gp.quicksum((category_activation_fee[g] * category_activation_vars[g] for g in categories))
    obj += gp.quicksum((bundle_bonus[b] * bundle_vars[b] for b in bundle_keys))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()