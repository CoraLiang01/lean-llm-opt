import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv'
f_item_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv'
f_item_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv'
f_usage_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv'
f_usage_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv'

def solve_problem():
    benefit_df = pd.read_csv(f_benefit, dtype=str, keep_default_na=False)
    bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
    capacity_ledger_df = pd.read_csv(f_capacity_ledger, dtype=str, keep_default_na=False)
    category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
    identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
    incompatible_df = pd.read_csv(f_incompatible, dtype=str, keep_default_na=False)
    item1_df = pd.read_csv(f_item_1, dtype=str, keep_default_na=False)
    item2_df = pd.read_csv(f_item_2, dtype=str, keep_default_na=False)
    item_fee_df = pd.read_csv(f_item_fee, dtype=str, keep_default_na=False)
    market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
    requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
    usage1_df = pd.read_csv(f_usage_1, dtype=str, keep_default_na=False)
    usage2_df = pd.read_csv(f_usage_2, dtype=str, keep_default_na=False)
    item_df = pd.concat([item1_df, item2_df], ignore_index=True)
    for col in ['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'location_id']:
        if col not in item_df.columns:
            raise ValueError(f'Missing column {col} in item option tables.')
    item_df['authorized'] = item_df['authorized'].astype(int)
    item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
    item_df['maximum_order'] = item_df['maximum_order'].astype(int)
    item_df['item_key'] = list(zip(item_df['item_ref'], item_df['location_id']))
    item_keys = list(item_df['item_key'])
    item_key_to_row = {k: row for (k, row) in zip(item_df['item_key'], item_df.to_dict(orient='records'))}
    benefit_df = benefit_df[benefit_df['table'].str.casefold() == 'benefit']
    benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
    per_unit_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
    item_key_to_benefit = {}
    for k in item_keys:
        item_ref = k[0]
        if item_ref not in per_unit_benefit:
            raise ValueError(f'Missing per-unit benefit for item_ref {item_ref}')
        item_key_to_benefit[k] = per_unit_benefit[item_ref]
    item_fee_df = item_fee_df[item_fee_df['table'].str.casefold() == 'item_fee']
    item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
    item_ref_to_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
    item_key_to_fee = {}
    for k in item_keys:
        item_ref = k[0]
        if item_ref not in item_ref_to_fee:
            raise ValueError(f'Missing item activation fee for item_ref {item_ref}')
        item_key_to_fee[k] = item_ref_to_fee[item_ref]
    category_df = category_df[category_df['table'].str.casefold() == 'category']
    category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
    category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
    category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
    categories = list(category_df['category'])
    cat_to_min = category_df.set_index('category')['minimum_quantity'].to_dict()
    cat_to_max = category_df.set_index('category')['maximum_quantity'].to_dict()
    cat_to_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
    bundle_df = bundle_df[bundle_df['table'].str.casefold() == 'bundle']
    bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
    bundle_keys = list(zip(bundle_df['item_a'], bundle_df['item_b']))
    bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
    usage_df = pd.concat([usage1_df, usage2_df], ignore_index=True)
    usage_df = usage_df[usage_df['table'].str.casefold() == 'usage']
    usage_df['amount'] = usage_df['amount'].astype(int)
    usage_df['amount_ml'] = usage_df['amount'] * 1000
    usage_map = {}
    for (_, row) in usage_df.iterrows():
        usage_map[row['item_ref'], row['resource']] = row['amount_ml']
    item_key_to_resource = {}
    item_key_to_usage = {}
    for k in item_keys:
        (item_ref, location_id) = k
        key = (item_ref, location_id)
        if (item_ref, location_id) in usage_map:
            amt = usage_map[item_ref, location_id]
        else:
            amt = usage_map.get((item_ref, location_id), 0)
        item_key_to_resource[k] = location_id
        amt = usage_map.get((item_ref, location_id), 0)
        item_key_to_usage[k] = amt
    capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['table'].str.casefold() == 'capacity_ledger']
    capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
    resource_capacity = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
    resources = list(resource_capacity.keys())
    incompatible_df = incompatible_df[incompatible_df['table'].str.casefold() == 'incompatible']
    incompatible_pairs = list(zip(incompatible_df['item_a'], incompatible_df['item_b']))
    requires_df = requires_df[requires_df['table'].str.casefold() == 'requires']
    requires_pairs = list(zip(requires_df['item_ref'], requires_df['prerequisite_ref']))
    from collections import defaultdict
    item_ref_to_keys = defaultdict(list)
    for k in item_keys:
        (item_ref, location_id) = k
        item_ref_to_keys[item_ref].append(k)
    item_key_to_category = {}
    for k in item_keys:
        item_key_to_category[k] = item_key_to_row[k]['category']
    cat_to_item_keys = defaultdict(list)
    for k in item_keys:
        c = item_key_to_category[k]
        cat_to_item_keys[c].append(k)
    resource_to_item_keys = defaultdict(list)
    for k in item_keys:
        r = item_key_to_resource[k]
        resource_to_item_keys[r].append(k)
    item_key_to_minlot = {k: item_key_to_row[k]['minimum_lot'] for k in item_keys}
    item_key_to_maxorder = {k: item_key_to_row[k]['maximum_order'] for k in item_keys}
    item_key_to_authorized = {k: item_key_to_row[k]['authorized'] for k in item_keys}
    for k in item_keys:
        item_key_to_minlot[k] = int(item_key_to_minlot[k])
        item_key_to_maxorder[k] = int(item_key_to_maxorder[k])
        item_key_to_authorized[k] = int(item_key_to_authorized[k])
    m = gp.Model('market_square_merchandising')
    quantity_vars = m.addVars(item_keys, lb=0, ub=[item_key_to_maxorder[k] if item_key_to_authorized[k] else 0 for k in item_keys], vtype=gp.GRB.INTEGER, name='')
    activation_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
    category_activation_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    bundle_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
    obj = gp.LinExpr()
    obj += gp.quicksum((item_key_to_benefit[k] * quantity_vars[k] for k in item_keys))
    obj -= gp.quicksum((item_key_to_fee[k] * activation_vars[k] for k in item_keys))
    obj -= gp.quicksum((cat_to_fee[c] * category_activation_vars[c] for c in categories))
    obj += gp.quicksum((bundle_bonus[bk] * bundle_vars[bk] for bk in bundle_keys))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    for k in item_keys:
        minlot = item_key_to_minlot[k]
        maxorder = item_key_to_maxorder[k]
        authorized = item_key_to_authorized[k]
        if not authorized:
            m.addConstr(quantity_vars[k] == 0, name=f'auth_{k}')
        else:
            m.addConstr(quantity_vars[k] >= minlot * activation_vars[k], name=f'minlot_{k}')
            m.addConstr(quantity_vars[k] <= maxorder * activation_vars[k], name=f'maxorder_{k}')
    for r in resources:
        m.addConstr(gp.quicksum((item_key_to_usage[k] * quantity_vars[k] for k in resource_to_item_keys[r])) <= resource_capacity[r], name=f'capacity_{r}')
    for c in categories:
        m.addConstr(gp.quicksum((quantity_vars[k] for k in cat_to_item_keys[c])) >= cat_to_min[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((quantity_vars[k] for k in cat_to_item_keys[c])) <= cat_to_max[c], name=f'cat_max_{c}')
    for k in item_keys:
        maxorder = item_key_to_maxorder[k]
        m.addConstr(activation_vars[k] >= quantity_vars[k] / maxorder, name=f'y_lb_{k}')
        m.addConstr(quantity_vars[k] <= maxorder * activation_vars[k], name=f'y_ub_{k}')
    for c in categories:
        for k in cat_to_item_keys[c]:
            maxorder = item_key_to_maxorder[k]
            m.addConstr(category_activation_vars[c] >= quantity_vars[k] / maxorder, name=f'z_lb_{c}_{k}')
    for bk in bundle_keys:
        (item_a, item_b) = bk
        keys_a = item_ref_to_keys.get(item_a, [])
        keys_b = item_ref_to_keys.get(item_b, [])
        if not keys_a or not keys_b:
            m.addConstr(bundle_vars[bk] == 0, name=f'bundle_zero_{bk}')
            continue
        m.addConstr(bundle_vars[bk] <= gp.quicksum((activation_vars[k] for k in keys_a)), name=f'bundle_a_ub_{bk}')
        m.addConstr(bundle_vars[bk] <= gp.quicksum((activation_vars[k] for k in keys_b)), name=f'bundle_b_ub_{bk}')
        m.addConstr(bundle_vars[bk] >= gp.quicksum((activation_vars[k] for k in keys_a)) + gp.quicksum((activation_vars[k] for k in keys_b)) - 1, name=f'bundle_lb_{bk}')
    for (item_a, item_b) in incompatible_pairs:
        keys_a = item_ref_to_keys.get(item_a, [])
        keys_b = item_ref_to_keys.get(item_b, [])
        for ka in keys_a:
            for kb in keys_b:
                m.addConstr(activation_vars[ka] + activation_vars[kb] <= 1, name=f'incompat_{ka}_{kb}')
    for (item_ref, prereq_ref) in requires_pairs:
        keys_a = item_ref_to_keys.get(item_ref, [])
        keys_b = item_ref_to_keys.get(prereq_ref, [])
        if not keys_b:
            for ka in keys_a:
                m.addConstr(activation_vars[ka] == 0, name=f'requires_zero_{ka}')
        else:
            for ka in keys_a:
                m.addConstr(activation_vars[ka] <= gp.quicksum((activation_vars[kb] for kb in keys_b)), name=f'requires_{ka}_{prereq_ref}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()