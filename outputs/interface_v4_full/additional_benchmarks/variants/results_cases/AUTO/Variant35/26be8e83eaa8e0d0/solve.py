import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    path_01 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv'
    path_02 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv'
    path_03 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv'
    path_04 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv'
    path_05 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv'
    path_06 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv'
    path_07 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv'
    path_08 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv'
    path_09 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv'
    path_10 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv'
    path_11 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv'
    path_12 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv'
    df_benefit = pd.read_csv(path_01, sep=',')
    df_bundle = pd.read_csv(path_02, sep=',')
    df_capacity_ledger = pd.read_csv(path_03, sep=',')
    df_category = pd.read_csv(path_04, sep=',')
    df_fx = pd.read_csv(path_05, sep=',')
    df_identity = pd.read_csv(path_06, sep=',')
    df_incompatible = pd.read_csv(path_07, sep=',')
    df_item = pd.read_csv(path_08, sep=',')
    df_item_fee = pd.read_csv(path_09, sep=',')
    df_market = pd.read_csv(path_10, sep=',')
    df_requires = pd.read_csv(path_11, sep=',')
    df_usage = pd.read_csv(path_12, sep=',')
    items = df_item['item_ref'].astype(str).tolist()
    item_set = set(items)
    categories = df_category['category'].astype(str).tolist()
    category_set = set(categories)
    platforms = sorted(df_item['location_id'].unique())
    platform_set = set(platforms)
    bundles = list(df_bundle[['item_a', 'item_b']].astype(str).itertuples(index=False, name=None))
    incompat_pairs = list(df_incompatible[['item_a', 'item_b']].astype(str).itertuples(index=False, name=None))
    prereq_pairs = list(df_requires[['item_ref', 'prerequisite_ref']].astype(str).itertuples(index=False, name=None))
    fx_dict = {}
    for _, row in df_fx.iterrows():
        fx_dict[str(row['currency'])] = (int(row['usd_cents_numerator']), int(row['denominator']))
    benefit_per_unit = {}
    for item in items:
        df_ben = df_benefit[df_benefit['item_ref'].astype(str) == item]
        total = 0
        for _, row in df_ben.iterrows():
            currency = str(row['currency'])
            amt = int(row['amount'])
            num, denom = fx_dict[currency]
            val = amt * num // denom
            total += val
        benefit_per_unit[item] = total
    item_fee_dict = dict(zip(df_item_fee['item_ref'].astype(str), df_item_fee['activation_fee_cents'].astype(int)))
    category_fee_dict = dict(zip(df_category['category'].astype(str), df_category['activation_fee_cents'].astype(int)))
    bundle_bonus_dict = {}
    for _, row in df_bundle.iterrows():
        a = str(row['item_a'])
        b = str(row['item_b'])
        bundle_bonus_dict[a, b] = int(row['bonus_cents'])
    usage_per_unit = {}
    for _, row in df_usage.iterrows():
        item = str(row['item_ref'])
        platform = str(row['resource'])
        amt = int(row['amount'])
        unit = str(row['unit'])
        if unit == 'GB':
            mb = amt * 1000
        elif unit == 'MB':
            mb = amt
        else:
            raise ValueError(f'Unknown unit {unit} in usage table')
        usage_per_unit[item, platform] = mb
    platform_capacity = {}
    for platform in platform_set:
        df_cap = df_capacity_ledger[df_capacity_ledger['resource'].astype(str) == platform]
        total = df_cap['amount'].sum()
        platform_capacity[platform] = int(total)
    category_min = dict(zip(df_category['category'].astype(str), df_category['minimum_quantity'].astype(int)))
    category_max = dict(zip(df_category['category'].astype(str), df_category['maximum_quantity'].astype(int)))
    item_authorized = dict(zip(df_item['item_ref'].astype(str), df_item['authorized'].astype(int)))
    item_min_lot = dict(zip(df_item['item_ref'].astype(str), df_item['minimum_lot'].astype(int)))
    item_max_order = dict(zip(df_item['item_ref'].astype(str), df_item['maximum_order'].astype(int)))
    item_category = dict(zip(df_item['item_ref'].astype(str), df_item['category'].astype(str)))
    item_platform = dict(zip(df_item['item_ref'].astype(str), df_item['location_id'].astype(str)))
    m = gp.Model('GameEditionAllocation')
    x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
    z = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in items:
        auth = item_authorized[i]
        min_lot = item_min_lot[i]
        max_order = item_max_order[i]
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'unauth_{i}')
            m.addConstr(z[i] == 0, name=f'unauth_z_{i}')
        else:
            m.addConstr(x[i] <= max_order * z[i], name=f'xleq_{i}')
            m.addConstr(x[i] >= min_lot * z[i], name=f'xgeq_{i}')
            m.addConstr(x[i] <= max_order, name=f'xmax_{i}')
            m.addConstr(x[i] >= 0, name=f'xmin_{i}')
    for i in items:
        c = item_category[i]
        m.addConstr(z[i] <= w[c], name=f'zleqw_{i}')
    for c in categories:
        items_in_c = [i for i in items if item_category[i] == c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= category_min[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= category_max[c], name=f'catmax_{c}')
    for p in platforms:
        items_on_p = [i for i in items if item_platform[i] == p]
        m.addConstr(gp.quicksum((usage_per_unit.get((i, p), 0) * x[i] for i in items_on_p)) <= platform_capacity[p], name=f'memcap_{p}')
    for i, j in incompat_pairs:
        if i in item_set and j in item_set:
            m.addConstr(z[i] + z[j] <= 1, name=f'incomp_{i}_{j}')
    for i, prereq in prereq_pairs:
        if i in item_set and prereq in item_set:
            m.addConstr(z[i] <= z[prereq], name=f'prereq_{i}_{prereq}')
    for a, b in bundles:
        if a in item_set and b in item_set:
            m.addConstr(b[a, b] <= z[a], name=f'bundle_a_{a}_{b}')
            m.addConstr(b[a, b] <= z[b], name=f'bundle_b_{a}_{b}')
            m.addConstr(b[a, b] >= z[a] + z[b] - 1, name=f'bundle_link_{a}_{b}')
    obj_benefit = gp.quicksum((benefit_per_unit[i] * x[i] for i in items))
    obj_item_fee = gp.quicksum((item_fee_dict.get(i, 0) * z[i] for i in items))
    obj_cat_fee = gp.quicksum((category_fee_dict[c] * w[c] for c in categories))
    obj_bundle = gp.quicksum((bundle_bonus_dict.get((a, b), 0) * b[a, b] for a, b in bundles))
    m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()