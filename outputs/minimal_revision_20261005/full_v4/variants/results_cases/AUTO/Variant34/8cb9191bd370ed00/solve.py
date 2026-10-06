import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
    df_benefit = pd.read_csv(paths[0], sep=',')
    df_bundle = pd.read_csv(paths[1], sep=',')
    df_capacity_ledger = pd.read_csv(paths[2], sep=',')
    df_category = pd.read_csv(paths[3], sep=',')
    df_identity = pd.read_csv(paths[4], sep=',')
    df_incompatible = pd.read_csv(paths[5], sep=',')
    df_item_1 = pd.read_csv(paths[6], sep=',')
    df_item_2 = pd.read_csv(paths[7], sep=',')
    df_item_fee = pd.read_csv(paths[8], sep=',')
    df_market = pd.read_csv(paths[9], sep=',')
    df_requires = pd.read_csv(paths[10], sep=',')
    df_usage_1 = pd.read_csv(paths[11], sep=',')
    df_usage_2 = pd.read_csv(paths[12], sep=',')
    item_cols = ['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'configuration_id', 'location_id']
    items1 = df_item_1[item_cols].copy()
    items2 = df_item_2[item_cols].copy()
    items = pd.concat([items1, items2], axis=0, ignore_index=True)
    items['option_key'] = list(zip(items['item_ref'], items['location_id']))
    option_keys = list(items['option_key'].unique())
    option_df = items.set_index('option_key')
    itemref_to_optionkeys = option_df.groupby('item_ref').apply(lambda df: list(df.index)).to_dict()
    benefit_sum = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
    per_unit_benefit = {}
    for ok in option_keys:
        item_ref = ok[0]
        if item_ref not in benefit_sum:
            raise ValueError(f'Missing benefit for item_ref {item_ref}')
        per_unit_benefit[ok] = benefit_sum[item_ref]
    item_fee_map = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
    option_fee = {}
    for ok in option_keys:
        item_ref = ok[0]
        if item_ref not in item_fee_map:
            raise ValueError(f'Missing item_fee for item_ref {item_ref}')
        option_fee[ok] = item_fee_map[item_ref]
    min_lot = option_df['minimum_lot'].to_dict()
    max_order = option_df['maximum_order'].to_dict()
    authorized = option_df['authorized'].to_dict()
    option_category = option_df['category'].to_dict()
    option_location = option_df['location_id'].to_dict()
    categories = list(df_category['category'].unique())
    cat_min = df_category.set_index('category')['minimum_quantity'].to_dict()
    cat_max = df_category.set_index('category')['maximum_quantity'].to_dict()
    cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
    cat_to_options = {c: [ok for ok in option_keys if option_category[ok] == c] for c in categories}
    df_capacity_ledger['amount_ml'] = df_capacity_ledger['amount']
    section_capacity = df_capacity_ledger.groupby('resource')['amount_ml'].sum().to_dict()
    resources = list(section_capacity.keys())
    usage_frames = []
    for df_usage in [df_usage_1, df_usage_2]:
        dfu = df_usage.copy()
        dfu['amount_ml'] = np.where(dfu['unit'].str.casefold() == 'liter', dfu['amount'] * 1000, dfu['amount'])
        usage_frames.append(dfu)
    df_usage_all = pd.concat(usage_frames, axis=0, ignore_index=True)
    usage_map = df_usage_all.set_index(['item_ref', 'resource'])['amount_ml'].to_dict()
    option_resource_usage = {}
    for ok in option_keys:
        (item_ref, location_id) = ok
        key = (item_ref, location_id)
        amt = usage_map.get((item_ref, location_id), 0)
        option_resource_usage[ok] = amt
    bundle_tuples = []
    bundle_bonus = {}
    for (_, row) in df_bundle.iterrows():
        a = row['item_a']
        b = row['item_b']
        bonus = row['bonus_cents']
        if a not in itemref_to_optionkeys or b not in itemref_to_optionkeys:
            continue
        for oka in itemref_to_optionkeys[a]:
            for okb in itemref_to_optionkeys[b]:
                bundle_tuples.append((oka, okb))
                bundle_bonus[oka, okb] = bonus
    incompatible_pairs = []
    for (_, row) in df_incompatible.iterrows():
        a = row['item_a']
        b = row['item_b']
        if a not in itemref_to_optionkeys or b not in itemref_to_optionkeys:
            continue
        for oka in itemref_to_optionkeys[a]:
            for okb in itemref_to_optionkeys[b]:
                incompatible_pairs.append((oka, okb))
    prerequisite_pairs = []
    for (_, row) in df_requires.iterrows():
        a = row['item_ref']
        b = row['prerequisite_ref']
        if a not in itemref_to_optionkeys or b not in itemref_to_optionkeys:
            continue
        for oka in itemref_to_optionkeys[a]:
            for okb in itemref_to_optionkeys[b]:
                prerequisite_pairs.append((oka, okb))
    m = gp.Model('market_square_merchandising')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(option_keys, lb=0, ub=[max_order[ok] for ok in option_keys], vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(option_keys, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
    for ok in option_keys:
        if authorized[ok] == 0:
            m.addConstr(x[ok] == 0, name='auth_' + str(ok))
    for ok in option_keys:
        m.addConstr(x[ok] <= max_order[ok] * y[ok], name='ub_' + str(ok))
        m.addConstr(x[ok] >= min_lot[ok] * y[ok], name='lb_' + str(ok))
        m.addConstr(x[ok] >= 0, name='nonneg_' + str(ok))
        m.addConstr(x[ok] <= max_order[ok], name='max_' + str(ok))
        if authorized[ok] == 0:
            m.addConstr(y[ok] == 0, name='yauth_' + str(ok))
    for r in resources:
        relevant_oks = [ok for ok in option_keys if option_location[ok] == r]
        m.addConstr(gp.quicksum((option_resource_usage[ok] * x[ok] for ok in relevant_oks)) <= section_capacity[r], name='cap_' + str(r))
    for c in categories:
        oks = cat_to_options[c]
        m.addConstr(gp.quicksum((x[ok] for ok in oks)) >= cat_min[c], name='catmin_' + str(c))
        m.addConstr(gp.quicksum((x[ok] for ok in oks)) <= cat_max[c], name='catmax_' + str(c))
    for c in categories:
        oks = cat_to_options[c]
        for ok in oks:
            m.addConstr(z[c] >= y[ok], name='zlink_' + str((c, ok)))
        m.addConstr(z[c] <= gp.quicksum((y[ok] for ok in oks)), name='zsum_' + str(c))
    for ok in option_keys:
        m.addConstr(x[ok] <= max_order[ok] * y[ok], name='yup_' + str(ok))
        m.addConstr(x[ok] >= min_lot[ok] * y[ok], name='ylow_' + str(ok))
    for (ok1, ok2) in incompatible_pairs:
        m.addConstr(y[ok1] + y[ok2] <= 1, name='inc_' + str((ok1, ok2)))
    for (ok, prereq) in prerequisite_pairs:
        m.addConstr(y[ok] <= y[prereq], name='req_' + str((ok, prereq)))
    for (oka, okb) in bundle_tuples:
        m.addConstr(b[oka, okb] <= y[oka], name='b1_' + str((oka, okb)))
        m.addConstr(b[oka, okb] <= y[okb], name='b2_' + str((oka, okb)))
        m.addConstr(b[oka, okb] >= y[oka] + y[okb] - 1, name='b3_' + str((oka, okb)))
    obj = gp.quicksum((per_unit_benefit[ok] * x[ok] for ok in option_keys)) - gp.quicksum((option_fee[ok] * y[ok] for ok in option_keys)) - gp.quicksum((cat_fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[bt] * b[bt] for bt in bundle_tuples))
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