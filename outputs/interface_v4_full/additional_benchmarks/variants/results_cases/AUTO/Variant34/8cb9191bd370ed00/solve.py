import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv'
df_benefit = pd.read_csv(f_benefit)
df_bundle = pd.read_csv(f_bundle)
df_capacity_ledger = pd.read_csv(f_capacity_ledger)
df_category = pd.read_csv(f_category)
df_identity = pd.read_csv(f_identity)
df_incompatible = pd.read_csv(f_incompatible)
df_item1 = pd.read_csv(f_item1)
df_item2 = pd.read_csv(f_item2)
df_item_fee = pd.read_csv(f_item_fee)
df_market = pd.read_csv(f_market)
df_requires = pd.read_csv(f_requires)
df_usage1 = pd.read_csv(f_usage1)
df_usage2 = pd.read_csv(f_usage2)
item_tables = [df_item1, df_item2]
options = []
option_data = dict()
for df in item_tables:
    for _, row in df.iterrows():
        key = (str(row['item_ref']), str(row['location_id']))
        options.append(key)
        option_data[key] = {'item_ref': str(row['item_ref']), 'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'configuration_id': str(row['configuration_id']), 'location_id': str(row['location_id'])}
options = list(dict.fromkeys(options))
benefit_per_itemref = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit = {}
for opt in options:
    item_ref = opt[0]
    benefit[opt] = int(benefit_per_itemref.get(item_ref, 0))
item_fee_per_itemref = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee = {}
for opt in options:
    item_ref = opt[0]
    item_fee[opt] = int(item_fee_per_itemref.get(item_ref, 0))
category_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
category_min = df_category.set_index('category')['minimum_quantity'].to_dict()
category_max = df_category.set_index('category')['maximum_quantity'].to_dict()
categories = list(df_category['category'].unique())
bundles = []
for _, row in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    bonus = int(row['bonus_cents'])
    bundles.append(((a, b), bonus))
incompat_pairs = []
for _, row in df_incompatible.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompat_pairs.append((a, b))
prereq_pairs = []
for _, row in df_requires.iterrows():
    a = str(row['item_ref'])
    b = str(row['prerequisite_ref'])
    prereq_pairs.append((a, b))
usage_rows = pd.concat([df_usage1, df_usage2], ignore_index=True)

def to_ml(amount, unit):
    if unit.strip().lower() == 'ml':
        return int(amount)
    elif unit.strip().lower() == 'liter':
        return int(amount) * 1000
    else:
        raise ValueError(f'Unknown unit: {unit}')
usage_per_option_resource = dict()
for _, row in usage_rows.iterrows():
    item_ref = str(row['item_ref'])
    resource = str(row['resource'])
    amount_ml = to_ml(row['amount'], row['unit'])
    usage_per_option_resource[item_ref, resource] = amount_ml
df_capacity_ledger['amount_ml'] = df_capacity_ledger.apply(lambda r: to_ml(r['amount'], r['unit']), axis=1)
capacity_per_resource = df_capacity_ledger.groupby('resource')['amount_ml'].sum().to_dict()
resources = list(capacity_per_resource.keys())
option_resource = {opt: opt[1] for opt in options}
option_category = {opt: option_data[opt]['category'] for opt in options}
option_authorized = {opt: option_data[opt]['authorized'] for opt in options}
option_minlot = {opt: option_data[opt]['minimum_lot'] for opt in options}
option_maxorder = {opt: option_data[opt]['maximum_order'] for opt in options}
category_options = {c: [opt for opt in options if option_category[opt] == c] for c in categories}
itemref_options = {}
for opt in options:
    item_ref = opt[0]
    itemref_options.setdefault(item_ref, []).append(opt)
bundle_option_pairs = []
for (item_a, item_b), bonus in bundles:
    opts_a = itemref_options.get(item_a, [])
    opts_b = itemref_options.get(item_b, [])
    for oa in opts_a:
        for ob in opts_b:
            if oa != ob:
                bundle_option_pairs.append(((oa, ob), bonus))
incompat_option_pairs = []
for item_a, item_b in incompat_pairs:
    opts_a = itemref_options.get(item_a, [])
    opts_b = itemref_options.get(item_b, [])
    for oa in opts_a:
        for ob in opts_b:
            if oa != ob:
                incompat_option_pairs.append((oa, ob))
prereq_option_pairs = []
for item_ref, prereq_ref in prereq_pairs:
    opts_a = itemref_options.get(item_ref, [])
    opts_b = itemref_options.get(prereq_ref, [])
    for oa in opts_a:
        for ob in opts_b:
            if oa != ob:
                prereq_option_pairs.append((oa, ob))

def solve_problem():
    m = gp.Model('market_square_merchandising')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(options, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundle_option_pairs, vtype=gp.GRB.BINARY, name='')
    for opt in options:
        if option_authorized[opt] == 0:
            m.addConstr(x[opt] == 0)
            m.addConstr(y[opt] == 0)
        else:
            m.addConstr(x[opt] >= 0)
            m.addConstr(x[opt] >= option_minlot[opt] * y[opt])
            m.addConstr(x[opt] <= option_maxorder[opt] * y[opt])
            m.addConstr(x[opt] <= option_maxorder[opt])
    for resource in resources:
        relevant_opts = [opt for opt in options if option_resource[opt] == resource]
        m.addConstr(gp.quicksum((usage_per_option_resource.get((opt[0], resource), 0) * x[opt] for opt in relevant_opts)) <= capacity_per_resource[resource])
    for c in categories:
        opts = category_options[c]
        m.addConstr(gp.quicksum((x[opt] for opt in opts)) >= category_min[c] * z[c])
        m.addConstr(gp.quicksum((x[opt] for opt in opts)) <= category_max[c] * z[c])
        m.addConstr(gp.quicksum((x[opt] for opt in opts)) <= category_max[c])
        m.addConstr(gp.quicksum((x[opt] for opt in opts)) >= 0)
    for c in categories:
        opts = category_options[c]
        for opt in opts:
            m.addConstr(y[opt] <= z[c])
        m.addConstr(gp.quicksum((y[opt] for opt in opts)) >= z[c])
    for (oa, ob), _ in bundle_option_pairs:
        m.addConstr(b[oa, ob] <= y[oa])
        m.addConstr(b[oa, ob] <= y[ob])
        m.addConstr(b[oa, ob] >= y[oa] + y[ob] - 1)
    for oa, ob in incompat_option_pairs:
        m.addConstr(y[oa] + y[ob] <= 1)
    for oa, ob in prereq_option_pairs:
        m.addConstr(y[oa] <= y[ob])
    obj = gp.quicksum((benefit[opt] * x[opt] for opt in options))
    obj -= gp.quicksum((item_fee[opt] * y[opt] for opt in options))
    obj -= gp.quicksum((int(category_fee[c]) * z[c] for c in categories))
    obj += gp.quicksum((bonus * b[oa, ob] for (oa, ob), bonus in bundle_option_pairs))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()