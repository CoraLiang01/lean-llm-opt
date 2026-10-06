import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_05/export_11.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_06/export_12.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant34/inputs/batch_01/export_13.csv'
df_benefit = pd.read_csv(f_benefit, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_item1 = pd.read_csv(f_item1, sep=',')
df_item2 = pd.read_csv(f_item2, sep=',')
df_item_fee = pd.read_csv(f_item_fee, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage2 = pd.read_csv(f_usage2, sep=',')
df_usage1 = pd.read_csv(f_usage1, sep=',')
item_tables = [df_item1, df_item2]
options = []
option_rows = []
for df in item_tables:
    for idx, row in df.iterrows():
        key = (str(row['item_ref']), str(row['location_id']))
        options.append(key)
        option_rows.append({'item_ref': str(row['item_ref']), 'location_id': str(row['location_id']), 'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'configuration_id': str(row['configuration_id'])})
options = list(dict.fromkeys(options))
option_data = {(row['item_ref'], row['location_id']): row for row in option_rows}
categories = sorted(df_category['category'].astype(str).unique())
category_data = {}
for idx, row in df_category.iterrows():
    c = str(row['category'])
    category_data[c] = {'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity']), 'activation_fee_cents': int(row['activation_fee_cents'])}
resources = sorted(df_capacity_ledger['resource'].astype(str).unique())
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
item_fee = {str(row['item_ref']): int(row['activation_fee_cents']) for idx, row in df_item_fee.iterrows()}
bundles = []
bundle_bonus = {}
for idx, row in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    bundles.append((a, b))
    bundle_bonus[a, b] = int(row['bonus_cents'])
incompatibles = []
for idx, row in df_incompatible.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompatibles.append((a, b))
prerequisites = []
for idx, row in df_requires.iterrows():
    item = str(row['item_ref'])
    prereq = str(row['prerequisite_ref'])
    prerequisites.append((item, prereq))
usage_rows = pd.concat([df_usage1, df_usage2], ignore_index=True)
usage_per_option_resource = {}
for idx, row in usage_rows.iterrows():
    item_ref = str(row['item_ref'])
    resource = str(row['resource'])
    amount = int(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        usage_ml = amount * 1000
    elif unit == 'ml':
        usage_ml = amount
    else:
        raise ValueError(f'Unknown unit {unit} in usage table')
    key = (item_ref, resource)
    usage_per_option_resource[key] = usage_per_option_resource.get(key, 0) + usage_ml
capacity_per_resource = {}
for resource in resources:
    df_r = df_capacity_ledger[df_capacity_ledger['resource'].astype(str) == resource]
    total = 0
    for idx, row in df_r.iterrows():
        amt = int(row['amount'])
        unit = str(row['unit']).strip().lower()
        if unit == 'liter':
            amt_ml = amt * 1000
        elif unit == 'ml':
            amt_ml = amt
        else:
            raise ValueError(f'Unknown unit {unit} in capacity_ledger')
        total += amt_ml
    capacity_per_resource[resource] = total
option_bounds = {}
option_authorized = {}
option_category = {}
for o in options:
    row = option_data[o]
    option_bounds[o] = (int(row['minimum_lot']), int(row['maximum_order']))
    option_authorized[o] = int(row['authorized'])
    option_category[o] = str(row['category'])
option_itemref = {o: o[0] for o in options}
option_locationid = {o: o[1] for o in options}
option_benefit = {}
option_item_fee = {}
for o in options:
    item_ref = o[0]
    option_benefit[o] = benefit_per_item.get(item_ref, 0)
    option_item_fee[o] = item_fee.get(item_ref, 0)
option_resource_usage = {}
for o in options:
    item_ref = o[0]
    for r in resources:
        option_resource_usage[o, r] = usage_per_option_resource.get((item_ref, r), 0)
category_options = {c: [] for c in categories}
for o in options:
    c = option_category[o]
    if c in category_options:
        category_options[c].append(o)
    else:
        category_options[c] = [o]
itemref_options = {}
for o in options:
    item_ref = o[0]
    itemref_options.setdefault(item_ref, []).append(o)
bundle_option_pairs = []
for item_a, item_b in bundles:
    options_a = itemref_options.get(item_a, [])
    options_b = itemref_options.get(item_b, [])
    for o_a in options_a:
        for o_b in options_b:
            bundle_option_pairs.append(((item_a, item_b), (o_a, o_b)))
incompatible_option_pairs = []
for item_a, item_b in incompatibles:
    options_a = itemref_options.get(item_a, [])
    options_b = itemref_options.get(item_b, [])
    for o_a in options_a:
        for o_b in options_b:
            incompatible_option_pairs.append((o_a, o_b))
prerequisite_option_pairs = []
for item_ref, prereq_ref in prerequisites:
    options_a = itemref_options.get(item_ref, [])
    options_b = itemref_options.get(prereq_ref, [])
    for o_a in options_a:
        for o_b in options_b:
            prerequisite_option_pairs.append((o_a, o_b))
M_category = max([option_bounds[o][1] for o in options]) * len(options) + 1
m = gp.Model('market_square_merchandising')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(options, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for o in options:
    min_lot, max_order = option_bounds[o]
    if option_authorized[o] == 0:
        m.addConstr(x[o] == 0, name=f'auth_{o}')
        m.addConstr(y[o] == 0, name=f'authy_{o}')
    else:
        m.addConstr(x[o] <= max_order * y[o], name=f'link_ub_{o}')
        if min_lot > 0:
            m.addConstr(x[o] >= min_lot * y[o], name=f'link_lb_{o}')
        m.addConstr(x[o] <= max_order, name=f'maxorder_{o}')
        m.addConstr(x[o] >= 0, name=f'minzero_{o}')
for r in resources:
    m.addConstr(gp.quicksum((option_resource_usage[o, r] * x[o] for o in options)) <= capacity_per_resource[r], name=f'capacity_{r}')
for c in categories:
    opts = category_options.get(c, [])
    minq = category_data[c]['minimum_quantity']
    maxq = category_data[c]['maximum_quantity']
    m.addConstr(gp.quicksum((x[o] for o in opts)) >= minq, name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[o] for o in opts)) <= maxq, name=f'catmax_{c}')
for c in categories:
    opts = category_options.get(c, [])
    if opts:
        m.addConstr(gp.quicksum((x[o] for o in opts)) <= M_category * z[c], name=f'catz_ub_{c}')
        m.addConstr(gp.quicksum((x[o] for o in opts)) >= z[c], name=f'catz_lb_{c}')
for o1, o2 in incompatible_option_pairs:
    m.addConstr(y[o1] + y[o2] <= 1, name=f'incomp_{o1}_{o2}')
for o1, o2 in prerequisite_option_pairs:
    m.addConstr(y[o1] <= y[o2], name=f'prereq_{o1}_{o2}')
for idx, (item_a, item_b) in enumerate(bundles):
    options_a = itemref_options.get(item_a, [])
    options_b = itemref_options.get(item_b, [])
    for o_a in options_a:
        for o_b in options_b:
            m.addConstr(b[item_a, item_b] <= y[o_a], name=f'bundle_b_le_ya_{item_a}_{item_b}_{o_a}_{o_b}')
            m.addConstr(b[item_a, item_b] <= y[o_b], name=f'bundle_b_le_yb_{item_a}_{item_b}_{o_a}_{o_b}')
            m.addConstr(b[item_a, item_b] >= y[o_a] + y[o_b] - 1, name=f'bundle_b_ge_sum_{item_a}_{item_b}_{o_a}_{o_b}')
obj = gp.quicksum((option_benefit[o] * x[o] for o in options)) - gp.quicksum((option_item_fee[o] * y[o] for o in options)) - gp.quicksum((category_data[c]['activation_fee_cents'] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[item_a, item_b] * b[item_a, item_b] for item_a, item_b in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()