import gurobipy as gp
import pandas as pd
import numpy as np
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', sep=',')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', sep=',')
df_item1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', sep=',')
df_item2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', sep=',')
df_item_fee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', sep=',')
df_usage2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', sep=',')
df_usage1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv', sep=',')
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
item_rows['location_id'] = item_rows['location_id'].astype(str)
options = []
for (idx, row) in item_rows.iterrows():
    options.append((row['item_ref'], row['location_id']))
options = list(dict.fromkeys(options))
option_data = {}
for (idx, row) in item_rows.iterrows():
    key = (str(row['item_ref']), str(row['location_id']))
    option_data[key] = {'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'configuration_id': str(row['configuration_id']), 'location_id': str(row['location_id']), 'item_ref': str(row['item_ref'])}
df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
option_benefit = {}
for o in options:
    item_ref = o[0]
    option_benefit[o] = benefit_per_item.get(item_ref, 0)
df_item_fee['item_ref'] = df_item_fee['item_ref'].astype(str)
item_fee_map = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
option_fee = {}
for o in options:
    item_ref = o[0]
    option_fee[o] = item_fee_map.get(item_ref, 0)
df_category['category'] = df_category['category'].astype(str)
category_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
category_min = df_category.set_index('category')['minimum_quantity'].to_dict()
category_max = df_category.set_index('category')['maximum_quantity'].to_dict()
categories = list(df_category['category'].unique())
df_bundle['item_a'] = df_bundle['item_a'].astype(str)
df_bundle['item_b'] = df_bundle['item_b'].astype(str)
bundle_list = []
bundle_bonus = {}
itemref_to_locs = {}
for o in options:
    itemref_to_locs.setdefault(o[0], []).append(o[1])
for (idx, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bonus = int(row['bonus_cents'])
    if a in itemref_to_locs and b in itemref_to_locs:
        for loc_a in itemref_to_locs[a]:
            for loc_b in itemref_to_locs[b]:
                o_a = (a, loc_a)
                o_b = (b, loc_b)
                if o_a in options and o_b in options:
                    bundle_key = (o_a, o_b)
                    bundle_list.append(bundle_key)
                    bundle_bonus[bundle_key] = bonus
df_incompat['item_a'] = df_incompat['item_a'].astype(str)
df_incompat['item_b'] = df_incompat['item_b'].astype(str)
incompat_pairs = []
for (idx, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in itemref_to_locs and b in itemref_to_locs:
        for loc_a in itemref_to_locs[a]:
            for loc_b in itemref_to_locs[b]:
                o_a = (a, loc_a)
                o_b = (b, loc_b)
                if o_a in options and o_b in options:
                    incompat_pairs.append((o_a, o_b))
df_requires['item_ref'] = df_requires['item_ref'].astype(str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
requires_pairs = []
for (idx, row) in df_requires.iterrows():
    item = row['item_ref']
    prereq = row['prerequisite_ref']
    if item in itemref_to_locs and prereq in itemref_to_locs:
        for loc_item in itemref_to_locs[item]:
            for loc_prereq in itemref_to_locs[prereq]:
                o = (item, loc_item)
                p = (prereq, loc_prereq)
                if o in options and p in options:
                    requires_pairs.append((o, p))
usage_tables = [df_usage1, df_usage2]
usage_rows = pd.concat(usage_tables, ignore_index=True)
usage_rows['item_ref'] = usage_rows['item_ref'].astype(str)
usage_rows['resource'] = usage_rows['resource'].astype(str)
usage_rows['unit'] = usage_rows['unit'].astype(str)

def usage_to_ml(row):
    if row['unit'].strip().lower() == 'liter':
        return int(row['amount']) * 1000
    elif row['unit'].strip().lower() == 'ml':
        return int(row['amount'])
    else:
        raise ValueError(f"Unknown unit: {row['unit']}")
usage_rows['amount_ml'] = usage_rows.apply(usage_to_ml, axis=1)
usage_map = {}
for (idx, row) in usage_rows.iterrows():
    usage_map[row['item_ref'], row['resource']] = row['amount_ml']
resources = sorted(usage_rows['resource'].unique())
option_resource_usage = {}
for o in options:
    item_ref = o[0]
    usage = {}
    for r in resources:
        usage[o, r] = usage_map.get((item_ref, r), 0)
    option_resource_usage[o] = {r: usage[o, r] for r in resources}
df_capacity['resource'] = df_capacity['resource'].astype(str)
df_capacity['unit'] = df_capacity['unit'].astype(str)

def cap_to_ml(row):
    if row['unit'].strip().lower() == 'liter':
        return int(row['amount']) * 1000
    elif row['unit'].strip().lower() == 'ml':
        return int(row['amount'])
    else:
        raise ValueError(f"Unknown unit: {row['unit']}")
df_capacity['amount_ml'] = df_capacity.apply(cap_to_ml, axis=1)
capacity_per_resource = df_capacity.groupby('resource')['amount_ml'].sum().to_dict()
option_to_category = {o: option_data[o]['category'] for o in options}
category_to_options = {c: [o for o in options if option_to_category[o] == c] for c in categories}
option_to_resource = {o: option_data[o]['location_id'] for o in options}
resource_to_options = {r: [o for o in options if option_to_resource[o] == r] for r in resources}
m = gp.Model('market_square_merchandising')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(options, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
for o in options:
    auth = option_data[o]['authorized']
    min_lot = option_data[o]['minimum_lot']
    max_order = option_data[o]['maximum_order']
    if auth == 0:
        m.addConstr(x[o] == 0)
        m.addConstr(y[o] == 0)
    else:
        m.addConstr(x[o] >= min_lot * y[o])
        m.addConstr(x[o] <= max_order * y[o])
        m.addConstr(x[o] <= max_order * y[o])
        m.addConstr(x[o] >= 0)
for r in resources:
    m.addConstr(gp.quicksum((option_resource_usage[o][r] * x[o] for o in resource_to_options[r])) <= capacity_per_resource[r])
for c in categories:
    minq = category_min[c]
    maxq = category_max[c]
    m.addConstr(gp.quicksum((x[o] for o in category_to_options[c])) >= minq * z[c])
    m.addConstr(gp.quicksum((x[o] for o in category_to_options[c])) <= maxq * z[c])
    m.addConstr(gp.quicksum((x[o] for o in category_to_options[c])) >= 0)
    m.addConstr(z[c] <= gp.quicksum((y[o] for o in category_to_options[c])))
for (o1, o2) in incompat_pairs:
    m.addConstr(y[o1] + y[o2] <= 1)
for (o, prereq) in requires_pairs:
    m.addConstr(y[o] <= y[prereq])
for bundle in bundle_list:
    (o_a, o_b) = bundle
    m.addConstr(b[bundle] <= y[o_a])
    m.addConstr(b[bundle] <= y[o_b])
    m.addConstr(b[bundle] >= y[o_a] + y[o_b] - 1)
benefit_term = gp.quicksum((option_benefit[o] * x[o] for o in options))
item_fee_term = gp.quicksum((option_fee[o] * y[o] for o in options))
category_fee_term = gp.quicksum((category_fee[c] * z[c] for c in categories))
bundle_term = gp.quicksum((bundle_bonus[bundle] * b[bundle] for bundle in bundle_list))
m.setObjective(benefit_term - item_fee_term - category_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net merchandising benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')