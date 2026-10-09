import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', sep=',')
df_usage_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', sep=',')
df_usage_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv', sep=',')
df_item_1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', sep=',')
df_item_2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', sep=',')
item_tables = [df_item_1, df_item_2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
item_rows['category'] = item_rows['category'].astype(str)
item_rows['location_id'] = item_rows['location_id'].astype(str)
item_refs = item_rows['item_ref'].unique().tolist()
category_rows = df_category.copy()
category_rows['category'] = category_rows['category'].astype(str)
categories = category_rows['category'].unique().tolist()
areas = sorted(set(df_capacity['resource'].unique()) | set(df_usage_1['resource'].unique()) | set(df_usage_2['resource'].unique()))
bundle_rows = df_bundle.copy()
bundles = bundle_rows.index.tolist()
incompat_rows = df_incompat.copy()
incompat_pairs = list(zip(incompat_rows['item_a'].astype(str), incompat_rows['item_b'].astype(str)))
requires_rows = df_requires.copy()
requires_pairs = list(zip(requires_rows['item_ref'].astype(str), requires_rows['prerequisite_ref'].astype(str)))
usage_rows = pd.concat([df_usage_1, df_usage_2], ignore_index=True)
usage_rows['item_ref'] = usage_rows['item_ref'].astype(str)
usage_rows['resource'] = usage_rows['resource'].astype(str)
usage_dict = {}
for (_, row) in usage_rows.iterrows():
    usage_dict[row['item_ref'], row['resource']] = int(row['amount'])
df_capacity['resource'] = df_capacity['resource'].astype(str)
capacity_per_area = df_capacity.groupby('resource')['amount'].sum().to_dict()
item_param_cols = ['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'location_id', 'unit_benefit_cents', 'item_fee_cents']
item_param_dict = {}
for (_, row) in item_rows.iterrows():
    i = str(row['item_ref'])
    item_param_dict[i] = {'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': str(row['location_id']), 'unit_benefit_cents': int(row['unit_benefit_cents']), 'item_fee_cents': int(row['item_fee_cents'])}
cat_param_dict = {}
for (_, row) in category_rows.iterrows():
    c = str(row['category'])
    cat_param_dict[c] = {'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity']), 'activation_fee_cents': int(row['activation_fee_cents'])}
bundle_param_dict = {}
for (idx, row) in bundle_rows.iterrows():
    bundle_param_dict[idx] = {'item_a': str(row['item_a']), 'item_b': str(row['item_b']), 'bonus_cents': int(row['bonus_cents'])}
cat2items = {c: [] for c in categories}
for i in item_refs:
    c = item_param_dict[i]['category']
    if c in cat2items:
        cat2items[c].append(i)
m = gp.Model('FC_EAST_HVAC_AC_Placement')
q = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_param_dict[i]['authorized']
    min_lot = item_param_dict[i]['minimum_lot']
    max_order = item_param_dict[i]['maximum_order']
    if auth == 0:
        m.addConstr(q[i] == 0, name=f'unauth_{i}')
        m.addConstr(z[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q[i] >= min_lot * z[i], name=f'minlot_{i}')
        m.addConstr(q[i] <= max_order * z[i], name=f'maxorder_{i}')
for area in areas:
    items_in_area = [i for i in item_refs if item_param_dict[i]['location_id'] == area]
    if len(items_in_area) == 0:
        continue
    usage_expr = gp.LinExpr()
    for i in items_in_area:
        usage_amt = usage_dict.get((i, area), 0)
        usage_expr += usage_amt * q[i]
    cap = capacity_per_area.get(area, 0)
    m.addConstr(usage_expr <= cap, name=f'cap_{area}')
for c in categories:
    items_in_c = cat2items[c]
    if len(items_in_c) == 0:
        continue
    minq = cat_param_dict[c]['minimum_quantity']
    maxq = cat_param_dict[c]['maximum_quantity']
    m.addConstr(gp.quicksum((q[i] for i in items_in_c)) >= minq * w[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((q[i] for i in items_in_c)) <= maxq * w[c], name=f'catmax_{c}')
    for i in items_in_c:
        m.addConstr(w[c] >= z[i], name=f'catw_{c}_{i}')
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in requires_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z[i] <= z[j], name=f'requires_{i}_{j}')
for bundle_idx in bundles:
    item_a = bundle_param_dict[bundle_idx]['item_a']
    item_b = bundle_param_dict[bundle_idx]['item_b']
    if item_a in item_refs and item_b in item_refs:
        m.addConstr(b[bundle_idx] <= z[item_a], name=f'bundle_a_{bundle_idx}')
        m.addConstr(b[bundle_idx] <= z[item_b], name=f'bundle_b_{bundle_idx}')
        m.addConstr(b[bundle_idx] >= z[item_a] + z[item_b] - 1, name=f'bundle_and_{bundle_idx}')
    else:
        m.addConstr(b[bundle_idx] == 0, name=f'bundle_skip_{bundle_idx}')
obj = gp.LinExpr()
for i in item_refs:
    obj += item_param_dict[i]['unit_benefit_cents'] * q[i]
    obj -= item_param_dict[i]['item_fee_cents'] * z[i]
for c in categories:
    obj -= cat_param_dict[c]['activation_fee_cents'] * w[c]
for bundle_idx in bundles:
    obj += bundle_param_dict[bundle_idx]['bonus_cents'] * b[bundle_idx]
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')