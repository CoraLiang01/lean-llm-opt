import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv', sep=',')
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv', sep=',')
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv', sep=',')

def to_base_unit(amount, unit):
    u = unit.strip().casefold()
    if u == 'ml':
        return amount
    elif u == 'liter':
        return amount * 1000
    elif u == 'wh':
        return amount
    elif u == 'kwh':
        return amount * 1000
    elif u == 'minute':
        return amount
    elif u == 'hour':
        return amount * 60
    else:
        raise ValueError(f'Unknown unit: {unit}')
items_df = df_item[df_item['authorized'] == 1].copy()
item_ids = items_df['item_ref'].astype(str).tolist()
if len(item_ids) == 0:
    raise ValueError('No authorized items found.')
cat_ids = df_category['category'].astype(str).tolist()
cat_min = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
item_cat = items_df.set_index('item_ref')['category'].to_dict()
item_minlot = items_df.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = items_df.set_index('item_ref')['maximum_order'].to_dict()
item_benefit = items_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = items_df.set_index('item_ref')['item_fee_cents'].to_dict()
usage_rows = df_usage[df_usage['item_ref'].isin(item_ids)].copy()
usage_rows['amount_base'] = usage_rows.apply(lambda r: to_base_unit(r['amount'], r['unit']), axis=1)
resources = usage_rows['resource'].unique().tolist()
usage_per_item_resource = {}
for (_, row) in usage_rows.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage_per_item_resource[i, r] = row['amount_base']
cap_rows = df_capacity.copy()
cap_rows['amount_base'] = cap_rows.apply(lambda r: to_base_unit(r['amount'], r['unit']), axis=1)
resource_capacity = cap_rows.groupby('resource')['amount_base'].sum().to_dict()
items_in_cat = {c: set(items_df[items_df['category'] == c]['item_ref'].astype(str)) for c in cat_ids}
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_ids and b in item_ids:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = str(row['item_ref'])
    j = str(row['prerequisite_ref'])
    if i in item_ids and j in item_ids:
        prereq_pairs.append((i, j))
bundle_tuples = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_ids and b in item_ids:
        bundle_id = (a, b)
        bundle_tuples.append(bundle_id)
        bundle_bonus[bundle_id] = row['bonus_cents']
m = gp.Model('central_fresh_order')
q = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
z = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
w = m.addVars(cat_ids, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    minlot = int(item_minlot[i])
    maxorder = int(item_maxorder[i])
    m.addConstr(q[i] >= minlot * z[i], name=f'minlot_{i}')
    m.addConstr(q[i] <= maxorder * z[i], name=f'maxorder_{i}')
    m.addConstr(q[i] == 0, name=f'zero_{i}').Lazy = True
for c in cat_ids:
    for i in items_in_cat[c]:
        m.addConstr(z[i] <= w[c], name=f'itemcat_{i}_{c}')
    m.addConstr(w[c] <= gp.quicksum((z[i] for i in items_in_cat[c])), name=f'catact_{c}')
for c in cat_ids:
    m.addConstr(gp.quicksum((q[i] for i in items_in_cat[c])) >= int(cat_min[c]), name=f'catmin_{c}')
    m.addConstr(gp.quicksum((q[i] for i in items_in_cat[c])) <= int(cat_max[c]), name=f'catmax_{c}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_item_resource.get((i, r), 0) * q[i] for i in item_ids)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(z[i] + z[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in prereq_pairs:
    m.addConstr(z[i] <= z[j], name=f'prereq_{i}_{j}')
for (i, j) in bundle_tuples:
    m.addConstr(b[i, j] <= z[i], name=f'bundle1_{i}_{j}')
    m.addConstr(b[i, j] <= z[j], name=f'bundle2_{i}_{j}')
    m.addConstr(b[i, j] >= z[i] + z[j] - 1, name=f'bundle3_{i}_{j}')
obj = gp.quicksum((item_benefit[i] * q[i] for i in item_ids)) - gp.quicksum((item_fee[i] * z[i] for i in item_ids)) - gp.quicksum((cat_fee[c] * w[c] for c in cat_ids)) + gp.quicksum((bundle_bonus[b] * b[b] for b in bundle_tuples))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')