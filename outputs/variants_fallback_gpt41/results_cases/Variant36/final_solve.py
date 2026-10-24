import gurobipy as gp
import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
df_bundle = read_csv('benchmark_dataset/Variant36/inputs/batch_01/export_01.csv')
df_capacity_ledger = read_csv('benchmark_dataset/Variant36/inputs/batch_02/export_02.csv')
df_category = read_csv('benchmark_dataset/Variant36/inputs/batch_03/export_03.csv')
df_identity = read_csv('benchmark_dataset/Variant36/inputs/batch_04/export_04.csv')
df_incompatible = read_csv('benchmark_dataset/Variant36/inputs/batch_05/export_05.csv')
df_requires = read_csv('benchmark_dataset/Variant36/inputs/batch_03/export_09.csv')
df_usage_1 = read_csv('benchmark_dataset/Variant36/inputs/batch_04/export_10.csv')
df_usage_2 = read_csv('benchmark_dataset/Variant36/inputs/batch_05/export_11.csv')
df_item_1 = read_csv('benchmark_dataset/Variant36/inputs/batch_06/export_06.csv')
df_item_2 = read_csv('benchmark_dataset/Variant36/inputs/batch_01/export_07.csv')
df_items = pd.concat([df_item_1[df_item_1['table'].str.strip().str.casefold() == 'item'], df_item_2[df_item_2['table'].str.strip().str.casefold() == 'item']], ignore_index=True)
df_items = df_items.drop_duplicates(subset=['item_ref'])
item_refs = df_items['item_ref'].tolist()
df_categories = df_category[df_category['table'].str.strip().str.casefold() == 'category']
categories = df_categories['category'].tolist()
resources = set(df_capacity_ledger['resource'].unique())
resources.update(df_usage_1['resource'].unique())
resources.update(df_usage_2['resource'].unique())
resources = sorted(resources)
df_bundles = df_bundle[df_bundle['table'].str.strip().str.casefold() == 'bundle']
bundle_indices = df_bundles.index.tolist()
df_incompat = df_incompatible[df_incompatible['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = list(zip(df_incompat['item_a'], df_incompat['item_b']))
df_requires = df_requires[df_requires['table'].str.strip().str.casefold() == 'requires']
prereq_pairs = list(zip(df_requires['item_ref'], df_requires['prerequisite_ref']))

def to_int(series):
    return series.astype(int)
unit_benefit_cents = df_items.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = df_items.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
authorized = df_items.set_index('item_ref')['authorized'].astype(int).to_dict()
minimum_lot = df_items.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
maximum_order = df_items.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_category = df_items.set_index('item_ref')['category'].to_dict()
item_location = df_items.set_index('item_ref')['location_id'].to_dict()
minimum_quantity = df_categories.set_index('category')['minimum_quantity'].astype(int).to_dict()
maximum_quantity = df_categories.set_index('category')['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = df_categories.set_index('category')['activation_fee_cents'].astype(int).to_dict()
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
df_usage = pd.concat([df_usage_1[df_usage_1['table'].str.strip().str.casefold() == 'usage'], df_usage_2[df_usage_2['table'].str.strip().str.casefold() == 'usage']], ignore_index=True)
df_usage['amount'] = df_usage['amount'].astype(int)
usage = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    usage[i, r] = row['amount']
bundle_item_a = df_bundles['item_a'].to_dict()
bundle_item_b = df_bundles['item_b'].to_dict()
bundle_bonus_cents = df_bundles['bonus_cents'].astype(int).to_dict()
category_items = {c: [] for c in categories}
for i in item_refs:
    c = item_category[i]
    if c in category_items:
        category_items[c].append(i)
resource_items = {r: [] for r in resources}
for i in item_refs:
    r = item_location[i]
    if r in resource_items:
        resource_items[r].append(i)
m = gp.Model('FC_EAST_HVAC_AC_Placement')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_indices, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if authorized[i] == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
for i in item_refs:
    m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
    m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
for c in categories:
    items_in_c = category_items[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= minimum_quantity[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= maximum_quantity[c])
for c in categories:
    items_in_c = category_items[c]
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for r in resources:
    items_in_r = resource_items[r]
    m.addConstr(gp.quicksum((usage.get((i, r), 0) * x_vars[i] for i in items_in_r)) <= resource_capacity.get(r, 0))
for (i, j) in incompat_pairs:
    if i in y_vars and j in y_vars:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, prereq) in prereq_pairs:
    if i in y_vars and prereq in y_vars:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for b in bundle_indices:
    i = bundle_item_a[b]
    j = bundle_item_b[b]
    if i in y_vars and j in y_vars:
        m.addConstr(w_vars[b] <= y_vars[i])
        m.addConstr(w_vars[b] <= y_vars[j])
        m.addConstr(w_vars[b] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(w_vars[b] == 0)
item_obj = gp.quicksum((unit_benefit_cents[i] * x_vars[i] - item_fee_cents[i] * y_vars[i] for i in item_refs))
cat_obj = gp.quicksum((activation_fee_cents[c] * z_vars[c] for c in categories))
bundle_obj = gp.quicksum((bundle_bonus_cents[b] * w_vars[b] for b in bundle_indices))
m.setObjective(item_obj - cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()