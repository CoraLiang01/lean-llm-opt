import gurobipy as gp
import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
df_benefit = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv')
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
df_bundle = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv')
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_capacity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv')
df_capacity['amount'] = df_capacity['amount'].astype(int)
df_capacity['resource'] = df_capacity['resource'].str.strip()
df_category = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv')
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
df_category['category'] = df_category['category'].str.strip()
df_incompat = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv')
df_incompat['item_a'] = df_incompat['item_a'].str.strip()
df_incompat['item_b'] = df_incompat['item_b'].str.strip()
df_item1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv')
df_item2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv')
df_item = pd.concat([df_item1, df_item2], ignore_index=True)
df_item['authorized'] = df_item['authorized'].astype(int)
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
df_item['category'] = df_item['category'].str.strip()
df_item['location_id'] = df_item['location_id'].str.strip()
df_item['item_ref'] = df_item['item_ref'].str.strip()
df_itemfee = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv')
df_itemfee['activation_fee_cents'] = df_itemfee['activation_fee_cents'].astype(int)
df_itemfee['item_ref'] = df_itemfee['item_ref'].str.strip()
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv')
df_requires['item_ref'] = df_requires['item_ref'].str.strip()
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].str.strip()
df_usage1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv')
df_usage2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv')
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['amount'] = df_usage['amount'].astype(int)
df_usage['item_ref'] = df_usage['item_ref'].str.strip()
df_usage['resource'] = df_usage['resource'].str.strip()
df_usage['unit'] = df_usage['unit'].str.strip().str.casefold()
item_rows = df_item.copy()
item_refs = item_rows['item_ref'].unique().tolist()
item_param = {}
for (_, row) in item_rows.iterrows():
    i = row['item_ref']
    item_param[i] = {'category': row['category'], 'location_id': row['location_id'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order'])}
benefit_per_unit = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in item_refs:
    if i not in benefit_per_unit:
        benefit_per_unit[i] = 0
item_fee = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in item_refs:
    if i not in item_fee:
        item_fee[i] = 0
categories = df_category['category'].unique().tolist()
cat_min = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_max = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
item_to_cat = {i: item_param[i]['category'] for i in item_refs}
cat_to_items = {g: [] for g in categories}
for i in item_refs:
    g = item_to_cat[i]
    if g in cat_to_items:
        cat_to_items[g].append(i)
item_to_section = {i: item_param[i]['location_id'] for i in item_refs}
sections = sorted(set(item_to_section.values()))
df_capacity['resource'] = df_capacity['resource'].str.strip()
section_capacity = df_capacity.groupby('resource')['amount'].sum().to_dict()
usage_per_item_section = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    s = row['resource']
    amt = int(row['amount'])
    unit = row['unit']
    if unit == 'liter':
        amt_ml = amt * 1000
    elif unit == 'ml':
        amt_ml = amt
    else:
        raise ValueError(f'Unknown unit {unit} for item_ref {i}')
    usage_per_item_section[i, s] = amt_ml
item_section_usage = {}
for i in item_refs:
    s = item_to_section[i]
    item_section_usage[i, s] = usage_per_item_section.get((i, s), 0)
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    prereq_pairs.append((i, prereq))
bundle_pairs = []
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bonus = int(row['bonus_cents'])
    bundle_pairs.append(((a, b), bonus))
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars([p for (p, _) in bundle_pairs], vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_param[i]['authorized']
    min_lot = item_param[i]['minimum_lot']
    max_order = item_param[i]['maximum_order']
    if auth == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i])
        m.addConstr(x_vars[i] <= max_order * y_vars[i])
        m.addConstr(x_vars[i] >= 0)
        m.addConstr(y_vars[i] <= 1)
        m.addConstr(y_vars[i] >= 0)
for s in sections:
    items_in_s = [i for i in item_refs if item_to_section[i] == s]
    m.addConstr(gp.quicksum((item_section_usage[i, s] * x_vars[i] for i in items_in_s)) <= section_capacity.get(s, 0))
for g in categories:
    items_in_g = cat_to_items[g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_min[g] * z_vars[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_max[g] * z_vars[g])
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g])
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, prereq) in prereq_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for ((a, b), _) in bundle_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(b_vars[a, b] <= y_vars[a])
        m.addConstr(b_vars[a, b] <= y_vars[b])
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
obj_benefit = gp.quicksum((benefit_per_unit[i] * x_vars[i] for i in item_refs))
obj_item_fee = gp.quicksum((item_fee[i] * y_vars[i] for i in item_refs))
obj_cat_fee = gp.quicksum((cat_fee[g] * z_vars[g] for g in categories))
obj_bundle = gp.quicksum((bonus * b_vars[a, b] for ((a, b), bonus) in bundle_pairs if a in item_refs and b in item_refs))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()