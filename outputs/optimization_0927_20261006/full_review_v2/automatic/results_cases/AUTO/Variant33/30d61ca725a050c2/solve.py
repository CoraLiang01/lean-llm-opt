import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_item1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_item2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_itemfee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
df_usage1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', dtype=str, keep_default_na=False)
df_usage2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv', dtype=str, keep_default_na=False)

def get_authorized_items(df):
    return df[df['authorized'].astype(int) == 1][['item_ref', 'category', 'minimum_lot', 'maximum_order']].copy()
items1 = get_authorized_items(df_item1)
items2 = get_authorized_items(df_item2)
items_df = pd.concat([items1, items2], axis=0, ignore_index=True)
items_df['item_ref'] = items_df['item_ref'].apply(norm_str)
items_df['category'] = items_df['category'].apply(norm_str)
items_df['minimum_lot'] = items_df['minimum_lot'].astype(int)
items_df['maximum_order'] = items_df['maximum_order'].astype(int)
item_set = set(items_df['item_ref'])
item_to_category = dict(zip(items_df['item_ref'], items_df['category']))
item_min_lot = dict(zip(items_df['item_ref'], items_df['minimum_lot']))
item_max_order = dict(zip(items_df['item_ref'], items_df['maximum_order']))
df_category['category'] = df_category['category'].apply(norm_str)
category_set = set(df_category['category'])
cat_min_qty = dict(zip(df_category['category'], df_category['minimum_quantity'].astype(int)))
cat_max_qty = dict(zip(df_category['category'], df_category['maximum_quantity'].astype(int)))
cat_fee = dict(zip(df_category['category'], df_category['activation_fee_cents'].astype(int)))
df_benefit['item_ref'] = df_benefit['item_ref'].apply(norm_str)
benefit_sum = df_benefit.groupby('item_ref')['amount_cents'].apply(lambda x: x.astype(int).sum()).to_dict()
item_benefit = {i: benefit_sum.get(i, 0) for i in item_set}
df_itemfee['item_ref'] = df_itemfee['item_ref'].apply(norm_str)
item_fee = dict(zip(df_itemfee['item_ref'], df_itemfee['activation_fee_cents'].astype(int)))
item_fee = {i: item_fee.get(i, 0) for i in item_set}
df_usage1['item_ref'] = df_usage1['item_ref'].apply(norm_str)
df_usage2['item_ref'] = df_usage2['item_ref'].apply(norm_str)
df_usage1['resource'] = df_usage1['resource'].apply(norm_str)
df_usage2['resource'] = df_usage2['resource'].apply(norm_str)
df_usage1['amount'] = df_usage1['amount'].astype(int)
df_usage2['amount'] = df_usage2['amount'].astype(int)
usage_df = pd.concat([df_usage1, df_usage2], axis=0, ignore_index=True)
usage_df = usage_df[usage_df['item_ref'].isin(item_set)]
resource_set = set(usage_df['resource'])
item_resource_usage = {}
for (_, row) in usage_df.iterrows():
    key = (row['item_ref'], row['resource'])
    item_resource_usage[key] = int(row['amount'])
df_capacity['resource'] = df_capacity['resource'].apply(norm_str)
df_capacity['amount'] = df_capacity['amount'].astype(int)
capacity_df = df_capacity.groupby('resource')['amount'].sum()
resource_capacity = capacity_df.to_dict()
resource_capacity = {r: resource_capacity[r] for r in resource_set if r in resource_capacity}
df_bundle['item_a'] = df_bundle['item_a'].apply(norm_str)
df_bundle['item_b'] = df_bundle['item_b'].apply(norm_str)
bundle_list = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        bundle_list.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
bundle_set = set(bundle_list)
df_incompat['item_a'] = df_incompat['item_a'].apply(norm_str)
df_incompat['item_b'] = df_incompat['item_b'].apply(norm_str)
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        incompat_pairs.append((a, b))
df_requires['item_ref'] = df_requires['item_ref'].apply(norm_str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].apply(norm_str)
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in item_set and p in item_set:
        prereq_pairs.append((i, p))
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_set, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_set, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_set, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_set, vtype=gp.GRB.BINARY, name='')
for i in item_set:
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'minlot_{i}')
    m.addConstr(x_vars[i] <= max_order * y_vars[i], name=f'maxorder_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'xnonneg_{i}')
for g in category_set:
    items_in_g = [i for i in item_set if item_to_category[i] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_min_qty[g], name=f'catmin_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_max_qty[g], name=f'catmax_{g}')
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'catlink_{g}_{i}')
    m.addConstr(gp.quicksum((y_vars[i] for i in items_in_g)) >= z_vars[g], name=f'catz_lb_{g}')
for r in resource_set:
    m.addConstr(gp.quicksum((item_resource_usage.get((i, r), 0) * x_vars[i] for i in item_set)) <= resource_capacity[r], name=f'rescap_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, p) in prereq_pairs:
    m.addConstr(y_vars[i] <= y_vars[p], name=f'prereq_{i}_{p}')
for (a, b) in bundle_set:
    m.addConstr(w_vars[a, b] <= y_vars[a], name=f'bundle1_{a}_{b}')
    m.addConstr(w_vars[a, b] <= y_vars[b], name=f'bundle2_{a}_{b}')
    m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle3_{a}_{b}')
obj = gp.quicksum((item_benefit[i] * x_vars[i] for i in item_set)) - gp.quicksum((item_fee[i] * y_vars[i] for i in item_set)) - gp.quicksum((cat_fee[g] * z_vars[g] for g in category_set)) + gp.quicksum((bundle_bonus[a, b] * w_vars[a, b] for (a, b) in bundle_set))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()