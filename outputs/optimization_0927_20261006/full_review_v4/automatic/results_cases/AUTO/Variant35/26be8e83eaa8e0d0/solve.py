import gurobipy as gp
import pandas as pd
import numpy as np
import re
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_fx = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_itemfee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
items_df = df_item[df_item['authorized'] == 1].copy()
item_refs = items_df['item_ref'].tolist()
item_set = set(item_refs)
platforms = sorted(items_df['location_id'].unique())
categories = df_category['category'].tolist()
category_set = set(categories)
bundles = []
for (_, row) in df_bundle.iterrows():
    bundles.append((row['item_a'], row['item_b']))
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    incompat_pairs.append((row['item_a'], row['item_b']))
requires_pairs = []
for (_, row) in df_requires.iterrows():
    requires_pairs.append((row['item_ref'], row['prerequisite_ref']))
resources = sorted(df_capacity['resource'].unique())
fx_map = {}
for (_, row) in df_fx.iterrows():
    fx_map[row['currency']] = (int(row['usd_cents_numerator']), int(row['denominator']))
df_benefit['amount'] = df_benefit['amount'].astype(int)
benefit_per_unit = {}
for i in item_refs:
    df_ben_i = df_benefit[df_benefit['item_ref'] == i]
    total = 0
    for (_, row) in df_ben_i.iterrows():
        currency = row['currency']
        amount = row['amount']
        (num, denom) = fx_map[currency]
        total += amount * num // denom
    benefit_per_unit[i] = total
item_fee_map = {}
df_itemfee['activation_fee_cents'] = df_itemfee['activation_fee_cents'].astype(int)
for (_, row) in df_itemfee.iterrows():
    item_fee_map[row['item_ref']] = row['activation_fee_cents']
for i in item_refs:
    if i not in item_fee_map:
        item_fee_map[i] = 0
category_min = {}
category_max = {}
category_fee = {}
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
for (_, row) in df_category.iterrows():
    c = row['category']
    category_min[c] = row['minimum_quantity']
    category_max[c] = row['maximum_quantity']
    category_fee[c] = row['activation_fee_cents']
bundle_bonus = {}
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
for (_, row) in df_bundle.iterrows():
    bundle_bonus[row['item_a'], row['item_b']] = row['bonus_cents']
usage_per_unit = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = int(row['amount'])
    unit = row['unit'].strip().upper()
    if unit == 'GB':
        amt_mb = amt * 1000
    elif unit == 'MB':
        amt_mb = amt
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    if i not in usage_per_unit:
        usage_per_unit[i] = {}
    usage_per_unit[i][r] = amt_mb
for i in item_refs:
    if i not in usage_per_unit:
        usage_per_unit[i] = {}
    for r in resources:
        if r not in usage_per_unit[i]:
            usage_per_unit[i][r] = 0
df_capacity['amount'] = df_capacity['amount'].astype(int)
capacity_total = {}
for r in resources:
    amt = df_capacity[df_capacity['resource'] == r]['amount'].sum()
    capacity_total[r] = amt
items_df['minimum_lot'] = items_df['minimum_lot'].astype(int)
items_df['maximum_order'] = items_df['maximum_order'].astype(int)
item_minlot = dict(zip(items_df['item_ref'], items_df['minimum_lot']))
item_maxorder = dict(zip(items_df['item_ref'], items_df['maximum_order']))
item_category = dict(zip(items_df['item_ref'], items_df['category']))
item_platform = dict(zip(items_df['item_ref'], items_df['location_id']))
category_items = {c: [] for c in categories}
for i in item_refs:
    c = item_category[i]
    category_items[c].append(i)
m = gp.Model('GameEditionAllocation')
q_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
g_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_bonus.keys(), vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    m.addConstr(q_vars[i] >= minlot * z_vars[i], name=f'minlot_{i}')
    m.addConstr(q_vars[i] <= maxorder * z_vars[i], name=f'maxorder_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_unit[i][r] * q_vars[i] for i in item_refs if item_platform[i] == r)) <= capacity_total[r], name=f'capacity_{r}')
for c in categories:
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) >= category_min[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) <= category_max[c], name=f'cat_max_{c}')
for c in categories:
    for i in category_items[c]:
        m.addConstr(g_vars[c] >= z_vars[i], name=f'cat_act_{c}_{i}')
for i in item_refs:
    m.addConstr(q_vars[i] <= item_maxorder[i] * z_vars[i], name=f'zlink_{i}')
for (a, b) in bundle_bonus.keys():
    if a in item_refs and b in item_refs:
        m.addConstr(b_vars[a, b] <= z_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(b_vars[a, b] <= z_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle_forbidden_{a}_{b}')
for (a, b) in incompat_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incompat_{a}_{b}')
for (i, pre) in requires_pairs:
    if i in item_refs and pre in item_refs:
        m.addConstr(z_vars[i] <= z_vars[pre], name=f'requires_{i}_{pre}')
obj = gp.quicksum((benefit_per_unit[i] * q_vars[i] for i in item_refs))
obj -= gp.quicksum((item_fee_map[i] * z_vars[i] for i in item_refs))
obj -= gp.quicksum((category_fee[c] * g_vars[c] for c in categories))
obj += gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_bonus.keys()))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()