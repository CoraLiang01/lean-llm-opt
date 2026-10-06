import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_01/export_01.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_04/export_04.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_05/export_05.csv'
f_items = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_06/export_06.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_02/export_08.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant28/inputs/batch_03/export_09.csv'
df_bundle = pd.read_csv(f_bundle)
df_capacity = pd.read_csv(f_capacity)
df_category = pd.read_csv(f_category)
df_identity = pd.read_csv(f_identity)
df_incompat = pd.read_csv(f_incompat)
df_items = pd.read_csv(f_items)
df_requires = pd.read_csv(f_requires)
df_usage = pd.read_csv(f_usage)
items_df = df_items[df_items['authorized'] == 1].copy()
items = list(items_df['item_ref'])
item_set = set(items)
categories = list(df_category['category'])
cat_set = set(categories)
usage_resources = set(df_usage['resource'])
capacity_resources = set(df_capacity['resource'])
resources = sorted(list(usage_resources | capacity_resources))
bundle_rows = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
bundle_list = []
bundle_bonus = {}
for idx, row in bundle_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        bundle_id = (a, b)
        bundle_list.append(bundle_id)
        bundle_bonus[bundle_id] = int(row['bonus_cents'])
bundles = bundle_list
incompat_rows = df_incompat[df_incompat['table'].str.casefold() == 'incompatible']
incompat_pairs = []
for idx, row in incompat_rows.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_set and b in item_set:
        incompat_pairs.append((a, b))
requires_rows = df_requires[df_requires['table'].str.casefold() == 'requires']
prereq_pairs = []
for idx, row in requires_rows.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    if i in item_set and prereq in item_set:
        prereq_pairs.append((i, prereq))
item_minlot = dict(zip(items_df['item_ref'], items_df['minimum_lot']))
item_maxorder = dict(zip(items_df['item_ref'], items_df['maximum_order']))
item_benefit = dict(zip(items_df['item_ref'], items_df['unit_benefit_cents']))
item_fee = dict(zip(items_df['item_ref'], items_df['item_fee_cents']))
item_category = dict(zip(items_df['item_ref'], items_df['category']))
cat_minqty = dict(zip(df_category['category'], df_category['minimum_quantity']))
cat_maxqty = dict(zip(df_category['category'], df_category['maximum_quantity']))
cat_fee = dict(zip(df_category['category'], df_category['activation_fee_cents']))
unit_to_base = {'liter': ('ml', 1000), 'ml': ('ml', 1), 'kwh': ('wh', 1000), 'wh': ('wh', 1), 'hour': ('minute', 60), 'minute': ('minute', 1)}
usage = {i: {} for i in items}
for idx, row in df_usage.iterrows():
    i = row['item_ref']
    if i not in item_set:
        continue
    r = row['resource']
    amt = row['amount']
    unit = row['unit'].strip().lower()
    base_unit, factor = unit_to_base[unit]
    amt_base = amt * factor
    if r not in usage[i]:
        usage[i][r] = 0
    usage[i][r] += amt_base
capacity = {}
for r in resources:
    rows = df_capacity[df_capacity['resource'] == r]
    total = 0
    for idx, row in rows.iterrows():
        amt = row['amount']
        unit = row['unit'].strip().lower()
        base_unit, factor = unit_to_base[unit]
        amt_base = amt * factor
        total += amt_base
    capacity[r] = total
cat_items = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    cat_items[c].append(i)
m = gp.Model('CENTRAL_FRESH_ProduceOrder')
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'maxorder_{i}')
    m.addConstr(y[i] <= 1, name=f'ybin_{i}')
for c in categories:
    for i in cat_items[c]:
        m.addConstr(y[i] <= z[c], name=f'catlink_{i}_{c}')
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c] * z[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c] * z[c], name=f'catmax_{c}')
for r in resources:
    m.addConstr(gp.quicksum((usage[i].get(r, 0) * x[i] for i in items)) <= capacity[r], name=f'res_{r}')
for i, j in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for i, prereq in prereq_pairs:
    m.addConstr(y[i] <= y[prereq], name=f'prereq_{i}_{prereq}')
for a, b_ in bundles:
    m.addConstr(b[a, b_] <= y[a], name=f'bundle1_{a}_{b_}')
    m.addConstr(b[a, b_] <= y[b_], name=f'bundle2_{a}_{b_}')
    m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle3_{a}_{b_}')
obj = gp.quicksum((item_benefit[i] * x[i] for i in items))
obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
obj += gp.quicksum((bundle_bonus[bb] * b[bb] for bb in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()