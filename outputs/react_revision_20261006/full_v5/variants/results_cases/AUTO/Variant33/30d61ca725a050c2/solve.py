import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv'
f_itemfee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv'
df_benefit = pd.read_csv(f_benefit, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_item1 = pd.read_csv(f_item1, sep=',')
df_item2 = pd.read_csv(f_item2, sep=',')
df_itemfee = pd.read_csv(f_itemfee, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage1 = pd.read_csv(f_usage1, sep=',')
df_usage2 = pd.read_csv(f_usage2, sep=',')
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
item_rows['category'] = item_rows['category'].astype(str)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows = item_rows[item_rows['authorized'] == 1].copy()
items = sorted(item_rows['item_ref'].unique())
item2cat = dict(zip(item_rows['item_ref'], item_rows['category']))
item2minlot = dict(zip(item_rows['item_ref'], item_rows['minimum_lot']))
item2maxorder = dict(zip(item_rows['item_ref'], item_rows['maximum_order']))
df_benefit = df_benefit[df_benefit['table'].str.casefold() == 'benefit']
df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in benefit_per_item:
        raise ValueError(f'Missing benefit for item {i}')
df_itemfee = df_itemfee[df_itemfee['table'].str.casefold() == 'item_fee']
df_itemfee['item_ref'] = df_itemfee['item_ref'].astype(str)
item_fee = dict(zip(df_itemfee['item_ref'], df_itemfee['activation_fee_cents']))
for i in items:
    if i not in item_fee:
        raise ValueError(f'Missing item activation fee for item {i}')
df_usage1 = df_usage1[df_usage1['table'].str.casefold() == 'usage']
df_usage2 = df_usage2[df_usage2['table'].str.casefold() == 'usage']
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['item_ref'] = df_usage['item_ref'].astype(str)
df_usage['resource'] = df_usage['resource'].astype(str)
df_usage['amount'] = df_usage['amount'].astype(int)
resources = sorted(df_capacity['resource'].unique())
usage = {(i, r): 0 for i in items for r in resources}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    if i in items and r in resources:
        usage[i, r] += row['amount']
df_capacity = df_capacity[df_capacity['table'].str.casefold() == 'capacity_ledger']
df_capacity['resource'] = df_capacity['resource'].astype(str)
df_capacity['amount'] = df_capacity['amount'].astype(int)
capacity = df_capacity.groupby('resource')['amount'].sum().to_dict()
for r in resources:
    if r not in capacity:
        raise ValueError(f'Missing capacity for resource {r}')
df_category = df_category[df_category['table'].str.casefold() == 'category']
df_category['category'] = df_category['category'].astype(str)
categories = sorted(df_category['category'].unique())
cat_min = dict(zip(df_category['category'], df_category['minimum_quantity']))
cat_max = dict(zip(df_category['category'], df_category['maximum_quantity']))
cat_fee = dict(zip(df_category['category'], df_category['activation_fee_cents']))
for c in categories:
    if c not in cat_min or c not in cat_max or c not in cat_fee:
        raise ValueError(f'Missing category bounds/fee for {c}')
df_bundle = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
df_bundle['item_a'] = df_bundle['item_a'].astype(str)
df_bundle['item_b'] = df_bundle['item_b'].astype(str)
bundles = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        bundles.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
df_incompat = df_incompat[df_incompat['table'].str.casefold() == 'incompatible']
df_incompat['item_a'] = df_incompat['item_a'].astype(str)
df_incompat['item_b'] = df_incompat['item_b'].astype(str)
incompat_pairs = []
for (_, row) in df_incompat.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a in items and b in items:
        incompat_pairs.append((a, b))
df_requires = df_requires[df_requires['table'].str.casefold() == 'requires']
df_requires['item_ref'] = df_requires['item_ref'].astype(str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    (i, prereq) = (row['item_ref'], row['prerequisite_ref'])
    if i in items and prereq in items:
        prereq_pairs.append((i, prereq))
cat2items = {c: [] for c in categories}
for i in items:
    c = item2cat[i]
    cat2items[c].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x = m.addVars(items, lb=0, ub=[item2maxorder[i] for i in items], vtype=gp.GRB.INTEGER, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    minlot = item2minlot[i]
    maxorder = item2maxorder[i]
    m.addConstr(x[i] >= minlot * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= maxorder * y[i], name=f'maxorder_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage[i, r] * x[i] for i in items)) <= capacity[r], name=f'res_{r}')
for c in categories:
    items_in_c = cat2items[c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_min[c] * z[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_max[c] * z[c], name=f'catmax_{c}')
    for i in items_in_c:
        m.addConstr(y[i] <= z[c], name=f'catlink_{i}_{c}')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, prereq) in prereq_pairs:
    m.addConstr(y[i] <= y[prereq], name=f'prereq_{i}_{prereq}')
for (i, j) in bundles:
    m.addConstr(b[i, j] <= y[i], name=f'bundle1_{i}_{j}')
    m.addConstr(b[i, j] <= y[j], name=f'bundle2_{i}_{j}')
    m.addConstr(b[i, j] >= y[i] + y[j] - 1, name=f'bundle3_{i}_{j}')
benefit_term = gp.quicksum((benefit_per_item[i] * x[i] for i in items))
itemfee_term = gp.quicksum((item_fee[i] * y[i] for i in items))
bundle_term = gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundles))
catfee_term = gp.quicksum((cat_fee[c] * z[c] for c in categories))
m.setObjective(benefit_term - itemfee_term + bundle_term - catfee_term, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()