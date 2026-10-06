import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv'
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
df_usage1 = pd.read_csv(f_usage1, sep=',')
df_usage2 = pd.read_csv(f_usage2, sep=',')
item_tables = [df_item1, df_item2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_refs = item_rows['item_ref'].astype(str).unique()
item_refs_set = set(item_refs)
item_param_df = pd.concat(item_tables, ignore_index=True)
item_param_df = item_param_df.drop_duplicates(subset=['item_ref'])
item_param_df['item_ref'] = item_param_df['item_ref'].astype(str)
item_param_df['category'] = item_param_df['category'].astype(str)
item_param_df['authorized'] = item_param_df['authorized'].astype(int)
item_param_df['minimum_lot'] = item_param_df['minimum_lot'].astype(int)
item_param_df['maximum_order'] = item_param_df['maximum_order'].astype(int)
item_to_category = dict(zip(item_param_df['item_ref'], item_param_df['category']))
item_to_authorized = dict(zip(item_param_df['item_ref'], item_param_df['authorized']))
item_to_minlot = dict(zip(item_param_df['item_ref'], item_param_df['minimum_lot']))
item_to_maxorder = dict(zip(item_param_df['item_ref'], item_param_df['maximum_order']))
df_benefit['item_ref'] = df_benefit['item_ref'].astype(str)
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in item_refs:
    if i not in benefit_per_item:
        benefit_per_item[i] = 0
df_item_fee['item_ref'] = df_item_fee['item_ref'].astype(str)
item_fee = dict(zip(df_item_fee['item_ref'], df_item_fee['activation_fee_cents']))
for i in item_refs:
    if i not in item_fee:
        item_fee[i] = 0
df_category['category'] = df_category['category'].astype(str)
category_list = df_category['category'].unique()
category_fee = dict(zip(df_category['category'], df_category['activation_fee_cents']))
category_min = dict(zip(df_category['category'], df_category['minimum_quantity']))
category_max = dict(zip(df_category['category'], df_category['maximum_quantity']))
df_usage1['item_ref'] = df_usage1['item_ref'].astype(str)
df_usage2['item_ref'] = df_usage2['item_ref'].astype(str)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
df_usage['resource'] = df_usage['resource'].astype(str)
df_usage['amount'] = df_usage['amount'].astype(int)
resources = df_usage['resource'].unique()
usage = {(row['item_ref'], row['resource']): row['amount'] for idx, row in df_usage.iterrows()}
for i in item_refs:
    for r in resources:
        if (i, r) not in usage:
            usage[i, r] = 0
df_capacity_ledger['resource'] = df_capacity_ledger['resource'].astype(str)
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
for r in resources:
    if r not in resource_capacity:
        resource_capacity[r] = 0
df_bundle['item_a'] = df_bundle['item_a'].astype(str)
df_bundle['item_b'] = df_bundle['item_b'].astype(str)
bundles = []
bundle_bonus = {}
for idx, row in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs_set and b in item_refs_set:
        bundles.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
df_incompatible['item_a'] = df_incompatible['item_a'].astype(str)
df_incompatible['item_b'] = df_incompatible['item_b'].astype(str)
incompat_pairs = []
for idx, row in df_incompatible.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs_set and b in item_refs_set:
        incompat_pairs.append((a, b))
df_requires['item_ref'] = df_requires['item_ref'].astype(str)
df_requires['prerequisite_ref'] = df_requires['prerequisite_ref'].astype(str)
prereq_pairs = []
for idx, row in df_requires.iterrows():
    i = row['item_ref']
    p = row['prerequisite_ref']
    if i in item_refs_set and p in item_refs_set:
        prereq_pairs.append((i, p))
category_items = {c: [] for c in category_list}
for i in item_refs:
    c = item_to_category[i]
    if c in category_items:
        category_items[c].append(i)

def solve_problem():
    m = gp.Model('NYC_Dev_Module_Portfolio')
    x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
    y = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(category_list, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    obj = gp.quicksum((benefit_per_item[i] * x[i] for i in item_refs))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in item_refs))
    obj -= gp.quicksum((category_fee[c] * z[c] for c in category_list))
    obj += gp.quicksum((bundle_bonus[a, b] * b[a, b] for a, b in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    for i in item_refs:
        auth = item_to_authorized[i]
        minlot = item_to_minlot[i]
        maxorder = item_to_maxorder[i]
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'unauth_{i}')
            m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
        else:
            m.addConstr(x[i] >= minlot * y[i], name=f'minlot_{i}')
            m.addConstr(x[i] <= maxorder * y[i], name=f'maxorder_{i}')
            m.addConstr(x[i] <= maxorder, name=f'maxorder2_{i}')
            m.addConstr(x[i] >= 0, name=f'xnonneg_{i}')
    for r in resources:
        m.addConstr(gp.quicksum((usage[i, r] * x[i] for i in item_refs)) <= resource_capacity[r], name=f'res_{r}')
    for c in category_list:
        items_in_c = category_items[c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= category_min[c] * z[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= category_max[c] * z[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
        m.addConstr(z[c] <= gp.quicksum((y[i] for i in items_in_c)), name=f'cat_zub_{c}')
    for i, j in incompat_pairs:
        m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
    for i, p in prereq_pairs:
        m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
    for a, b_ in bundles:
        m.addConstr(b[a, b_] <= y[a], name=f'bundle1_{a}_{b_}')
        m.addConstr(b[a, b_] <= y[b_], name=f'bundle2_{a}_{b_}')
        m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle3_{a}_{b_}')
    m.optimize()
    return m
m = solve_problem()