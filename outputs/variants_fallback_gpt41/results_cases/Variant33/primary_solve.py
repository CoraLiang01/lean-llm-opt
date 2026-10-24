import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', 'benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', 'benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', 'benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', 'benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', 'benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', 'benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', 'benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', 'benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', 'benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', 'benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', 'benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', 'benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
benefit_df = dfs[0]
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
bundle_df = dfs[1]
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
capacity_ledger_df = dfs[2]
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
resource_capacity = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
resources = list(resource_capacity.keys())
category_df = dfs[3]
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = category_df['category'].tolist()
category_min = category_df.set_index('category')['minimum_quantity'].to_dict()
category_max = category_df.set_index('category')['maximum_quantity'].to_dict()
category_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
incompatible_df = dfs[5]
incompatible_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompatible_df.iterrows()]
item_df1 = dfs[6]
item_df2 = dfs[7]
item_df = pd.concat([item_df1, item_df2], ignore_index=True)
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df = item_df.drop_duplicates(subset=['item_ref'])
items = item_df['item_ref'].tolist()
item_category = item_df.set_index('item_ref')['category'].to_dict()
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_minlot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_fee_df = dfs[8]
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
requires_df = dfs[10]
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
usage_df1 = dfs[11]
usage_df2 = dfs[12]
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    if i not in usage_dict:
        usage_dict[i] = {}
    usage_dict[i][r] = amt
m = gp.Model('NY_Development_Module_Portfolio')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in items:
    if item_authorized[i] == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= item_minlot[i] * y_vars[i])
        m.addConstr(x_vars[i] <= item_maxorder[i] * y_vars[i])
        m.addConstr(x_vars[i] >= 0)
        m.addConstr(y_vars[i] <= 1)
        m.addConstr(y_vars[i] >= 0)
for r in resources:
    m.addConstr(gp.quicksum((usage_dict.get(i, {}).get(r, 0) * x_vars[i] for i in items)) <= resource_capacity[r])
for g in categories:
    items_in_g = [i for i in items if item_category[i] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= category_min[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= category_max[g])
for g in categories:
    items_in_g = [i for i in items if item_category[i] == g]
    for i in items_in_g:
        m.addConstr(z_vars[g] >= y_vars[i])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)))
for (i, j) in incompatible_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in requires_pairs:
    if i in items and j in items:
        m.addConstr(y_vars[i] <= y_vars[j])
for (i, j) in bundle_pairs:
    if i in items and j in items:
        m.addConstr(b_vars[i, j] <= y_vars[i])
        m.addConstr(b_vars[i, j] <= y_vars[j])
        m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(b_vars[i, j] == 0)
obj_benefit = gp.quicksum((item_benefit.get(i, 0) * x_vars[i] for i in items))
obj_item_fee = gp.quicksum((item_fee.get(i, 0) * y_vars[i] for i in items))
obj_category_fee = gp.quicksum((category_fee[g] * z_vars[g] for g in categories))
obj_bundle_bonus = gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
m.setObjective(obj_benefit - obj_item_fee - obj_category_fee + obj_bundle_bonus, gp.GRB.MAXIMIZE)
m.optimize()