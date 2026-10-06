import gurobipy as gp
import pandas as pd
import numpy as np
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
df_requires = pd.read_csv(f_requires, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
df_item_auth = df_item[df_item['authorized'] == 1].copy()
authorized_items = df_item_auth['item_ref'].astype(str).tolist()
item_minlot = df_item_auth.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = df_item_auth.set_index('item_ref')['maximum_order'].to_dict()
item_benefit = df_item_auth.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = df_item_auth.set_index('item_ref')['item_fee_cents'].to_dict()
item_category = df_item_auth.set_index('item_ref')['category'].to_dict()
categories = df_category['category'].astype(str).tolist()
cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_usage_auth = df_usage[df_usage['item_ref'].isin(authorized_items)].copy()

def normalize_usage(row):
    amt = row['amount']
    unit = row['unit'].strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'ml':
        return amt
    elif unit == 'hour':
        return amt * 60
    elif unit == 'minute':
        return amt
    elif unit == 'kwh':
        return amt * 1000
    elif unit == 'wh':
        return amt
    else:
        raise ValueError(f'Unknown unit in usage: {unit}')
df_usage_auth['amount_base'] = df_usage_auth.apply(normalize_usage, axis=1)
usage = {}
for _, row in df_usage_auth.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage[i, r] = row['amount_base']
resources = sorted(df_usage_auth['resource'].unique())

def normalize_capacity(row):
    amt = row['amount']
    unit = row['unit'].strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'ml':
        return amt
    elif unit == 'hour':
        return amt * 60
    elif unit == 'minute':
        return amt
    elif unit == 'kwh':
        return amt * 1000
    elif unit == 'wh':
        return amt
    else:
        raise ValueError(f'Unknown unit in capacity_ledger: {unit}')
df_capacity['amount_base'] = df_capacity.apply(normalize_capacity, axis=1)
resource_capacity = df_capacity.groupby('resource')['amount_base'].sum().to_dict()
incompat_pairs = []
for _, row in df_incompat.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in authorized_items and b in authorized_items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for _, row in df_requires.iterrows():
    i = str(row['item_ref'])
    j = str(row['prerequisite_ref'])
    if i in authorized_items and j in authorized_items:
        prereq_pairs.append((i, j))
bundle_list = []
bundle_bonus = {}
for _, row in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in authorized_items and b in authorized_items:
        bundle_id = (a, b)
        bundle_list.append(bundle_id)
        bundle_bonus[bundle_id] = row['bonus_cents']
cat_items = {c: [] for c in categories}
for i in authorized_items:
    c = item_category[i]
    cat_items[c].append(i)
m = gp.Model('BakeryOrderNetBenefit')
x = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'maxorder_{i}')
for c in categories:
    for i in cat_items[c]:
        m.addConstr(z[c] >= y[i], name=f'catact_lb_{c}_{i}')
for c in categories:
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c], name=f'catmin_{c}')
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c], name=f'catmax_{c}')
for r in resources:
    m.addConstr(gp.quicksum((usage.get((i, r), 0) * x[i] for i in authorized_items)) <= resource_capacity[r], name=f'res_{r}')
for i, j in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for i, j in prereq_pairs:
    m.addConstr(y[i] <= y[j], name=f'prereq_{i}_{j}')
for i, j in bundle_list:
    m.addConstr(b[i, j] <= y[i], name=f'bundle_lb1_{i}_{j}')
    m.addConstr(b[i, j] <= y[j], name=f'bundle_lb2_{i}_{j}')
    m.addConstr(b[i, j] >= y[i] + y[j] - 1, name=f'bundle_ub_{i}_{j}')
obj = gp.quicksum((item_benefit[i] * x[i] for i in authorized_items)) - gp.quicksum((item_fee[i] * y[i] for i in authorized_items)) - gp.quicksum((cat_fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_bonus[i, j] * b[i, j] for i, j in bundle_list))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()