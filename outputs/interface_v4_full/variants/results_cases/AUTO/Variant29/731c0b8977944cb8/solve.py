import gurobipy as gp
import pandas as pd
import numpy as np
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_02/export_02.csv'
f_items = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_04/export_04.csv'
f_categories = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_05/export_05.csv'
f_incompat = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_01/export_07.csv'
f_bundles = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant29/inputs/batch_03/export_09.csv'
df_requires = pd.read_csv(f_requires, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_items = pd.read_csv(f_items, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
df_categories = pd.read_csv(f_categories, sep=',')
df_incompat = pd.read_csv(f_incompat, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_bundles = pd.read_csv(f_bundles, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
items_df = df_items[df_items['authorized'] == 1].copy()
items = list(items_df['item_ref'])
if len(items) == 0:
    raise ValueError('No authorized items found.')
categories = list(df_categories['category'])
resources_usage = set(df_usage['resource'].unique())
resources_capacity = set(df_capacity['resource'].unique())
resources = sorted(resources_usage | resources_capacity)
bundles_df = df_bundles.copy()
bundles_df = bundles_df[bundles_df['item_a'].isin(items) & bundles_df['item_b'].isin(items)].reset_index(drop=True)
bundles = list(bundles_df.index)
incompat_df = df_incompat.copy()
incompat_df = incompat_df[incompat_df['item_a'].isin(items) & incompat_df['item_b'].isin(items)].reset_index(drop=True)
incompat_pairs = list(incompat_df.itertuples(index=False))
prereq_df = df_requires.copy()
prereq_df = prereq_df[prereq_df['item_ref'].isin(items) & prereq_df['prerequisite_ref'].isin(items)].reset_index(drop=True)
prereq_pairs = list(prereq_df.itertuples(index=False))
item_minlot = dict(zip(items_df['item_ref'], items_df['minimum_lot']))
item_maxorder = dict(zip(items_df['item_ref'], items_df['maximum_order']))
item_benefit = dict(zip(items_df['item_ref'], items_df['unit_benefit_cents']))
item_fee = dict(zip(items_df['item_ref'], items_df['item_fee_cents']))
item_category = dict(zip(items_df['item_ref'], items_df['category']))
cat_minqty = dict(zip(df_categories['category'], df_categories['minimum_quantity']))
cat_maxqty = dict(zip(df_categories['category'], df_categories['maximum_quantity']))
cat_fee = dict(zip(df_categories['category'], df_categories['activation_fee_cents']))
resource_unit = {}
for r in resources:
    units = df_capacity[df_capacity['resource'] == r]['unit'].unique()
    if len(units) != 1:
        raise ValueError(f'Resource {r} has ambiguous units in capacity_ledger: {units}')
    resource_unit[r] = units[0]

def get_conversion_factor(from_unit, to_unit):
    if from_unit == to_unit:
        return 1.0
    if from_unit == 'hour' and to_unit == 'minute':
        return 60.0
    if from_unit == 'minute' and to_unit == 'hour':
        return 1 / 60.0
    if from_unit == 'liter' and to_unit == 'ml':
        return 1000.0
    if from_unit == 'ml' and to_unit == 'liter':
        return 1 / 1000.0
    if from_unit == 'kwh' and to_unit == 'wh':
        return 1000.0
    if from_unit == 'wh' and to_unit == 'kwh':
        return 1 / 1000.0
    raise ValueError(f'Unknown unit conversion: {from_unit} to {to_unit}')
resource_usage = {i: {r: 0.0 for r in resources} for i in items}
for row in df_usage.itertuples(index=False):
    i = getattr(row, 'item_ref')
    r = getattr(row, 'resource')
    if i not in items:
        continue
    amt = getattr(row, 'amount')
    from_unit = getattr(row, 'unit')
    to_unit = resource_unit[r]
    factor = get_conversion_factor(from_unit, to_unit)
    resource_usage[i][r] += amt * factor
resource_capacity = {}
for r in resources:
    cap_rows = df_capacity[df_capacity['resource'] == r]
    total = 0.0
    to_unit = resource_unit[r]
    for row in cap_rows.itertuples(index=False):
        amt = getattr(row, 'amount')
        from_unit = getattr(row, 'unit')
        factor = get_conversion_factor(from_unit, to_unit)
        total += amt * factor
    resource_capacity[r] = total
bundle_bonus = {}
bundle_items = {}
for idx, row in bundles_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_bonus[idx] = row['bonus_cents']
    bundle_items[idx] = (a, b)
cat_items = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    cat_items[c].append(i)
m = gp.Model('BakeryOrderNetBenefit')
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    m.addConstr(x[i] >= y[i] * item_minlot[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= y[i] * item_maxorder[i], name=f'maxorder_{i}')
for c in categories:
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) >= cat_minqty[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in cat_items[c])) <= cat_maxqty[c], name=f'cat_max_{c}')
    for i in cat_items[c]:
        m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
for r in resources:
    m.addConstr(gp.quicksum((resource_usage[i][r] * x[i] for i in items)) <= resource_capacity[r], name=f'res_{r}')
for row in incompat_df.itertuples(index=False):
    i = getattr(row, 'item_a')
    j = getattr(row, 'item_b')
    m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for row in prereq_df.itertuples(index=False):
    i = getattr(row, 'item_ref')
    j = getattr(row, 'prerequisite_ref')
    m.addConstr(y[i] <= y[j], name=f'prereq_{i}_req_{j}')
for idx in bundles:
    i, j = bundle_items[idx]
    m.addConstr(b[idx] <= y[i], name=f'bundle1_{idx}')
    m.addConstr(b[idx] <= y[j], name=f'bundle2_{idx}')
    m.addConstr(b[idx] >= y[i] + y[j] - 1, name=f'bundle3_{idx}')
obj = gp.quicksum((item_benefit[i] * x[i] for i in items))
obj -= gp.quicksum((item_fee[i] * y[i] for i in items))
obj -= gp.quicksum((cat_fee[c] * z[c] for c in categories))
obj += gp.quicksum((bundle_bonus[idx] * b[idx] for idx in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()