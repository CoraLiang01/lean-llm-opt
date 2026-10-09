import gurobipy as gp
import pandas as pd
import numpy as np
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
df_requires = pd.read_csv(f_requires, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity = pd.read_csv(f_capacity, sep=',')
authorized_items = df_item[df_item['authorized'] == 1]['item_ref'].astype(str).unique()
items = list(authorized_items)
categories = df_category['category'].astype(str).unique()
resources_usage = df_usage['resource'].astype(str).unique()
resources_capacity = df_capacity['resource'].astype(str).unique()
resources = sorted(set(resources_usage).union(resources_capacity))
incompat_pairs = []
for (_, row) in df_incompatible.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in items and b in items:
        incompat_pairs.append((a, b))
prereq_pairs = []
for (_, row) in df_requires.iterrows():
    i = str(row['item_ref'])
    p = str(row['prerequisite_ref'])
    if i in items and p in items:
        prereq_pairs.append((i, p))
bundle_tuples = []
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in items and b in items:
        bundle_tuples.append((a, b))
        bundle_bonus[a, b] = int(row['bonus_cents'])
item_df = df_item.set_index('item_ref')
minimum_lot = {}
maximum_order = {}
unit_benefit_cents = {}
item_fee_cents = {}
item_category = {}
for i in items:
    rec = item_df.loc[i]
    minimum_lot[i] = int(rec['minimum_lot'])
    maximum_order[i] = int(rec['maximum_order'])
    unit_benefit_cents[i] = int(rec['unit_benefit_cents'])
    item_fee_cents[i] = int(rec['item_fee_cents'])
    item_category[i] = str(rec['category'])
cat_df = df_category.set_index('category')
minimum_quantity = {}
maximum_quantity = {}
activation_fee_cents = {}
for c in categories:
    rec = cat_df.loc[c]
    minimum_quantity[c] = int(rec['minimum_quantity'])
    maximum_quantity[c] = int(rec['maximum_quantity'])
    activation_fee_cents[c] = int(rec['activation_fee_cents'])
resource_unit = {}
for r in resources:
    units = df_capacity[df_capacity['resource'].astype(str) == r]['unit'].unique()
    if len(units) == 0:
        units = df_usage[df_usage['resource'].astype(str) == r]['unit'].unique()
    if len(units) == 0:
        raise ValueError(f'No unit found for resource {r}')
    resource_unit[r] = units[0]

def get_conversion_factor(from_unit, to_unit):
    from_unit = from_unit.strip().casefold()
    to_unit = to_unit.strip().casefold()
    if from_unit == to_unit:
        return 1.0
    if from_unit == 'kwh' and to_unit == 'wh':
        return 1000.0
    if from_unit == 'wh' and to_unit == 'kwh':
        return 1.0 / 1000.0
    if from_unit == 'hour' and to_unit == 'minute':
        return 60.0
    if from_unit == 'minute' and to_unit == 'hour':
        return 1.0 / 60.0
    if from_unit == 'liter' and to_unit == 'ml':
        return 1000.0
    if from_unit == 'ml' and to_unit == 'liter':
        return 1.0 / 1000.0
    raise ValueError(f'Unknown conversion from {from_unit} to {to_unit}')
resource_usage = {i: {r: 0.0 for r in resources} for i in items}
for (_, row) in df_usage.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    if i not in items or r not in resources:
        continue
    amt = float(row['amount'])
    from_unit = str(row['unit'])
    to_unit = resource_unit[r]
    factor = get_conversion_factor(from_unit, to_unit)
    resource_usage[i][r] += amt * factor
resource_capacity = {}
for r in resources:
    cap_rows = df_capacity[df_capacity['resource'].astype(str) == r]
    total = 0.0
    to_unit = resource_unit[r]
    for (_, row) in cap_rows.iterrows():
        amt = float(row['amount'])
        from_unit = str(row['unit'])
        factor = get_conversion_factor(from_unit, to_unit)
        total += amt * factor
    resource_capacity[r] = total
m = gp.Model('BakeryOrder')
m.setParam('MIPGap', 0.0001)
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, ub={i: maximum_order[i] for i in items}, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in items:
    m.addConstr(x[i] >= minimum_lot[i] * y[i], name=f'lot_lb_{i}')
    m.addConstr(x[i] <= maximum_order[i] * y[i], name=f'lot_ub_{i}')
for c in categories:
    for i in items:
        if item_category[i] == c:
            m.addConstr(z[c] >= y[i], name=f'cat_act_{c}_{i}')
for c in categories:
    m.addConstr(gp.quicksum((x[i] for i in items if item_category[i] == c)) >= minimum_quantity[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items if item_category[i] == c)) <= maximum_quantity[c], name=f'cat_max_{c}')
for r in resources:
    m.addConstr(gp.quicksum((resource_usage[i][r] * x[i] for i in items)) <= resource_capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
for (i, p) in prereq_pairs:
    m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
for (i, j) in bundle_tuples:
    m.addConstr(b[i, j] <= y[i], name=f'bundle_le_yi_{i}_{j}')
    m.addConstr(b[i, j] <= y[j], name=f'bundle_le_yj_{i}_{j}')
    m.addConstr(b[i, j] >= y[i] + y[j] - 1, name=f'bundle_ge_sum_{i}_{j}')
obj = gp.LinExpr()
obj += gp.quicksum((unit_benefit_cents[i] * x[i] for i in items))
obj -= gp.quicksum((item_fee_cents[i] * y[i] for i in items))
obj -= gp.quicksum((activation_fee_cents[c] * z[c] for c in categories))
obj += gp.quicksum((bundle_bonus[i, j] * b[i, j] for (i, j) in bundle_tuples))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.0f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')