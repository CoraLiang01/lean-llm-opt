import gurobipy as gp
import pandas as pd
import numpy as np
import re
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_01.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_02.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_03.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_04/export_04.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_06/export_06.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_01/export_07.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_02/export_08.csv'
f_capacity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant29/inputs/batch_03/export_09.csv'
requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
item_df = pd.read_csv(f_item, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(f_usage, dtype=str, keep_default_na=False)
category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
incompatible_df = pd.read_csv(f_incompatible, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(f_capacity, dtype=str, keep_default_na=False)

def norm_id(x):
    return x.strip().casefold()
item_df['authorized'] = item_df['authorized'].astype(int)
authorized_items_df = item_df[item_df['authorized'] == 1].copy()
authorized_items = set(authorized_items_df['item_ref'])
categories = set(category_df['category'])
usage_resources = set(usage_df['resource'])
capacity_resources = set(capacity_df['resource'])
resources = usage_resources.union(capacity_resources)
incompatible_pairs = []
for (_, row) in incompatible_df.iterrows():
    incompatible_pairs.append((row['item_a'], row['item_b']))
prerequisite_pairs = []
for (_, row) in requires_df.iterrows():
    prerequisite_pairs.append((row['item_ref'], row['prerequisite_ref']))
bundle_tuples = []
for (idx, row) in bundle_df.iterrows():
    bundle_tuples.append((row['item_a'], row['item_b']))
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['unit_benefit_cents'] = item_df['unit_benefit_cents'].astype(int)
item_df['item_fee_cents'] = item_df['item_fee_cents'].astype(int)
item_df['category'] = item_df['category'].astype(str)
item_params = {}
for (_, row) in authorized_items_df.iterrows():
    i = row['item_ref']
    item_params[i] = {'category': row['category'], 'minimum_lot': row['minimum_lot'], 'maximum_order': row['maximum_order'], 'unit_benefit_cents': row['unit_benefit_cents'], 'item_fee_cents': row['item_fee_cents']}
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_params = {}
for (_, row) in category_df.iterrows():
    c = row['category']
    category_params[c] = {'minimum_quantity': row['minimum_quantity'], 'maximum_quantity': row['maximum_quantity'], 'activation_fee_cents': row['activation_fee_cents']}
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_params = {}
for (idx, row) in bundle_df.iterrows():
    bundle_params[row['item_a'], row['item_b']] = row['bonus_cents']
unit_to_base = {'liter': ('ml', 1000), 'ml': ('ml', 1), 'hour': ('minute', 60), 'minute': ('minute', 1), 'kwh': ('wh', 1000), 'wh': ('wh', 1)}
resource_base_unit = {}
for r in resources:
    unit = None
    for (_, row) in capacity_df[capacity_df['resource'] == r].iterrows():
        unit = row['unit']
        break
    if unit is None:
        for (_, row) in usage_df[usage_df['resource'] == r].iterrows():
            unit = row['unit']
            break
    if unit is None:
        raise ValueError(f'Cannot determine unit for resource {r}')
    (base_unit, _) = unit_to_base[unit]
    resource_base_unit[r] = base_unit
usage_df['amount'] = usage_df['amount'].astype(int)
item_resource_usage = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    if i not in authorized_items:
        continue
    amount = row['amount']
    unit = row['unit']
    (base_unit, factor) = unit_to_base[unit]
    amount_base = amount * factor
    if base_unit != resource_base_unit[r]:
        raise ValueError(f'Unit mismatch for resource {r}: {base_unit} vs {resource_base_unit[r]}')
    item_resource_usage[i, r] = amount_base
for i in authorized_items:
    for r in resources:
        if (i, r) not in item_resource_usage:
            item_resource_usage[i, r] = 0
capacity_df['amount'] = capacity_df['amount'].astype(int)
resource_capacity = {}
for r in resources:
    total = 0
    for (_, row) in capacity_df[capacity_df['resource'] == r].iterrows():
        amount = row['amount']
        unit = row['unit']
        (base_unit, factor) = unit_to_base[unit]
        if base_unit != resource_base_unit[r]:
            raise ValueError(f'Unit mismatch in capacity_ledger for resource {r}: {base_unit} vs {resource_base_unit[r]}')
        total += amount * factor
    resource_capacity[r] = total
category_items = {c: [] for c in categories}
for i in authorized_items:
    c = item_params[i]['category']
    category_items[c].append(i)
m = gp.Model('BakeryOrder')
q_vars = m.addVars(authorized_items, vtype=gp.GRB.INTEGER, lb=0, ub={i: item_params[i]['maximum_order'] for i in authorized_items}, name='')
y_vars = m.addVars(authorized_items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_tuples, vtype=gp.GRB.BINARY, name='')
for i in authorized_items:
    min_lot = item_params[i]['minimum_lot']
    max_order = item_params[i]['maximum_order']
    m.addConstr(q_vars[i] >= min_lot * y_vars[i])
    m.addConstr(q_vars[i] <= max_order * y_vars[i])
for i in authorized_items:
    m.addConstr(q_vars[i] >= y_vars[i])
for c in categories:
    for i in category_items[c]:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in category_items[c])))
for c in categories:
    minq = category_params[c]['minimum_quantity']
    maxq = category_params[c]['maximum_quantity']
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) >= minq)
    m.addConstr(gp.quicksum((q_vars[i] for i in category_items[c])) <= maxq)
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * q_vars[i] for i in authorized_items)) <= resource_capacity[r])
for (i, j) in incompatible_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(y_vars[i] + y_vars[j] <= 1)
for (i, j) in prerequisite_pairs:
    if i in authorized_items and j in authorized_items:
        m.addConstr(y_vars[i] <= y_vars[j])
for (i, j) in bundle_tuples:
    if i in authorized_items and j in authorized_items:
        m.addConstr(b_vars[i, j] <= y_vars[i])
        m.addConstr(b_vars[i, j] <= y_vars[j])
        m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1)
    else:
        m.addConstr(b_vars[i, j] == 0)
item_benefit = gp.quicksum((item_params[i]['unit_benefit_cents'] * q_vars[i] - item_params[i]['item_fee_cents'] * y_vars[i] for i in authorized_items))
category_fees = gp.quicksum((category_params[c]['activation_fee_cents'] * z_vars[c] for c in categories))
bundle_bonuses = gp.quicksum((bundle_params[i, j] * b_vars[i, j] for (i, j) in bundle_tuples))
m.setObjective(item_benefit - category_fees + bundle_bonuses, gp.GRB.MAXIMIZE)
m.optimize()