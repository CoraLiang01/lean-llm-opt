import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
df_benefit = dfs[0]
df_bundle = dfs[1]
df_capacity_ledger = dfs[2]
df_category = dfs[3]
df_identity = dfs[4]
df_incompatible = dfs[5]
df_item_1 = dfs[6]
df_item_2 = dfs[7]
df_item_fee = dfs[8]
df_market = dfs[9]
df_requires = dfs[10]
df_usage_1 = dfs[11]
df_usage_2 = dfs[12]
item_tables = [df_item_1, df_item_2]
item_rows = []
for df in item_tables:
    for _, row in df.iterrows():
        item_rows.append({'item_ref': row['item_ref'], 'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': row['location_id']})
item_keys = [(r['item_ref'], r['location_id']) for r in item_rows]
item_key_set = set(item_keys)
item_attr = {(r['item_ref'], r['location_id']): r for r in item_rows}
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
benefit_per_itemref = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit_per_item = {}
for key in item_key_set:
    item_ref = key[0]
    benefit_per_item[key] = benefit_per_itemref.get(item_ref, 0)
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
item_fee_per_itemref = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
item_fee_per_item = {}
for key in item_key_set:
    item_ref = key[0]
    item_fee_per_item[key] = item_fee_per_itemref.get(item_ref, 0)
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
category_set = set(df_category['category'])
category_min = df_category.set_index('category')['minimum_quantity'].to_dict()
category_max = df_category.set_index('category')['maximum_quantity'].to_dict()
category_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for _, row in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = row['bonus_cents']
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
resource_capacity = df_capacity_ledger.groupby('resource')['amount'].sum().to_dict()
resource_set = set(resource_capacity.keys())
usage_rows = []
for df in [df_usage_1, df_usage_2]:
    for _, row in df.iterrows():
        usage_rows.append({'item_ref': row['item_ref'], 'resource': row['resource'], 'amount': int(row['amount']), 'unit': row['unit']})
for r in usage_rows:
    if r['unit'].strip().casefold() == 'liter':
        r['amount_ml'] = r['amount'] * 1000
    elif r['unit'].strip().casefold() == 'ml':
        r['amount_ml'] = r['amount']
    else:
        raise ValueError(f"Unknown unit in usage: {r['unit']}")
usage_per_itemref_resource = {}
for r in usage_rows:
    usage_per_itemref_resource[r['item_ref'], r['resource']] = r['amount_ml']
usage_per_item_resource = {}
for key in item_key_set:
    item_ref, location_id = key
    for resource in resource_set:
        usage_per_item_resource[key, resource] = usage_per_itemref_resource.get((item_ref, resource), 0)
min_lot = {}
max_order = {}
authorized = {}
item_category = {}
for key in item_key_set:
    attr = item_attr[key]
    min_lot[key] = attr['minimum_lot']
    max_order[key] = attr['maximum_order']
    authorized[key] = attr['authorized']
    item_category[key] = attr['category']
incompat_pairs = []
for _, row in df_incompatible.iterrows():
    a = row['item_a']
    b = row['item_b']
    incompat_pairs.append((a, b))
itemref_to_keys = {}
for key in item_key_set:
    item_ref, location_id = key
    itemref_to_keys.setdefault(item_ref, []).append(key)
prereq_pairs = []
for _, row in df_requires.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    prereq_pairs.append((i, prereq))
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_key_set, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_key_set, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_set, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for key in item_key_set:
    if authorized[key] == 0:
        m.addConstr(x_vars[key] == 0)
        m.addConstr(y_vars[key] == 0)
    else:
        m.addConstr(x_vars[key] >= min_lot[key] * y_vars[key])
        m.addConstr(x_vars[key] <= max_order[key] * y_vars[key])
for resource in resource_set:
    m.addConstr(gp.quicksum((usage_per_item_resource[key, resource] * x_vars[key] for key in item_key_set)) <= resource_capacity[resource])
for c in category_set:
    keys_in_c = [key for key in item_key_set if item_category[key] == c]
    m.addConstr(gp.quicksum((x_vars[key] for key in keys_in_c)) >= category_min[c])
    m.addConstr(gp.quicksum((x_vars[key] for key in keys_in_c)) <= category_max[c])
for c in category_set:
    keys_in_c = [key for key in item_key_set if item_category[key] == c]
    for key in keys_in_c:
        m.addConstr(y_vars[key] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[key] for key in keys_in_c)))
for item_a, item_b in incompat_pairs:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    for ka in keys_a:
        for kb in keys_b:
            m.addConstr(y_vars[ka] + y_vars[kb] <= 1)
for item_i, prereq in prereq_pairs:
    keys_i = itemref_to_keys.get(item_i, [])
    keys_prereq = itemref_to_keys.get(prereq, [])
    for ki in keys_i:
        for kp in keys_prereq:
            m.addConstr(y_vars[ki] <= y_vars[kp])
for item_a, item_b in bundle_pairs:
    keys_a = itemref_to_keys.get(item_a, [])
    keys_b = itemref_to_keys.get(item_b, [])
    for ka in keys_a:
        for kb in keys_b:
            m.addConstr(b_vars[item_a, item_b] <= y_vars[ka])
            m.addConstr(b_vars[item_a, item_b] <= y_vars[kb])
            m.addConstr(b_vars[item_a, item_b] >= y_vars[ka] + y_vars[kb] - 1)
obj_benefit = gp.quicksum((benefit_per_item[key] * x_vars[key] for key in item_key_set))
obj_item_fee = gp.quicksum((item_fee_per_item[key] * y_vars[key] for key in item_key_set))
obj_category_fee = gp.quicksum((category_fee[c] * z_vars[c] for c in category_set))
obj_bundle = gp.quicksum((bundle_bonus[p] * b_vars[p] for p in bundle_pairs))
m.setObjective(obj_benefit - obj_item_fee - obj_category_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()