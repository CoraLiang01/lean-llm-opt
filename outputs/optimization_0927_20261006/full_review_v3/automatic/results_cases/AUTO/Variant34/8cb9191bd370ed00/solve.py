import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
(df_benefit, df_bundle, df_capacity_ledger, df_category, df_identity, df_incompatible, df_item_1, df_item_2, df_item_fee, df_market, df_requires, df_usage_1, df_usage_2) = dfs
item_tables = [df_item_1, df_item_2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_ref_set = set(item_rows['item_ref'])
item_category = item_rows.set_index('item_ref')['category'].to_dict()
item_authorized = item_rows.set_index('item_ref')['authorized'].to_dict()
item_min_lot = item_rows.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_rows.set_index('item_ref')['maximum_order'].to_dict()
item_location = item_rows.set_index('item_ref')['location_id'].to_dict()
category_rows = df_category.copy()
category_rows['minimum_quantity'] = category_rows['minimum_quantity'].astype(int)
category_rows['maximum_quantity'] = category_rows['maximum_quantity'].astype(int)
category_rows['activation_fee_cents'] = category_rows['activation_fee_cents'].astype(int)
category_set = set(category_rows['category'])
category_min = category_rows.set_index('category')['minimum_quantity'].to_dict()
category_max = category_rows.set_index('category')['maximum_quantity'].to_dict()
category_fee = category_rows.set_index('category')['activation_fee_cents'].to_dict()
df_benefit['amount_cents'] = df_benefit['amount_cents'].astype(int)
benefit_per_item = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in item_ref_set:
    if i not in benefit_per_item:
        benefit_per_item[i] = 0
df_item_fee['activation_fee_cents'] = df_item_fee['activation_fee_cents'].astype(int)
item_fee = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in item_ref_set:
    if i not in item_fee:
        item_fee[i] = 0
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = list(zip(df_bundle['item_a'], df_bundle['item_b']))
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in df_bundle.iterrows()}
incompat_pairs = set()
for (_, row) in df_incompatible.iterrows():
    incompat_pairs.add((row['item_a'], row['item_b']))
    incompat_pairs.add((row['item_b'], row['item_a']))
requires_pairs = set()
for (_, row) in df_requires.iterrows():
    requires_pairs.add((row['item_ref'], row['prerequisite_ref']))
usage_rows = pd.concat([df_usage_1, df_usage_2], ignore_index=True)
usage_rows['amount'] = usage_rows['amount'].astype(int)
usage_rows['amount_ml'] = usage_rows.apply(lambda r: r['amount'] * 1000 if r['unit'].strip().casefold() == 'liter' else r['amount'], axis=1)
usage_per_item_resource = {}
for (_, row) in usage_rows.iterrows():
    usage_per_item_resource[row['item_ref'], row['resource']] = row['amount_ml']
df_capacity_ledger['amount'] = df_capacity_ledger['amount'].astype(int)
df_capacity_ledger['amount_ml'] = df_capacity_ledger['amount']
resource_capacity = df_capacity_ledger.groupby('resource')['amount_ml'].sum().to_dict()
resource_items = {}
for (item, resource) in usage_per_item_resource:
    resource_items.setdefault(resource, set()).add(item)
category_items = {}
for i in item_ref_set:
    c = item_category[i]
    category_items.setdefault(c, set()).add(i)
section_items = {}
for i in item_ref_set:
    loc = item_location[i]
    section_items.setdefault(loc, set()).add(i)
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_ref_set, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_ref_set, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(category_set, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_ref_set:
    if item_authorized[i] == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= item_min_lot[i] * z_vars[i])
        m.addConstr(x_vars[i] <= item_max_order[i] * z_vars[i])
        m.addConstr(x_vars[i] >= 0)
for resource in resource_capacity:
    items = [i for i in item_ref_set if (i, resource) in usage_per_item_resource]
    m.addConstr(gp.quicksum((usage_per_item_resource[i, resource] * x_vars[i] for i in items)) <= resource_capacity[resource])
for c in category_set:
    items = category_items.get(c, set())
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) >= category_min[c])
    m.addConstr(gp.quicksum((x_vars[i] for i in items)) <= category_max[c])
for c in category_set:
    items = category_items.get(c, set())
    m.addConstr(gp.quicksum((z_vars[i] for i in items)) >= w_vars[c])
    m.addConstr(w_vars[c] <= gp.quicksum((z_vars[i] for i in items)))
for (i, j) in incompat_pairs:
    if i in item_ref_set and j in item_ref_set:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, j) in requires_pairs:
    if i in item_ref_set and j in item_ref_set:
        m.addConstr(z_vars[i] <= z_vars[j])
for (a, b) in bundle_pairs:
    if a in item_ref_set and b in item_ref_set:
        m.addConstr(b_vars[a, b] <= z_vars[a])
        m.addConstr(b_vars[a, b] <= z_vars[b])
        m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
obj = gp.quicksum((benefit_per_item[i] * x_vars[i] for i in item_ref_set)) - gp.quicksum((item_fee[i] * z_vars[i] for i in item_ref_set)) - gp.quicksum((category_fee[c] * w_vars[c] for c in category_set)) + gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()