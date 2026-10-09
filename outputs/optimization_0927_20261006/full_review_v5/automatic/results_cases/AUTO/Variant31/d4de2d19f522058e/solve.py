import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
bundle_df = dfs[0]
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
capacity_ledger_df = dfs[1]
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
category_df = dfs[2]
category_df = category_df[category_df['table'].str.strip().str.casefold() == 'category']
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
identity_df = dfs[3]
identity_df = identity_df[identity_df['table'].str.strip().str.casefold() == 'identity']
incompatible_df = dfs[4]
incompatible_df = incompatible_df[incompatible_df['table'].str.strip().str.casefold() == 'incompatible']
item_df = dfs[5]
item_df = item_df[item_df['table'].str.strip().str.casefold() == 'item']
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['unit_benefit_cents'] = item_df['unit_benefit_cents'].astype(int)
item_df['item_fee_cents'] = item_df['item_fee_cents'].astype(int)
requires_df = dfs[7]
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
usage_df = dfs[8]
usage_df = usage_df[usage_df['table'].str.strip().str.casefold() == 'usage']
usage_df['amount'] = usage_df['amount'].astype(int)
item_keys = list(item_df['item_ref'])
category_keys = list(category_df['category'])
resource_keys = sorted(set(capacity_ledger_df['resource'].unique()) | set(usage_df['resource'].unique()))
bundle_keys = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
incompatible_keys = [(row['item_a'], row['item_b']) for (_, row) in incompatible_df.iterrows()]
requires_keys = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
item_authorized = item_df.set_index('item_ref')['authorized'].to_dict()
item_min_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_max_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_unit_benefit = item_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = item_df.set_index('item_ref')['item_fee_cents'].to_dict()
item_category = item_df.set_index('item_ref')['category'].to_dict()
cat_min_qty = category_df.set_index('category')['minimum_quantity'].to_dict()
cat_max_qty = category_df.set_index('category')['maximum_quantity'].to_dict()
cat_activation_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    usage_dict[i, r] = int(row['amount'])
resource_capacity = {}
for r in resource_keys:
    cap_rows = capacity_ledger_df[capacity_ledger_df['resource'] == r]
    total = cap_rows['amount'].astype(int).sum()
    resource_capacity[r] = total
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_keys, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_keys:
    auth = item_authorized[i]
    min_lot = item_min_lot[i]
    max_order = item_max_order[i]
    if auth == 1:
        m.addConstr(q_vars[i] >= min_lot * z_vars[i])
        m.addConstr(q_vars[i] <= max_order * z_vars[i])
        m.addConstr(q_vars[i] >= z_vars[i])
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for r in resource_keys:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * q_vars[i] for i in item_keys)) <= resource_capacity[r])
for c in category_keys:
    items_in_c = [i for i in item_keys if item_category[i] == c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= cat_min_qty[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= cat_max_qty[c])
    for i in items_in_c:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)))
for (i, j) in incompatible_keys:
    if i in item_keys and j in item_keys:
        m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, prereq) in requires_keys:
    if i in item_keys and prereq in item_keys:
        m.addConstr(z_vars[i] <= z_vars[prereq])
for (i, j) in bundle_keys:
    if i in item_keys and j in item_keys:
        m.addConstr(b_vars[i, j] <= z_vars[i])
        m.addConstr(b_vars[i, j] <= z_vars[j])
        m.addConstr(b_vars[i, j] >= z_vars[i] + z_vars[j] - 1)
    else:
        m.addConstr(b_vars[i, j] == 0)
item_obj = gp.quicksum((item_unit_benefit[i] * q_vars[i] - item_fee[i] * z_vars[i] for i in item_keys))
cat_obj = gp.quicksum((cat_activation_fee[c] * y_vars[c] for c in category_keys))
bundle_obj = gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_keys))
m.setObjective(item_obj - cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()