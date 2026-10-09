import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
bundle_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'bundle'] for df in dfs if 'table' in df.columns], ignore_index=True)
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in bundle_df.iterrows()]
bonus_cents = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in bundle_df.iterrows()}
capacity_ledger_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'capacity_ledger'] for df in dfs if 'table' in df.columns], ignore_index=True)
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
capacity_by_resource = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
resources = sorted(capacity_by_resource.keys())
category_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'category'] for df in dfs if 'table' in df.columns], ignore_index=True)
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
categories = sorted(category_df['category'].unique())
minimum_quantity = category_df.set_index('category')['minimum_quantity'].to_dict()
maximum_quantity = category_df.set_index('category')['maximum_quantity'].to_dict()
activation_fee_cents = category_df.set_index('category')['activation_fee_cents'].to_dict()
incompatible_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'incompatible'] for df in dfs if 'table' in df.columns], ignore_index=True)
incompatible_pairs = [(row['item_a'], row['item_b']) for (_, row) in incompatible_df.iterrows()]
item_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'item'] for df in dfs if 'table' in df.columns], ignore_index=True)
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_df['unit_benefit_cents'] = item_df['unit_benefit_cents'].astype(int)
item_df['item_fee_cents'] = item_df['item_fee_cents'].astype(int)
authorized_items = item_df[item_df['authorized'] == 1]['item_ref'].tolist()
items = authorized_items
item_category = item_df.set_index('item_ref')['category'].to_dict()
item_minimum_lot = item_df.set_index('item_ref')['minimum_lot'].to_dict()
item_maximum_order = item_df.set_index('item_ref')['maximum_order'].to_dict()
item_unit_benefit_cents = item_df.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee_cents = item_df.set_index('item_ref')['item_fee_cents'].to_dict()
item_location = item_df.set_index('item_ref')['location_id'].to_dict()
requires_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'requires'] for df in dfs if 'table' in df.columns], ignore_index=True)
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in requires_df.iterrows()]
usage_df = pd.concat([df[df['table'].str.strip().str.casefold() == 'usage'] for df in dfs if 'table' in df.columns], ignore_index=True)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    if i not in usage_dict:
        usage_dict[i] = {}
    usage_dict[i][r] = amt
m = gp.Model('FC_EAST_HVAC_Placement')
q_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
g_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
bundle_vars = {}
for (a, b) in bundle_pairs:
    if a in items and b in items:
        bundle_vars[a, b] = m.addVar(vtype=gp.GRB.BINARY, name=f'b_{a}_{b}')
for r in resources:
    m.addConstr(gp.quicksum((usage_dict.get(i, {}).get(r, 0) * q_vars[i] for i in items)) <= capacity_by_resource[r], name=f'cap_{r}')
for i in items:
    m.addConstr(q_vars[i] >= item_minimum_lot[i] * z_vars[i], name=f'minlot_{i}')
    m.addConstr(q_vars[i] <= item_maximum_order[i] * z_vars[i], name=f'maxorder_{i}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= minimum_quantity[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= maximum_quantity[c], name=f'cat_max_{c}')
for c in categories:
    items_in_c = [i for i in items if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(z_vars[i] <= g_vars[c], name=f'catact1_{i}_{c}')
    m.addConstr(g_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)), name=f'catact2_{c}')
for (a, b) in incompatible_pairs:
    if a in items and b in items:
        m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incomp_{a}_{b}')
for (i, prereq) in requires_pairs:
    if i in items and prereq in items:
        m.addConstr(z_vars[i] <= z_vars[prereq], name=f'req_{i}_{prereq}')
for ((a, b), bvar) in bundle_vars.items():
    m.addConstr(bvar <= z_vars[a], name=f'bundle1_{a}_{b}')
    m.addConstr(bvar <= z_vars[b], name=f'bundle2_{a}_{b}')
    m.addConstr(bvar >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{a}_{b}')
obj_expr = gp.quicksum((item_unit_benefit_cents[i] * q_vars[i] - item_fee_cents[i] * z_vars[i] for i in items))
obj_expr -= gp.quicksum((activation_fee_cents[c] * g_vars[c] for c in categories))
obj_expr += gp.quicksum((bonus_cents[a_b] * bundle_vars[a_b] for a_b in bundle_vars))
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
m.optimize()