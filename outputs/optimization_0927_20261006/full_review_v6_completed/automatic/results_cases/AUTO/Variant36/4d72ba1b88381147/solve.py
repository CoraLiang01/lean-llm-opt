import gurobipy as gp
import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
df_bundle = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv')
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_capacity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv')
df_capacity['amount'] = df_capacity['amount'].astype(int)
df_category = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv')
df_category['minimum_quantity'] = df_category['minimum_quantity'].astype(int)
df_category['maximum_quantity'] = df_category['maximum_quantity'].astype(int)
df_category['activation_fee_cents'] = df_category['activation_fee_cents'].astype(int)
df_identity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv')
df_incompat = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv')
df_item1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv')
df_item2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv')
for df in [df_item1, df_item2]:
    df['authorized'] = df['authorized'].astype(int)
    df['minimum_lot'] = df['minimum_lot'].astype(int)
    df['maximum_order'] = df['maximum_order'].astype(int)
    df['unit_benefit_cents'] = df['unit_benefit_cents'].astype(int)
    df['item_fee_cents'] = df['item_fee_cents'].astype(int)
df_items = pd.concat([df_item1, df_item2], ignore_index=True)
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv')
df_usage1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv')
df_usage2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv')
for df in [df_usage1, df_usage2]:
    df['amount'] = df['amount'].astype(int)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
item_keys = df_items['item_ref'].unique()
item_keys_set = set(item_keys)
category_keys = df_category['category'].unique()
category_keys_set = set(category_keys)
resource_keys = pd.concat([df_capacity['resource'], df_usage['resource']]).unique()
resource_keys_set = set(resource_keys)
bundle_pairs = [(row['item_a'], row['item_b']) for (_, row) in df_bundle.iterrows() if row['item_a'] in item_keys_set and row['item_b'] in item_keys_set]
incompat_pairs = [(row['item_a'], row['item_b']) for (_, row) in df_incompat.iterrows() if row['item_a'] in item_keys_set and row['item_b'] in item_keys_set]
requires_pairs = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in df_requires.iterrows() if row['item_ref'] in item_keys_set and row['prerequisite_ref'] in item_keys_set]
item_param_cols = ['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order', 'location_id', 'unit_benefit_cents', 'item_fee_cents']
df_items = df_items.drop_duplicates(subset=['item_ref'], keep='first')
item_params = df_items.set_index('item_ref')[item_param_cols[1:]].to_dict(orient='index')
cat_param_cols = ['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents']
df_category = df_category.drop_duplicates(subset=['category'], keep='first')
cat_params = df_category.set_index('category')[cat_param_cols[1:]].to_dict(orient='index')
capacity_by_resource = df_capacity.groupby('resource')['amount'].sum().to_dict()
usage_by_item_resource = df_usage.groupby(['item_ref', 'resource'])['amount'].sum().to_dict()
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in df_bundle.iterrows() if row['item_a'] in item_keys_set and row['item_b'] in item_keys_set}
m = gp.Model('FC_EAST_HVAC_Placement')
x_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(category_keys, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_keys:
    p = item_params[i]
    authorized = int(p['authorized'])
    min_lot = int(p['minimum_lot'])
    max_order = int(p['maximum_order'])
    if not authorized:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= max_order * y_vars[i], name=f'maxorder_{i}')
for r in resource_keys:
    usage_sum = gp.quicksum((usage_by_item_resource.get((i, r), 0) * x_vars[i] for i in item_keys))
    cap = capacity_by_resource.get(r, 0)
    m.addConstr(usage_sum <= cap, name=f'cap_{r}')
for g in category_keys:
    minq = cat_params[g]['minimum_quantity']
    maxq = cat_params[g]['maximum_quantity']
    items_in_g = [i for i in item_keys if item_params[i]['category'] == g]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= minq, name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= maxq, name=f'cat_max_{g}')
for g in category_keys:
    items_in_g = [i for i in item_keys if item_params[i]['category'] == g]
    for i in items_in_g:
        m.addConstr(y_vars[i] <= z_vars[g], name=f'catact1_{g}_{i}')
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in items_in_g)), name=f'catact2_{g}')
for (i, j) in incompat_pairs:
    m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, k) in requires_pairs:
    m.addConstr(y_vars[i] <= y_vars[k], name=f'requires_{i}_{k}')
for (i, j) in bundle_pairs:
    m.addConstr(b_vars[i, j] <= y_vars[i], name=f'bundle1_{i}_{j}')
    m.addConstr(b_vars[i, j] <= y_vars[j], name=f'bundle2_{i}_{j}')
    m.addConstr(b_vars[i, j] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{i}_{j}')
item_obj = gp.quicksum((int(item_params[i]['unit_benefit_cents']) * x_vars[i] - int(item_params[i]['item_fee_cents']) * y_vars[i] for i in item_keys))
bundle_obj = gp.quicksum((bundle_bonus[i, j] * b_vars[i, j] for (i, j) in bundle_pairs))
cat_obj = gp.quicksum((int(cat_params[g]['activation_fee_cents']) * z_vars[g] for g in category_keys))
m.setObjective(item_obj + bundle_obj - cat_obj, gp.GRB.MAXIMIZE)
m.optimize()