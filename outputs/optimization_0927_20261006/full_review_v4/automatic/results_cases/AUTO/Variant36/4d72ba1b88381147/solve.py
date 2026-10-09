import gurobipy as gp
import pandas as pd
import numpy as np
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]
bundle_df = dfs[0]
bundle_rows = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_keys = bundle_rows.index.tolist()
bundles = []
for (idx, row) in bundle_rows.iterrows():
    a = row['item_a'].strip()
    b = row['item_b'].strip()
    bonus = int(row['bonus_cents'])
    bundles.append({'item_a': a, 'item_b': b, 'bonus_cents': bonus})
capacity_ledger_df = dfs[1]
cap_rows = capacity_ledger_df[capacity_ledger_df['table'].str.strip().str.casefold() == 'capacity_ledger']
resource_caps = {}
for (r, group) in cap_rows.groupby('resource'):
    resource_caps[r.strip()] = group['amount'].astype(int).sum()
category_df = dfs[2]
cat_rows = category_df[category_df['table'].str.strip().str.casefold() == 'category']
categories = cat_rows['category'].apply(str.strip).tolist()
category_min = dict(zip(cat_rows['category'].apply(str.strip), cat_rows['minimum_quantity'].astype(int)))
category_max = dict(zip(cat_rows['category'].apply(str.strip), cat_rows['maximum_quantity'].astype(int)))
category_fee = dict(zip(cat_rows['category'].apply(str.strip), cat_rows['activation_fee_cents'].astype(int)))
incompat_df = dfs[4]
incompat_rows = incompat_df[incompat_df['table'].str.strip().str.casefold() == 'incompatible']
incompat_pairs = set()
for (idx, row) in incompat_rows.iterrows():
    i = row['item_a'].strip()
    j = row['item_b'].strip()
    incompat_pairs.add((i, j))
    incompat_pairs.add((j, i))
item_dfs = []
for i in [5, 6]:
    df = dfs[i]
    item_rows = df[df['table'].str.strip().str.casefold() == 'item']
    item_dfs.append(item_rows)
all_items_df = pd.concat(item_dfs, ignore_index=True)
item_ids = all_items_df['item_ref'].apply(str.strip).tolist()
item_ids_set = set(item_ids)

def get_item_param_dict(col, dtype=int):
    return dict(zip(all_items_df['item_ref'].apply(str.strip), all_items_df[col].astype(dtype)))
item_authorized = get_item_param_dict('authorized', int)
item_min_lot = get_item_param_dict('minimum_lot', int)
item_max_order = get_item_param_dict('maximum_order', int)
item_category = dict(zip(all_items_df['item_ref'].apply(str.strip), all_items_df['category'].apply(str.strip)))
item_location = dict(zip(all_items_df['item_ref'].apply(str.strip), all_items_df['location_id'].apply(str.strip)))
item_unit_benefit = get_item_param_dict('unit_benefit_cents', int)
item_fee = get_item_param_dict('item_fee_cents', int)
resources = sorted(set(cap_rows['resource'].apply(str.strip)))
usage_dfs = []
for i in [9, 10]:
    df = dfs[i]
    usage_rows = df[df['table'].str.strip().str.casefold() == 'usage']
    usage_dfs.append(usage_rows)
all_usage_df = pd.concat(usage_dfs, ignore_index=True)
usage = {i: {r: 0 for r in resources} for i in item_ids}
for (idx, row) in all_usage_df.iterrows():
    i = row['item_ref'].strip()
    r = row['resource'].strip()
    amt = int(row['amount'])
    if i in usage and r in usage[i]:
        usage[i][r] += amt
    elif i in item_ids and r in resources:
        usage[i][r] = amt
requires_df = dfs[8]
requires_rows = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
requires_pairs = []
for (idx, row) in requires_rows.iterrows():
    i = row['item_ref'].strip()
    j = row['prerequisite_ref'].strip()
    requires_pairs.append((i, j))
bundle_indices = list(range(len(bundles)))
m = gp.Model('FC_EAST_HVAC_Placement')
x_vars = m.addVars(item_ids, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_indices, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    if item_authorized[i] == 0:
        m.addConstr(x_vars[i] == 0, name=f'auth_x_{i}')
        m.addConstr(y_vars[i] == 0, name=f'auth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= item_min_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= item_max_order[i] * y_vars[i], name=f'maxorder_{i}')
        m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
for c in categories:
    items_in_c = [i for i in item_ids if item_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_min[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_max[c], name=f'cat_max_{c}')
for c in categories:
    items_in_c = [i for i in item_ids if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c], name=f'catact1_{i}_{c}')
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)), name=f'catact2_{c}')
for r in resources:
    m.addConstr(gp.quicksum((usage[i][r] * x_vars[i] for i in item_ids)) <= resource_caps[r], name=f'cap_{r}')
for (i, j) in incompat_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in requires_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(y_vars[i] <= y_vars[j], name=f'requires_{i}_{j}')
for (b_idx, bundle) in enumerate(bundles):
    i = bundle['item_a']
    j = bundle['item_b']
    if i in item_ids and j in item_ids:
        m.addConstr(w_vars[b_idx] <= y_vars[i], name=f'bundle1_{b_idx}')
        m.addConstr(w_vars[b_idx] <= y_vars[j], name=f'bundle2_{b_idx}')
        m.addConstr(w_vars[b_idx] >= y_vars[i] + y_vars[j] - 1, name=f'bundle3_{b_idx}')
    else:
        m.addConstr(w_vars[b_idx] == 0, name=f'bundle0_{b_idx}')
item_obj = gp.quicksum((item_unit_benefit[i] * x_vars[i] - item_fee[i] * y_vars[i] for i in item_ids))
cat_obj = gp.quicksum((category_fee[c] * z_vars[c] for c in categories))
bundle_obj = gp.quicksum((bundles[b_idx]['bonus_cents'] * w_vars[b_idx] for b_idx in bundle_indices))
m.setObjective(item_obj - cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()