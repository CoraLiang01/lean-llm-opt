import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
df_item_6 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_item_7 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_usage_10 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
df_usage_11 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
item_tables = [df_item_6, df_item_7]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref_norm'] = item_rows['item_ref'].apply(norm_str)
item_rows['category_norm'] = item_rows['category'].apply(norm_str)
item_rows['location_id_norm'] = item_rows['location_id'].apply(norm_str)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows['unit_benefit_cents'] = item_rows['unit_benefit_cents'].astype(int)
item_rows['item_fee_cents'] = item_rows['item_fee_cents'].astype(int)
I = list(item_rows['item_ref_norm'])
I_set = set(I)
C = list(df_category['category'].apply(norm_str))
C_set = set(C)
R = set(df_capacity['resource'].apply(norm_str))
B = list(df_bundle.apply(lambda row: (norm_str(row['item_a']), norm_str(row['item_b'])), axis=1))
B_idx = list(range(len(B)))
incompat_pairs = list(df_incompat.apply(lambda row: (norm_str(row['item_a']), norm_str(row['item_b'])), axis=1))
requires_pairs = list(df_requires.apply(lambda row: (norm_str(row['item_ref']), norm_str(row['prerequisite_ref'])), axis=1))
category_min = dict(zip(df_category['category'].apply(norm_str), df_category['minimum_quantity'].astype(int)))
category_max = dict(zip(df_category['category'].apply(norm_str), df_category['maximum_quantity'].astype(int)))
category_fee = dict(zip(df_category['category'].apply(norm_str), df_category['activation_fee_cents'].astype(int)))
bundle_bonus = {}
for (idx, row) in df_bundle.iterrows():
    a = norm_str(row['item_a'])
    b = norm_str(row['item_b'])
    bundle_bonus[a, b] = int(row['bonus_cents'])
item_min_lot = dict(zip(item_rows['item_ref_norm'], item_rows['minimum_lot']))
item_max_order = dict(zip(item_rows['item_ref_norm'], item_rows['maximum_order']))
item_benefit = dict(zip(item_rows['item_ref_norm'], item_rows['unit_benefit_cents']))
item_fee = dict(zip(item_rows['item_ref_norm'], item_rows['item_fee_cents']))
item_category = dict(zip(item_rows['item_ref_norm'], item_rows['category_norm']))
item_location = dict(zip(item_rows['item_ref_norm'], item_rows['location_id_norm']))
item_authorized = dict(zip(item_rows['item_ref_norm'], item_rows['authorized']))
df_capacity['resource_norm'] = df_capacity['resource'].apply(norm_str)
df_capacity['amount'] = df_capacity['amount'].astype(int)
resource_capacity = df_capacity.groupby('resource_norm')['amount'].sum().to_dict()
df_usage_10['item_ref_norm'] = df_usage_10['item_ref'].apply(norm_str)
df_usage_10['resource_norm'] = df_usage_10['resource'].apply(norm_str)
df_usage_10['amount'] = df_usage_10['amount'].astype(int)
df_usage_11['item_ref_norm'] = df_usage_11['item_ref'].apply(norm_str)
df_usage_11['resource_norm'] = df_usage_11['resource'].apply(norm_str)
df_usage_11['amount'] = df_usage_11['amount'].astype(int)
df_usage = pd.concat([df_usage_10, df_usage_11], ignore_index=True)
usage_dict = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref_norm']
    r = row['resource_norm']
    amt = row['amount']
    usage_dict[i, r] = amt
m = gp.Model('FC_EAST_HVAC_Placement')
x_vars = m.addVars(I, vtype=gp.GRB.INTEGER, name='')
z_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(C, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(B_idx, vtype=gp.GRB.BINARY, name='')
for i in I:
    if item_authorized[i] == 0:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(x_vars[i] >= 0, name=f'x_nonneg_{i}')
        m.addConstr(x_vars[i] <= item_max_order[i] * z_vars[i], name=f'x_max_{i}')
        m.addConstr(x_vars[i] >= item_min_lot[i] * z_vars[i], name=f'x_min_{i}')
for i in I:
    pass
for c in C:
    items_in_c = [i for i in I if item_category[i] == c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= category_min[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= category_max[c], name=f'cat_max_{c}')
for c in C:
    items_in_c = [i for i in I if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(y_vars[c] >= z_vars[i], name=f'cat_act_{c}_{i}')
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)), name=f'cat_act_sum_{c}')
for r in R:
    items_in_r = [i for i in I if item_location[i] == r]
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x_vars[i] for i in items_in_r)) <= resource_capacity.get(r, 0), name=f'res_cap_{r}')
for (i, j) in incompat_pairs:
    if i in I_set and j in I_set:
        m.addConstr(z_vars[i] + z_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in requires_pairs:
    if i in I_set and j in I_set:
        m.addConstr(z_vars[i] <= z_vars[j], name=f'requires_{i}_{j}')
for (b_idx, (a, b)) in enumerate(B):
    if a in I_set and b in I_set:
        m.addConstr(w_vars[b_idx] <= z_vars[a], name=f'bundle1_{b_idx}')
        m.addConstr(w_vars[b_idx] <= z_vars[b], name=f'bundle2_{b_idx}')
        m.addConstr(w_vars[b_idx] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{b_idx}')
    else:
        m.addConstr(w_vars[b_idx] == 0, name=f'bundle_forbid_{b_idx}')
obj = gp.quicksum((item_benefit[i] * x_vars[i] for i in I))
obj -= gp.quicksum((item_fee[i] * z_vars[i] for i in I))
obj -= gp.quicksum((category_fee[c] * y_vars[c] for c in C))
obj += gp.quicksum((bundle_bonus[B[b_idx]] * w_vars[b_idx] for b_idx in range(len(B))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()