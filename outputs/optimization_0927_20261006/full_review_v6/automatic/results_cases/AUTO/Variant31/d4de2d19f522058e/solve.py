import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
bundle_pairs = list(df_bundle[['item_a', 'item_b']].itertuples(index=False, name=None))
bundle_bonus = {(row['item_a'], row['item_b']): row['bonus_cents'] for (_, row) in df_bundle.iterrows()}
df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
df_cap['amount'] = df_cap['amount'].astype(int)
cap_sum = df_cap.groupby('resource', as_index=True)['amount'].sum()
resource_list = list(cap_sum.index)
resource_capacity = cap_sum.to_dict()
df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
df_cat['minimum_quantity'] = df_cat['minimum_quantity'].astype(int)
df_cat['maximum_quantity'] = df_cat['maximum_quantity'].astype(int)
df_cat['activation_fee_cents'] = df_cat['activation_fee_cents'].astype(int)
category_list = df_cat['category'].tolist()
category_min = df_cat.set_index('category')['minimum_quantity'].to_dict()
category_max = df_cat.set_index('category')['maximum_quantity'].to_dict()
category_fee = df_cat.set_index('category')['activation_fee_cents'].to_dict()
df_id = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', dtype=str, keep_default_na=False)
item_display_name = df_id.set_index('ref')['display_name'].to_dict()
df_incomp = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
incomp_pairs = list(df_incomp[['item_a', 'item_b']].itertuples(index=False, name=None))
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
df_item['authorized'] = df_item['authorized'].astype(int)
df_item['minimum_lot'] = df_item['minimum_lot'].astype(int)
df_item['maximum_order'] = df_item['maximum_order'].astype(int)
df_item['unit_benefit_cents'] = df_item['unit_benefit_cents'].astype(int)
df_item['item_fee_cents'] = df_item['item_fee_cents'].astype(int)
item_list = df_item['item_ref'].tolist()
item_category = df_item.set_index('item_ref')['category'].to_dict()
item_authorized = df_item.set_index('item_ref')['authorized'].to_dict()
item_minlot = df_item.set_index('item_ref')['minimum_lot'].to_dict()
item_maxorder = df_item.set_index('item_ref')['maximum_order'].to_dict()
item_unit_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].to_dict()
item_fee = df_item.set_index('item_ref')['item_fee_cents'].to_dict()
df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', dtype=str, keep_default_na=False)
requires_pairs = list(df_req[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
df_usage['amount'] = df_usage['amount'].astype(int)
usage_dict = {}
for (_, row) in df_usage.iterrows():
    usage_dict[row['item_ref'], row['resource']] = row['amount']
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
q_vars = m.addVars(item_list, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_list, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_list, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_list:
    auth = item_authorized[i]
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    if auth == 1:
        m.addConstr(q_vars[i] >= minlot * z_vars[i])
        m.addConstr(q_vars[i] <= maxorder * z_vars[i])
        m.addConstr(q_vars[i] >= 0)
        m.addConstr(z_vars[i] >= 0)
        m.addConstr(z_vars[i] <= 1)
    else:
        m.addConstr(q_vars[i] == 0)
        m.addConstr(z_vars[i] == 0)
for r in resource_list:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * q_vars[i] for i in item_list)) <= resource_capacity[r])
for c in category_list:
    items_in_c = [i for i in item_list if item_category[i] == c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= category_min[c])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= category_max[c])
for c in category_list:
    items_in_c = [i for i in item_list if item_category[i] == c]
    for i in items_in_c:
        m.addConstr(z_vars[i] <= y_vars[c])
    m.addConstr(y_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)))
for (i_a, i_b) in bundle_pairs:
    m.addConstr(b_vars[i_a, i_b] <= z_vars[i_a])
    m.addConstr(b_vars[i_a, i_b] <= z_vars[i_b])
    m.addConstr(b_vars[i_a, i_b] >= z_vars[i_a] + z_vars[i_b] - 1)
for (i, j) in incomp_pairs:
    m.addConstr(z_vars[i] + z_vars[j] <= 1)
for (i, j) in requires_pairs:
    m.addConstr(z_vars[i] <= z_vars[j])
item_benefit = gp.quicksum((item_unit_benefit[i] * q_vars[i] - item_fee[i] * z_vars[i] for i in item_list))
category_cost = gp.quicksum((category_fee[c] * y_vars[c] for c in category_list))
bundle_benefit = gp.quicksum((bundle_bonus[i_a, i_b] * b_vars[i_a, i_b] for (i_a, i_b) in bundle_pairs))
m.setObjective(item_benefit - category_cost + bundle_benefit, gp.GRB.MAXIMIZE)
m.optimize()