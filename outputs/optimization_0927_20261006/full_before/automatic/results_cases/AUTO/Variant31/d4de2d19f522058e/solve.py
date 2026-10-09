import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv', sep=',')
df_cap = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv', sep=',')
df_cat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv', sep=',')
df_id = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv', sep=',')
df_incomp = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv', sep=',')
df_item = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv', sep=',')
df_req = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv', sep=',')
df_usage = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv', sep=',')
item_ids = df_item['item_ref'].astype(str).tolist()
cat_ids = df_cat['category'].astype(str).tolist()
resource_ids = sorted(set(df_cap['resource'].astype(str)).union(df_usage['resource'].astype(str)))
bundle_pairs = [(str(row['item_a']), str(row['item_b'])) for (_, row) in df_bundle.iterrows()]
incomp_pairs = [(str(row['item_a']), str(row['item_b'])) for (_, row) in df_incomp.iterrows()]
req_pairs = [(str(row['item_ref']), str(row['prerequisite_ref'])) for (_, row) in df_req.iterrows()]
item_authorized = df_item.set_index('item_ref')['authorized'].astype(int).to_dict()
item_minlot = df_item.set_index('item_ref')['minimum_lot'].astype(int).to_dict()
item_maxorder = df_item.set_index('item_ref')['maximum_order'].astype(int).to_dict()
item_cat = df_item.set_index('item_ref')['category'].astype(str).to_dict()
item_unit_benefit = df_item.set_index('item_ref')['unit_benefit_cents'].astype(int).to_dict()
item_fee = df_item.set_index('item_ref')['item_fee_cents'].astype(int).to_dict()
cat_minqty = df_cat.set_index('category')['minimum_quantity'].astype(int).to_dict()
cat_maxqty = df_cat.set_index('category')['maximum_quantity'].astype(int).to_dict()
cat_fee = df_cat.set_index('category')['activation_fee_cents'].astype(int).to_dict()
bundle_bonus = {(str(row['item_a']), str(row['item_b'])): int(row['bonus_cents']) for (_, row) in df_bundle.iterrows()}
usage_dict = {}
for (_, row) in df_usage.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage_dict[i, r] = int(row['amount'])
cap_opening = df_cap[df_cap['entry'].str.casefold() == 'opening']
cap_resv = df_cap[df_cap['entry'].str.casefold() == 'reservation']
cap_opening_sum = cap_opening.groupby('resource')['amount'].sum()
cap_resv_sum = cap_resv.groupby('resource')['amount'].sum()
total_capacity = {}
for r in resource_ids:
    opening = cap_opening_sum[r] if r in cap_opening_sum else 0
    resv = cap_resv_sum[r] if r in cap_resv_sum else 0
    total_capacity[r] = int(opening + resv)
cat_items = {c: [] for c in cat_ids}
for i in item_ids:
    c = item_cat[i]
    if c in cat_items:
        cat_items[c].append(i)
item_resources = {i: [] for i in item_ids}
for ((i, r), amt) in usage_dict.items():
    if i in item_resources:
        item_resources[i].append(r)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
x = m.addVars(item_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
z = m.addVars(cat_ids, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_ids:
    auth = item_authorized[i]
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    if auth == 0:
        m.addConstr(x[i] == 0, name=f'unauth_{i}')
        m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x[i] <= maxorder * y[i], name=f'link_x_y_ub_{i}')
        m.addConstr(x[i] >= minlot * y[i], name=f'link_x_y_lb_{i}')
        m.addConstr(x[i] <= maxorder, name=f'maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
for c in cat_ids:
    items_in_c = cat_items[c]
    if items_in_c:
        for i in items_in_c:
            m.addConstr(y[i] <= z[c], name=f'cat_link_{i}_{c}')
        m.addConstr(gp.quicksum((y[i] for i in items_in_c)) >= z[c], name=f'cat_sum_{c}')
for c in cat_ids:
    items_in_c = cat_items[c]
    minq = cat_minqty[c]
    maxq = cat_maxqty[c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= minq, name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= maxq, name=f'cat_max_{c}')
for r in resource_ids:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x[i] for i in item_ids)) <= total_capacity[r], name=f'res_cap_{r}')
for (i, j) in incomp_pairs:
    if i in item_ids and j in item_ids:
        m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
for (i, prereq) in req_pairs:
    if i in item_ids and prereq in item_ids:
        m.addConstr(y[i] <= y[prereq], name=f'req_{i}_{prereq}')
for (i_a, i_b) in bundle_pairs:
    if item_authorized.get(i_a, 0) == 0 or item_authorized.get(i_b, 0) == 0:
        m.addConstr(b[i_a, i_b] == 0, name=f'bundle_unauth_{i_a}_{i_b}')
    else:
        m.addConstr(b[i_a, i_b] <= y[i_a], name=f'bundle_le_ya_{i_a}_{i_b}')
        m.addConstr(b[i_a, i_b] <= y[i_b], name=f'bundle_le_yb_{i_a}_{i_b}')
        m.addConstr(b[i_a, i_b] >= y[i_a] + y[i_b] - 1, name=f'bundle_ge_sum_{i_a}_{i_b}')
obj_benefit = gp.quicksum((item_unit_benefit[i] * x[i] for i in item_ids))
obj_item_fee = gp.quicksum((item_fee[i] * y[i] for i in item_ids))
obj_cat_fee = gp.quicksum((cat_fee[c] * z[c] for c in cat_ids))
obj_bundle = gp.quicksum((bundle_bonus[i_a, i_b] * b[i_a, i_b] for (i_a, i_b) in bundle_pairs))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')