import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
item_df1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv', dtype=str, keep_default_na=False)
item_df2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv', dtype=str, keep_default_na=False)
category_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv', dtype=str, keep_default_na=False)
capacity_ledger_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv', dtype=str, keep_default_na=False)
usage_df1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv', dtype=str, keep_default_na=False)
usage_df2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv', dtype=str, keep_default_na=False)
bundle_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv', dtype=str, keep_default_na=False)
incompat_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv', dtype=str, keep_default_na=False)
requires_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv', dtype=str, keep_default_na=False)
item_tables = [item_df1, item_df2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows['unit_benefit_cents'] = item_rows['unit_benefit_cents'].astype(int)
item_rows['item_fee_cents'] = item_rows['item_fee_cents'].astype(int)
item_rows['item_ref'] = item_rows['item_ref'].astype(str)
item_rows['category'] = item_rows['category'].astype(str)
item_rows['location_id'] = item_rows['location_id'].astype(str)
I = list(item_rows['item_ref'].unique())
item_row_map = {row['item_ref']: row for (_, row) in item_rows.iterrows()}
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_df['category'] = category_df['category'].astype(str)
C = list(category_df['category'].unique())
cat_row_map = {row['category']: row for (_, row) in category_df.iterrows()}
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
capacity_ledger_df['resource'] = capacity_ledger_df['resource'].astype(str)
R = sorted(capacity_ledger_df['resource'].unique())
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_df['item_a'] = bundle_df['item_a'].astype(str)
bundle_df['item_b'] = bundle_df['item_b'].astype(str)
B = list(bundle_df.index)
incompat_pairs = set()
if not incompat_df.empty:
    incompat_df['item_a'] = incompat_df['item_a'].astype(str)
    incompat_df['item_b'] = incompat_df['item_b'].astype(str)
    for (_, row) in incompat_df.iterrows():
        incompat_pairs.add((row['item_a'], row['item_b']))
        incompat_pairs.add((row['item_b'], row['item_a']))
requires_pairs = []
if not requires_df.empty:
    requires_df['item_ref'] = requires_df['item_ref'].astype(str)
    requires_df['prerequisite_ref'] = requires_df['prerequisite_ref'].astype(str)
    for (_, row) in requires_df.iterrows():
        requires_pairs.append((row['item_ref'], row['prerequisite_ref']))
usage_df1['amount'] = usage_df1['amount'].astype(int)
usage_df1['item_ref'] = usage_df1['item_ref'].astype(str)
usage_df1['resource'] = usage_df1['resource'].astype(str)
usage_df2['amount'] = usage_df2['amount'].astype(int)
usage_df2['item_ref'] = usage_df2['item_ref'].astype(str)
usage_df2['resource'] = usage_df2['resource'].astype(str)
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
usage_map = {}
for (_, row) in usage_df.iterrows():
    key = (row['item_ref'], row['resource'])
    usage_map[key] = row['amount']
cap_ledger = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
item_to_category = {row['item_ref']: row['category'] for (_, row) in item_rows.iterrows()}
item_to_location = {row['item_ref']: row['location_id'] for (_, row) in item_rows.iterrows()}
cat_to_items = {c: [] for c in C}
for i in I:
    c = item_to_category[i]
    if c in cat_to_items:
        cat_to_items[c].append(i)
resource_to_items = {r: [] for r in R}
for i in I:
    loc = item_to_location[i]
    if loc in resource_to_items:
        resource_to_items[loc].append(i)
authorized = {i: int(item_row_map[i]['authorized']) for i in I}
min_lot = {i: int(item_row_map[i]['minimum_lot']) for i in I}
max_order = {i: int(item_row_map[i]['maximum_order']) for i in I}
unit_benefit = {i: int(item_row_map[i]['unit_benefit_cents']) for i in I}
item_fee = {i: int(item_row_map[i]['item_fee_cents']) for i in I}
cat_min_qty = {c: int(cat_row_map[c]['minimum_quantity']) for c in C}
cat_max_qty = {c: int(cat_row_map[c]['maximum_quantity']) for c in C}
cat_activation_fee = {c: int(cat_row_map[c]['activation_fee_cents']) for c in C}
bundle_item_a = {b: bundle_df.loc[b, 'item_a'] for b in B}
bundle_item_b = {b: bundle_df.loc[b, 'item_b'] for b in B}
bundle_bonus = {b: int(bundle_df.loc[b, 'bonus_cents']) for b in B}
m = gp.Model('FC_EAST_HVAC_AC_Placement')
x_vars = m.addVars(I, vtype=gp.GRB.INTEGER, name='')
y_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(C, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(B, vtype=gp.GRB.BINARY, name='')
for i in I:
    if authorized[i] == 0:
        m.addConstr(x_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(y_vars[i] == 0, name=f'unauth_y_{i}')
    else:
        m.addConstr(x_vars[i] >= min_lot[i] * y_vars[i], name=f'minlot_{i}')
        m.addConstr(x_vars[i] <= max_order[i] * y_vars[i], name=f'maxorder_{i}')
        m.addConstr(x_vars[i] <= max_order[i] * y_vars[i], name=f'link_y_{i}')
for r in R:
    items_in_r = resource_to_items[r]
    m.addConstr(gp.quicksum((usage_map.get((i, r), 0) * x_vars[i] for i in items_in_r)) <= cap_ledger[r], name=f'cap_{r}')
for c in C:
    items_in_c = cat_to_items[c]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= cat_min_qty[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= cat_max_qty[c], name=f'cat_max_{c}')
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c], name=f'cat_act1_{c}_{i}')
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)), name=f'cat_act2_{c}')
for (i, j) in incompat_pairs:
    if i in I and j in I:
        m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incompat_{i}_{j}')
for (i, j) in requires_pairs:
    if i in I and j in I:
        m.addConstr(y_vars[i] <= y_vars[j], name=f'requires_{i}_{j}')
for b in B:
    i_a = bundle_item_a[b]
    i_b = bundle_item_b[b]
    if i_a in I and i_b in I:
        m.addConstr(w_vars[b] <= y_vars[i_a], name=f'bundle1_{b}')
        m.addConstr(w_vars[b] <= y_vars[i_b], name=f'bundle2_{b}')
        m.addConstr(w_vars[b] >= y_vars[i_a] + y_vars[i_b] - 1, name=f'bundle3_{b}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle_forbid_{b}')
obj = gp.quicksum((unit_benefit[i] * x_vars[i] for i in I)) - gp.quicksum((item_fee[i] * y_vars[i] for i in I)) - gp.quicksum((cat_activation_fee[c] * z_vars[c] for c in C)) + gp.quicksum((bundle_bonus[b] * w_vars[b] for b in B))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()