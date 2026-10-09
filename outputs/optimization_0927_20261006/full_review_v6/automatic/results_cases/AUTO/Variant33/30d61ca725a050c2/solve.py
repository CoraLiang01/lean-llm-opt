import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
item_tables = []
for idx in [6, 7]:
    df = dfs[idx].copy()
    df['authorized'] = df['authorized'].astype(int)
    df['minimum_lot'] = df['minimum_lot'].astype(int)
    df['maximum_order'] = df['maximum_order'].astype(int)
    item_tables.append(df[['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order']])
items_df = pd.concat(item_tables, ignore_index=True)
item_refs = items_df['item_ref'].unique().tolist()
cat_df = dfs[3].copy()
cat_df['minimum_quantity'] = cat_df['minimum_quantity'].astype(int)
cat_df['maximum_quantity'] = cat_df['maximum_quantity'].astype(int)
cat_df['activation_fee_cents'] = cat_df['activation_fee_cents'].astype(int)
categories = cat_df['category'].unique().tolist()
benefit_df = dfs[0].copy()
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
benefit_sum = benefit_df.groupby('item_ref')['amount_cents'].sum()
benefit_dict = benefit_sum.to_dict()
item_fee_df = dfs[8].copy()
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee_dict = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
cat_fee_dict = cat_df.set_index('category')['activation_fee_cents'].to_dict()
usage_df1 = dfs[11].copy()
usage_df2 = dfs[12].copy()
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_dict = {}
for (_, row) in usage_df.iterrows():
    usage_dict[row['item_ref'], row['resource']] = row['amount']
resources = usage_df['resource'].unique().tolist()
cap_df = dfs[2].copy()
cap_df['amount'] = cap_df['amount'].astype(int)
cap_sum = cap_df.groupby('resource')['amount'].sum()
cap_dict = cap_sum.to_dict()
bundle_df = dfs[1].copy()
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = row['bonus_cents']
incomp_df = dfs[5].copy()
incomp_pairs = []
for (_, row) in incomp_df.iterrows():
    incomp_pairs.append((row['item_a'], row['item_b']))
req_df = dfs[10].copy()
requires_pairs = []
for (_, row) in req_df.iterrows():
    requires_pairs.append((row['item_ref'], row['prerequisite_ref']))
item_param = {}
for (_, row) in items_df.iterrows():
    item_param[row['item_ref']] = {'category': row['category'], 'authorized': row['authorized'], 'minimum_lot': row['minimum_lot'], 'maximum_order': row['maximum_order']}
cat_to_items = {c: [] for c in categories}
for i in item_refs:
    c = item_param[i]['category']
    cat_to_items[c].append(i)
m = gp.Model('NY_Dev_Module_Portfolio')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    auth = item_param[i]['authorized']
    min_lot = item_param[i]['minimum_lot']
    max_ord = item_param[i]['maximum_order']
    if auth == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= min_lot * y_vars[i])
        m.addConstr(x_vars[i] <= max_ord * y_vars[i])
        m.addConstr(x_vars[i] >= 0)
        m.addConstr(y_vars[i] <= 1)
        m.addConstr(x_vars[i] <= max_ord)
for r in resources:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x_vars[i] for i in item_refs)) <= cap_dict.get(r, 0))
for c in categories:
    items_in_c = cat_to_items[c]
    min_q = cat_df.loc[cat_df['category'] == c, 'minimum_quantity'].iloc[0]
    max_q = cat_df.loc[cat_df['category'] == c, 'maximum_quantity'].iloc[0]
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) >= min_q)
    m.addConstr(gp.quicksum((x_vars[i] for i in items_in_c)) <= max_q)
    for i in items_in_c:
        m.addConstr(y_vars[i] <= z_vars[c])
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in items_in_c)))
for (a, b) in incomp_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(y_vars[a] + y_vars[b] <= 1)
for (i, prereq) in requires_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(y_vars[i] <= y_vars[prereq])
for (a, b) in bundle_pairs:
    if a in item_refs and b in item_refs:
        m.addConstr(b_vars[a, b] <= y_vars[a])
        m.addConstr(b_vars[a, b] <= y_vars[b])
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
obj_expr = gp.LinExpr()
for i in item_refs:
    obj_expr += benefit_dict.get(i, 0) * x_vars[i]
for i in item_refs:
    obj_expr -= item_fee_dict.get(i, 0) * y_vars[i]
for c in categories:
    obj_expr -= cat_fee_dict.get(c, 0) * z_vars[c]
for (a, b) in bundle_pairs:
    obj_expr += bundle_bonus.get((a, b), 0) * b_vars[a, b]
m.setObjective(obj_expr, gp.GRB.MAXIMIZE)
m.optimize()