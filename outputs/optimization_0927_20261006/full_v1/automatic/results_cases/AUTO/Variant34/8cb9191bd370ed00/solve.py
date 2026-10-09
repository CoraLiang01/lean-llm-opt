import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]

def norm(s):
    return s.casefold().strip() if isinstance(s, str) else s
item_df1 = dfs[6]
item_df2 = dfs[7]
item_df = pd.concat([item_df1, item_df2], ignore_index=True)
item_df['item_ref'] = item_df['item_ref'].apply(norm)
item_df['category'] = item_df['category'].apply(norm)
item_df['location_id'] = item_df['location_id'].apply(norm)
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
options = list(item_df['item_ref'].unique())
option_params = {}
for (_, row) in item_df.iterrows():
    i = row['item_ref']
    option_params[i] = {'category': row['category'], 'authorized': row['authorized'], 'minimum_lot': row['minimum_lot'], 'maximum_order': row['maximum_order'], 'location_id': row['location_id']}
sections = sorted(item_df['location_id'].unique())
cat_df = dfs[3]
cat_df['category'] = cat_df['category'].apply(norm)
cat_df['minimum_quantity'] = cat_df['minimum_quantity'].astype(int)
cat_df['maximum_quantity'] = cat_df['maximum_quantity'].astype(int)
cat_df['activation_fee_cents'] = cat_df['activation_fee_cents'].astype(int)
categories = list(cat_df['category'].unique())
cat_min = dict(zip(cat_df['category'], cat_df['minimum_quantity']))
cat_max = dict(zip(cat_df['category'], cat_df['maximum_quantity']))
cat_fee = dict(zip(cat_df['category'], cat_df['activation_fee_cents']))
benefit_df = dfs[0]
benefit_df['item_ref'] = benefit_df['item_ref'].apply(norm)
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
benefit_per_option = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in options:
    if i not in benefit_per_option:
        benefit_per_option[i] = 0
item_fee_df = dfs[8]
item_fee_df['item_ref'] = item_fee_df['item_ref'].apply(norm)
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
for i in options:
    if i not in item_fee:
        item_fee[i] = 0
cap_df = dfs[2]
cap_df['resource'] = cap_df['resource'].apply(norm)
cap_df['amount'] = cap_df['amount'].astype(int)
cap_df['unit'] = cap_df['unit'].apply(norm)
resource_caps = cap_df.groupby('resource')['amount'].sum().to_dict()
resources = list(resource_caps.keys())
usage_df1 = dfs[11]
usage_df2 = dfs[12]
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
usage_df['item_ref'] = usage_df['item_ref'].apply(norm)
usage_df['resource'] = usage_df['resource'].apply(norm)
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['unit'] = usage_df['unit'].apply(norm)
usage_df['amount_ml'] = usage_df['amount'] * 1000
usage_per_option_resource = {}
for (_, row) in usage_df.iterrows():
    usage_per_option_resource[row['item_ref'], row['resource']] = row['amount_ml']
for i in options:
    for r in resources:
        if (i, r) not in usage_per_option_resource:
            usage_per_option_resource[i, r] = 0
option_bounds = {}
for i in options:
    p = option_params[i]
    if p['authorized'] == 1:
        option_bounds[i] = (0, p['maximum_order'])
    else:
        option_bounds[i] = (0, 0)
incomp_df = dfs[5]
incomp_df['item_a'] = incomp_df['item_a'].apply(norm)
incomp_df['item_b'] = incomp_df['item_b'].apply(norm)
incompatibles = set()
for (_, row) in incomp_df.iterrows():
    incompatibles.add((row['item_a'], row['item_b']))
    incompatibles.add((row['item_b'], row['item_a']))
req_df = dfs[10]
req_df['item_ref'] = req_df['item_ref'].apply(norm)
req_df['prerequisite_ref'] = req_df['prerequisite_ref'].apply(norm)
requires = set()
for (_, row) in req_df.iterrows():
    requires.add((row['item_ref'], row['prerequisite_ref']))
bundle_df = dfs[1]
bundle_df['item_a'] = bundle_df['item_a'].apply(norm)
bundle_df['item_b'] = bundle_df['item_b'].apply(norm)
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundles = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    bundles.append((a, b))
    bundle_bonus[a, b] = row['bonus_cents']
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=[option_bounds[i][0] for i in options], ub=[option_bounds[i][1] for i in options], name='')
y_vars = m.addVars(options, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in options:
    m.addConstr(x_vars[i] <= option_bounds[i][1] * y_vars[i], name=f'link_x_y_{i}')
    if option_params[i]['authorized'] == 1:
        m.addConstr(x_vars[i] >= option_params[i]['minimum_lot'] * y_vars[i], name=f'minlot_x_y_{i}')
    else:
        m.addConstr(x_vars[i] == 0, name=f'unauth_x_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage_per_option_resource[i, r] * x_vars[i] for i in options)) <= resource_caps[r], name=f'cap_{r}')
for c in categories:
    m.addConstr(gp.quicksum((x_vars[i] for i in options if option_params[i]['category'] == c)) >= cat_min[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x_vars[i] for i in options if option_params[i]['category'] == c)) <= cat_max[c], name=f'cat_max_{c}')
for c in categories:
    for i in options:
        if option_params[i]['category'] == c:
            m.addConstr(z_vars[c] >= y_vars[i], name=f'cat_z_ge_y_{c}_{i}')
    m.addConstr(z_vars[c] <= gp.quicksum((y_vars[i] for i in options if option_params[i]['category'] == c)), name=f'cat_z_le_sumy_{c}')
for (a, b) in incompatibles:
    if a in options and b in options:
        m.addConstr(y_vars[a] + y_vars[b] <= 1, name=f'incomp_{a}_{b}')
for (i, pre) in requires:
    if i in options and pre in options:
        m.addConstr(y_vars[i] <= y_vars[pre], name=f'req_{i}_{pre}')
for (a, b) in bundles:
    if a in options and b in options:
        m.addConstr(b_vars[a, b] <= y_vars[a], name=f'bundle_b_le_ya_{a}_{b}')
        m.addConstr(b_vars[a, b] <= y_vars[b], name=f'bundle_b_le_yb_{a}_{b}')
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'bundle_b_ge_sum_{a}_{b}')
    else:
        m.addConstr(b_vars[a, b] == 0, name=f'bundle_b_zero_{a}_{b}')
benefit_term = gp.quicksum((benefit_per_option[i] * x_vars[i] for i in options))
item_fee_term = gp.quicksum((item_fee[i] * y_vars[i] for i in options))
cat_fee_term = gp.quicksum((cat_fee[c] * z_vars[c] for c in categories))
bundle_term = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundles))
m.setObjective(benefit_term - item_fee_term - cat_fee_term + bundle_term, gp.GRB.MAXIMIZE)
m.optimize()