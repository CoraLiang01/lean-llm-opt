import gurobipy as gp
import pandas as pd
import numpy as np
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
benefit_df = dfs[0]
bundle_df = dfs[1]
capacity_ledger_df = dfs[2]
category_df = dfs[3]
incompatible_df = dfs[5]
item_df1 = dfs[6]
item_df2 = dfs[7]
item_fee_df = dfs[8]
requires_df = dfs[10]
usage_df1 = dfs[11]
usage_df2 = dfs[12]
item_tables = [item_df1, item_df2]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows['item_ref'] = item_rows['item_ref'].str.strip()
item_rows['category'] = item_rows['category'].str.strip()
item_rows['location_id'] = item_rows['location_id'].str.strip()
item_rows['authorized'] = item_rows['authorized'].astype(int)
item_rows['minimum_lot'] = item_rows['minimum_lot'].astype(int)
item_rows['maximum_order'] = item_rows['maximum_order'].astype(int)
item_rows = item_rows.drop_duplicates(subset=['item_ref'])
item_refs = list(item_rows['item_ref'])
categories = sorted(category_df['category'].str.strip().unique())
resources = sorted(capacity_ledger_df['resource'].str.strip().unique())
authorized = dict(zip(item_rows['item_ref'], item_rows['authorized']))
minimum_lot = dict(zip(item_rows['item_ref'], item_rows['minimum_lot']))
maximum_order = dict(zip(item_rows['item_ref'], item_rows['maximum_order']))
item_category = dict(zip(item_rows['item_ref'], item_rows['category']))
item_location = dict(zip(item_rows['item_ref'], item_rows['location_id']))
benefit_df = benefit_df[benefit_df['table'].str.strip().str.casefold() == 'benefit']
benefit_df['item_ref'] = benefit_df['item_ref'].str.strip()
benefit_df['amount_cents'] = benefit_df['amount_cents'].astype(int)
item_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in item_refs:
    if i not in item_benefit:
        item_benefit[i] = 0
item_fee_df = item_fee_df[item_fee_df['table'].str.strip().str.casefold() == 'item_fee']
item_fee_df['item_ref'] = item_fee_df['item_ref'].str.strip()
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_activation_fee = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
for i in item_refs:
    if i not in item_activation_fee:
        item_activation_fee[i] = 0
category_df = category_df[category_df['table'].str.strip().str.casefold() == 'category']
category_df['category'] = category_df['category'].str.strip()
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
category_min = dict(zip(category_df['category'], category_df['minimum_quantity']))
category_max = dict(zip(category_df['category'], category_df['maximum_quantity']))
category_activation_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
bundle_df = bundle_df[bundle_df['table'].str.strip().str.casefold() == 'bundle']
bundle_df['item_a'] = bundle_df['item_a'].str.strip()
bundle_df['item_b'] = bundle_df['item_b'].str.strip()
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_pairs = list(zip(bundle_df['item_a'], bundle_df['item_b']))
bundle_bonus = dict((((row['item_a'], row['item_b']), row['bonus_cents']) for (_, row) in bundle_df.iterrows()))
incompatible_df = incompatible_df[incompatible_df['table'].str.strip().str.casefold() == 'incompatible']
incompatible_df['item_a'] = incompatible_df['item_a'].str.strip()
incompatible_df['item_b'] = incompatible_df['item_b'].str.strip()
incompatible_pairs = list(zip(incompatible_df['item_a'], incompatible_df['item_b']))
requires_df = requires_df[requires_df['table'].str.strip().str.casefold() == 'requires']
requires_df['item_ref'] = requires_df['item_ref'].str.strip()
requires_df['prerequisite_ref'] = requires_df['prerequisite_ref'].str.strip()
prerequisite_pairs = list(zip(requires_df['item_ref'], requires_df['prerequisite_ref']))
usage_df = pd.concat([usage_df1, usage_df2], ignore_index=True)
usage_df = usage_df[usage_df['table'].str.strip().str.casefold() == 'usage']
usage_df['item_ref'] = usage_df['item_ref'].str.strip()
usage_df['resource'] = usage_df['resource'].str.strip()
usage_df['amount'] = usage_df['amount'].astype(int)
usage_df['unit'] = usage_df['unit'].str.strip().str.casefold()

def to_ml(row):
    if row['unit'] == 'ml':
        return row['amount']
    elif row['unit'] == 'liter':
        return row['amount'] * 1000
    else:
        raise ValueError(f"Unknown unit: {row['unit']}")
usage_df['amount_ml'] = usage_df.apply(to_ml, axis=1)
usage_dict = {}
for (_, row) in usage_df.iterrows():
    usage_dict[row['item_ref'], row['resource']] = row['amount_ml']
capacity_ledger_df = capacity_ledger_df[capacity_ledger_df['table'].str.strip().str.casefold() == 'capacity_ledger']
capacity_ledger_df['resource'] = capacity_ledger_df['resource'].str.strip()
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
capacity_ledger_df['unit'] = capacity_ledger_df['unit'].str.strip().str.casefold()

def cap_to_ml(row):
    if row['unit'] == 'ml':
        return row['amount']
    elif row['unit'] == 'liter':
        return row['amount'] * 1000
    else:
        raise ValueError(f"Unknown unit: {row['unit']}")
capacity_ledger_df['amount_ml'] = capacity_ledger_df.apply(cap_to_ml, axis=1)
resource_capacity = capacity_ledger_df.groupby('resource')['amount_ml'].sum().to_dict()
for r in resources:
    if r not in resource_capacity:
        resource_capacity[r] = 0
category_items = {g: [] for g in categories}
for i in item_refs:
    g = item_category[i]
    if g in category_items:
        category_items[g].append(i)
    else:
        category_items[g] = [i]
resource_items = {r: [] for r in resources}
for i in item_refs:
    loc = item_location[i]
    if loc in resource_items:
        resource_items[loc].append(i)
    else:
        resource_items[loc] = [i]
m = gp.Model('market_square_merchandising')
x_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    if authorized[i] == 0:
        m.addConstr(x_vars[i] == 0)
        m.addConstr(y_vars[i] == 0)
    else:
        m.addConstr(x_vars[i] >= minimum_lot[i] * y_vars[i])
        m.addConstr(x_vars[i] <= maximum_order[i] * y_vars[i])
for g in categories:
    for i in category_items[g]:
        m.addConstr(y_vars[i] <= z_vars[g])
    m.addConstr(z_vars[g] <= gp.quicksum((y_vars[i] for i in category_items[g])))
for g in categories:
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) >= category_min[g])
    m.addConstr(gp.quicksum((x_vars[i] for i in category_items[g])) <= category_max[g])
for r in resources:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x_vars[i] for i in resource_items[r])) <= resource_capacity[r])
for (a, b) in incompatible_pairs:
    if a in y_vars and b in y_vars:
        m.addConstr(y_vars[a] + y_vars[b] <= 1)
for (i, pre) in prerequisite_pairs:
    if i in y_vars and pre in y_vars:
        m.addConstr(y_vars[i] <= y_vars[pre])
for (a, b) in bundle_pairs:
    if a in y_vars and b in y_vars:
        m.addConstr(b_vars[a, b] <= y_vars[a])
        m.addConstr(b_vars[a, b] <= y_vars[b])
        m.addConstr(b_vars[a, b] >= y_vars[a] + y_vars[b] - 1)
    else:
        m.addConstr(b_vars[a, b] == 0)
item_obj = gp.quicksum((item_benefit[i] * x_vars[i] - item_activation_fee[i] * y_vars[i] for i in item_refs))
cat_obj = gp.quicksum((category_activation_fee[g] * z_vars[g] for g in categories))
bundle_obj = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for (a, b) in bundle_pairs))
m.setObjective(item_obj - cat_obj + bundle_obj, gp.GRB.MAXIMIZE)
m.optimize()