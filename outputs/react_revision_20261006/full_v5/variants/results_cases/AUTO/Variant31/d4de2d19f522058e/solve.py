import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_01.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_04/export_04.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_05/export_05.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_06/export_06.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_01/export_07.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_02/export_08.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant31/inputs/batch_03/export_09.csv'
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
items = df_item['item_ref'].astype(str).unique().tolist()
categories = df_category['category'].astype(str).unique().tolist()
resources = sorted(set(df_capacity_ledger['resource'].astype(str)).union(df_usage['resource'].astype(str)))
bundles = list(df_bundle.index)
incompat_pairs = list(df_incompatible[['item_a', 'item_b']].itertuples(index=False, name=None))
requires_pairs = list(df_requires[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
item_df = df_item.set_index('item_ref')
authorized = item_df['authorized'].astype(int).to_dict()
minimum_lot = item_df['minimum_lot'].astype(int).to_dict()
maximum_order = item_df['maximum_order'].astype(int).to_dict()
unit_benefit_cents = item_df['unit_benefit_cents'].astype(int).to_dict()
item_fee_cents = item_df['item_fee_cents'].astype(int).to_dict()
item_category = item_df['category'].astype(str).to_dict()
cat_df = df_category.set_index('category')
minimum_quantity = cat_df['minimum_quantity'].astype(int).to_dict()
maximum_quantity = cat_df['maximum_quantity'].astype(int).to_dict()
activation_fee_cents = cat_df['activation_fee_cents'].astype(int).to_dict()
bundle_item_a = df_bundle['item_a'].astype(str).tolist()
bundle_item_b = df_bundle['item_b'].astype(str).tolist()
bundle_bonus_cents = df_bundle['bonus_cents'].astype(int).tolist()
bundle_pairs = [(bundle_item_a[k], bundle_item_b[k]) for k in range(len(df_bundle))]
bundle_idx = list(range(len(df_bundle)))
usage_df = df_usage[['item_ref', 'resource', 'amount']]
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage_dict[i, r] = int(row['amount'])
cap_opening = df_capacity_ledger[df_capacity_ledger['entry'].str.casefold() == 'opening']
cap_reservation = df_capacity_ledger[df_capacity_ledger['entry'].str.casefold() == 'reservation']
total_capacity = {}
for r in resources:
    opening = cap_opening[cap_opening['resource'].str.casefold() == r.casefold()]['amount'].sum()
    reservation = cap_reservation[cap_reservation['resource'].str.casefold() == r.casefold()]['amount'].sum()
    total_capacity[r] = int(opening + reservation)
items_in_category = {c: [] for c in categories}
for i in items:
    c = item_category[i]
    items_in_category[c].append(i)
m = gp.Model('RIVERSIDE_AUTO_Vehicle_Selection')
x = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_idx, vtype=gp.GRB.BINARY, name='')
for i in items:
    if authorized[i] == 1:
        m.addConstr(x[i] == 0 + gp.quicksum(0), name='')
        m.addConstr(x[i] >= minimum_lot[i] * y[i], name='minlot_' + i)
        m.addConstr(x[i] <= maximum_order[i] * y[i], name='maxorder_' + i)
    else:
        m.addConstr(x[i] == 0, name='unauth_' + i)
        m.addConstr(y[i] == 0, name='unauth_y_' + i)
for r in resources:
    expr = gp.LinExpr()
    for i in items:
        if (i, r) in usage_dict:
            expr += usage_dict[i, r] * x[i]
    m.addConstr(expr <= total_capacity[r], name='rescap_' + r)
for c in categories:
    expr = gp.quicksum((x[i] for i in items_in_category[c]))
    m.addConstr(expr <= maximum_quantity[c], name='catmax_' + c)
    m.addConstr(expr >= minimum_quantity[c] * z[c], name='catmin_' + c)
    for i in items_in_category[c]:
        m.addConstr(y[i] <= z[c], name='catlink_' + i + '_' + c)
for (i, j) in incompat_pairs:
    if i in items and j in items:
        m.addConstr(y[i] + y[j] <= 1, name='incomp_' + i + '_' + j)
for (i, prereq) in requires_pairs:
    if i in items and prereq in items:
        m.addConstr(y[i] <= y[prereq], name='req_' + i + '_' + prereq)
for (k, (i_a, i_b)) in enumerate(bundle_pairs):
    if i_a in items and i_b in items:
        m.addConstr(b[k] <= y[i_a], name='bundle1_' + str(k))
        m.addConstr(b[k] <= y[i_b], name='bundle2_' + str(k))
        m.addConstr(b[k] >= y[i_a] + y[i_b] - 1, name='bundle3_' + str(k))
    else:
        m.addConstr(b[k] == 0, name='bundle_invalid_' + str(k))
obj_benefit = gp.quicksum((unit_benefit_cents[i] * x[i] for i in items))
obj_item_fee = gp.quicksum((item_fee_cents[i] * y[i] for i in items))
obj_cat_fee = gp.quicksum((activation_fee_cents[c] * z[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus_cents[k] * b[k] for k in bundle_idx))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()