import gurobipy as gp
import pandas as pd
import numpy as np
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv'
f_usage_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv'
f_usage_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv'
f_item_1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv'
f_item_2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv'
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage_1 = pd.read_csv(f_usage_1, sep=',')
df_usage_2 = pd.read_csv(f_usage_2, sep=',')
df_item_1 = pd.read_csv(f_item_1, sep=',')
df_item_2 = pd.read_csv(f_item_2, sep=',')
items_1 = df_item_1[df_item_1['authorized'] == 1].copy()
items_2 = df_item_2[df_item_2['authorized'] == 1].copy()
items = pd.concat([items_1, items_2], ignore_index=True)
items = items.drop_duplicates(subset=['item_ref'])
I = list(items['item_ref'])
item_category = dict(zip(items['item_ref'], items['category']))
item_location = dict(zip(items['item_ref'], items['location_id']))
item_minlot = dict(zip(items['item_ref'], items['minimum_lot']))
item_maxorder = dict(zip(items['item_ref'], items['maximum_order']))
item_unit_benefit = dict(zip(items['item_ref'], items['unit_benefit_cents']))
item_fee = dict(zip(items['item_ref'], items['item_fee_cents']))
C = list(df_category['category'])
cat_minqty = dict(zip(df_category['category'], df_category['minimum_quantity']))
cat_maxqty = dict(zip(df_category['category'], df_category['maximum_quantity']))
cat_fee = dict(zip(df_category['category'], df_category['activation_fee_cents']))
areas_from_capacity = df_capacity_ledger['resource'].unique()
areas_from_usage = pd.concat([df_usage_1['resource'], df_usage_2['resource']]).unique()
R = sorted(set(areas_from_capacity).union(set(areas_from_usage)))
df_bundle = df_bundle[df_bundle['table'].str.casefold() == 'bundle']
B = list(df_bundle.index)
bundle_item_a = dict(zip(df_bundle.index, df_bundle['item_a']))
bundle_item_b = dict(zip(df_bundle.index, df_bundle['item_b']))
bundle_bonus = dict(zip(df_bundle.index, df_bundle['bonus_cents']))
df_incompatible = df_incompatible[df_incompatible['table'].str.casefold() == 'incompatible']
Inc = [(row['item_a'], row['item_b']) for (_, row) in df_incompatible.iterrows()]
df_requires = df_requires[df_requires['table'].str.casefold() == 'requires']
Req = [(row['item_ref'], row['prerequisite_ref']) for (_, row) in df_requires.iterrows()]
usage_df = pd.concat([df_usage_1, df_usage_2], ignore_index=True)
usage_df = usage_df[usage_df['table'].str.casefold() == 'usage']
usage_dict = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    if (i, r) in usage_dict:
        usage_dict[i, r] += amt
    else:
        usage_dict[i, r] = amt
df_capacity_ledger = df_capacity_ledger[df_capacity_ledger['table'].str.casefold() == 'capacity_ledger']
capacity = {}
for r in R:
    df_r = df_capacity_ledger[df_capacity_ledger['resource'] == r]
    cap = df_r['amount'].sum()
    capacity[r] = cap

def solve_problem():
    m = gp.Model('FC_EAST_HVAC_Placement')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(C, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(B, vtype=gp.GRB.BINARY, name='')
    for i in I:
        minlot = int(item_minlot[i])
        maxorder = int(item_maxorder[i])
        m.addConstr(x[i] >= minlot * y[i], name=f'item_minlot_{i}')
        m.addConstr(x[i] <= maxorder * y[i], name=f'item_maxorder_{i}')
    for r in R:
        items_in_r = [i for i in I if item_location[i] == r]
        expr = gp.LinExpr()
        for i in items_in_r:
            amt = usage_dict.get((i, r), 0)
            expr += amt * x[i]
        m.addConstr(expr <= capacity[r], name=f'cap_{r}')
    for c in C:
        items_in_c = [i for i in I if item_category[i] == c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= int(cat_minqty[c]), name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= int(cat_maxqty[c]), name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(y[i] <= z[c], name=f'cat_act_link_{i}_{c}')
        m.addConstr(z[c] <= gp.quicksum((y[i] for i in items_in_c)), name=f'cat_act_sum_{c}')
    for (i, j) in Inc:
        if i in I and j in I:
            m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
    for (i, prereq) in Req:
        if i in I and prereq in I:
            m.addConstr(y[i] <= y[prereq], name=f'requires_{i}_{prereq}')
    for bidx in B:
        i = bundle_item_a[bidx]
        j = bundle_item_b[bidx]
        if i in I and j in I:
            m.addConstr(b[bidx] <= y[i], name=f'bundle1_{bidx}')
            m.addConstr(b[bidx] <= y[j], name=f'bundle2_{bidx}')
            m.addConstr(b[bidx] >= y[i] + y[j] - 1, name=f'bundle3_{bidx}')
        else:
            m.addConstr(b[bidx] == 0, name=f'bundle_forced0_{bidx}')
    obj = gp.quicksum((item_unit_benefit[i] * x[i] for i in I))
    obj -= gp.quicksum((item_fee[i] * y[i] for i in I))
    obj -= gp.quicksum((cat_fee[c] * z[c] for c in C))
    obj += gp.quicksum((bundle_bonus[bidx] * b[bidx] for bidx in B))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')