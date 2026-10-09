import gurobipy as gp
import pandas as pd
import numpy as np
df_benefit = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_01.csv', sep=',')
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_02.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_03.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_04.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_05.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_06.csv', sep=',')
df_item1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_07.csv', sep=',')
df_item2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_02/export_08.csv', sep=',')
df_itemfee = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_03/export_09.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_04/export_10.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_05/export_11.csv', sep=',')
df_usage1 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_06/export_12.csv', sep=',')
df_usage2 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant33/inputs/batch_01/export_13.csv', sep=',')
item_rows = pd.concat([df_item1, df_item2], ignore_index=True)
item_refs = sorted(item_rows['item_ref'].unique())
items = item_refs
categories = sorted(df_category['category'].unique())
resources = sorted(df_capacity['resource'].unique())
bundles = list(df_bundle[['item_a', 'item_b']].itertuples(index=False, name=None))
incompat_pairs = list(df_incompat[['item_a', 'item_b']].itertuples(index=False, name=None))
prereq_pairs = list(df_requires[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
benefit = df_benefit.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in benefit:
        benefit[i] = 0
item_fee = df_itemfee.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in items:
    if i not in item_fee:
        item_fee[i] = 0
cat_fee = df_category.set_index('category')['activation_fee_cents'].to_dict()
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
usage = {}
for (_, row) in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    usage.setdefault((i, r), 0)
    usage[i, r] += amt
df_cap = df_capacity.groupby('resource')['amount'].sum()
capacity = df_cap.to_dict()
item_info = pd.concat([df_item1, df_item2], ignore_index=True)
item_auth = {}
item_minlot = {}
item_maxorder = {}
item_cat = {}
for (_, row) in item_info.iterrows():
    i = row['item_ref']
    item_auth[i] = int(row['authorized'])
    item_minlot[i] = int(row['minimum_lot'])
    item_maxorder[i] = int(row['maximum_order'])
    item_cat[i] = row['category']
cat_minqty = df_category.set_index('category')['minimum_quantity'].to_dict()
cat_maxqty = df_category.set_index('category')['maximum_quantity'].to_dict()
bundle_bonus = {}
for (_, row) in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_bonus[a, b] = int(row['bonus_cents'])
m = gp.Model('NY_Dev_Module_Portfolio')
x = m.addVars(items, vtype=gp.GRB.INTEGER, name='')
y = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    if item_auth[i] == 0:
        m.addConstr(x[i] == 0, name=f'auth_{i}')
        m.addConstr(y[i] == 0, name=f'authy_{i}')
    else:
        m.addConstr(x[i] >= item_minlot[i] * y[i], name=f'minlot_{i}')
        m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'xnonneg_{i}')
for c in categories:
    items_in_c = [i for i in items if item_cat[i] == c]
    if items_in_c:
        for i in items_in_c:
            m.addConstr(x[i] <= item_maxorder[i] * z[c], name=f'catlink_{i}_{c}')
        m.addConstr(gp.quicksum((y[i] for i in items_in_c)) >= z[c], name=f'catminy_{c}')
    else:
        m.addConstr(z[c] == 0, name=f'catempty_{c}')
for c in categories:
    items_in_c = [i for i in items if item_cat[i] == c]
    if items_in_c:
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= cat_minqty[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= cat_maxqty[c], name=f'catmax_{c}')
    else:
        m.addConstr(0 >= cat_minqty[c], name=f'catmin0_{c}')
        m.addConstr(0 <= cat_maxqty[c], name=f'catmax0_{c}')
for r in resources:
    m.addConstr(gp.quicksum((usage.get((i, r), 0) * x[i] for i in items)) <= capacity[r], name=f'res_{r}')
for (i, j) in incompat_pairs:
    if i in items and j in items:
        m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
for (i, pre) in prereq_pairs:
    if i in items and pre in items:
        m.addConstr(y[i] <= y[pre], name=f'prereq_{i}_{pre}')
for (a, b_) in bundles:
    if a in items and b_ in items:
        m.addConstr(b[a, b_] <= y[a], name=f'bundle1_{a}_{b_}')
        m.addConstr(b[a, b_] <= y[b_], name=f'bundle2_{a}_{b_}')
        m.addConstr(b[a, b_] >= y[a] + y[b_] - 1, name=f'bundle3_{a}_{b_}')
    else:
        m.addConstr(b[a, b_] == 0, name=f'bundle0_{a}_{b_}')
obj_benefit = gp.quicksum((benefit[i] * x[i] for i in items))
obj_itemfee = gp.quicksum((item_fee[i] * y[i] for i in items))
obj_catfee = gp.quicksum((cat_fee[c] * z[c] for c in categories))
obj_bundle = gp.quicksum((bundle_bonus[a, b_] * b[a, b_] for (a, b_) in bundles))
m.setObjective(obj_benefit - obj_itemfee - obj_catfee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Maximum net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')