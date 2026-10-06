import gurobipy as gp
import pandas as pd
import numpy as np
df_bundle = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_01/export_01.csv', sep=',')
df_capacity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_02/export_02.csv', sep=',')
df_category = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_03/export_03.csv', sep=',')
df_identity = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_04/export_04.csv', sep=',')
df_incompat = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_05/export_05.csv', sep=',')
df_requires = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_03/export_09.csv', sep=',')
df_usage_10 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_04/export_10.csv', sep=',')
df_usage_11 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_05/export_11.csv', sep=',')
df_item_06 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_06/export_06.csv', sep=',')
df_item_07 = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_01/export_07.csv', sep=',')
df_market = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant36/inputs/batch_02/export_08.csv', sep=',')
item_tables = [df_item_06, df_item_07]
item_rows = pd.concat(item_tables, ignore_index=True)
item_rows = item_rows.drop_duplicates(subset=['item_ref'])
item_refs = item_rows['item_ref'].astype(str).tolist()
item_to_category = dict(zip(item_rows['item_ref'].astype(str), item_rows['category'].astype(str)))
item_to_authorized = dict(zip(item_rows['item_ref'].astype(str), item_rows['authorized'].astype(int)))
item_to_minlot = dict(zip(item_rows['item_ref'].astype(str), item_rows['minimum_lot'].astype(int)))
item_to_maxorder = dict(zip(item_rows['item_ref'].astype(str), item_rows['maximum_order'].astype(int)))
item_to_location = dict(zip(item_rows['item_ref'].astype(str), item_rows['location_id'].astype(str)))
item_to_unit_benefit = dict(zip(item_rows['item_ref'].astype(str), item_rows['unit_benefit_cents'].astype(int)))
item_to_item_fee = dict(zip(item_rows['item_ref'].astype(str), item_rows['item_fee_cents'].astype(int)))
category_rows = df_category
categories = category_rows['category'].astype(str).tolist()
cat_to_minqty = dict(zip(category_rows['category'].astype(str), category_rows['minimum_quantity'].astype(int)))
cat_to_maxqty = dict(zip(category_rows['category'].astype(str), category_rows['maximum_quantity'].astype(int)))
cat_to_fee = dict(zip(category_rows['category'].astype(str), category_rows['activation_fee_cents'].astype(int)))
resources = sorted(set(df_capacity['resource'].astype(str)).union(df_usage_10['resource'].astype(str)).union(df_usage_11['resource'].astype(str)))
bundle_rows = df_bundle
bundles = bundle_rows.index.tolist()
bundle_to_itema = dict(zip(bundles, bundle_rows['item_a'].astype(str)))
bundle_to_itemb = dict(zip(bundles, bundle_rows['item_b'].astype(str)))
bundle_to_bonus = dict(zip(bundles, bundle_rows['bonus_cents'].astype(int)))
incompat_pairs = list(zip(df_incompat['item_a'].astype(str), df_incompat['item_b'].astype(str)))
requires_pairs = list(zip(df_requires['item_ref'].astype(str), df_requires['prerequisite_ref'].astype(str)))
usage_rows = pd.concat([df_usage_10, df_usage_11], ignore_index=True)
usage_dict = {}
for _, row in usage_rows.iterrows():
    usage_dict[str(row['item_ref']), str(row['resource'])] = int(row['amount'])
cap_ledger = df_capacity.groupby('resource', as_index=True)['amount'].sum().to_dict()
for r in set([k[1] for k in usage_dict.keys()]):
    if r not in cap_ledger:
        raise ValueError(f'Resource {r} in usage table not found in capacity_ledger.')
cat_to_items = {c: [] for c in categories}
for i in item_refs:
    c = item_to_category[i]
    if c in cat_to_items:
        cat_to_items[c].append(i)

def solve_problem():
    m = gp.Model('FC_EAST_HVAC_Placement')
    x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
    y = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
    b = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
    for i in item_refs:
        auth = item_to_authorized[i]
        minlot = item_to_minlot[i]
        maxorder = item_to_maxorder[i]
        if auth == 0:
            m.addConstr(x[i] == 0, name=f'unauth_{i}')
            m.addConstr(y[i] == 0, name=f'unauth_y_{i}')
        else:
            m.addConstr(x[i] <= maxorder * y[i], name=f'xmax_{i}')
            m.addConstr(x[i] >= minlot * y[i], name=f'xmin_{i}')
            m.addConstr(x[i] <= maxorder, name=f'xmax2_{i}')
            m.addConstr(x[i] >= 0, name=f'xnonneg_{i}')
    for r in resources:
        m.addConstr(gp.quicksum((usage_dict.get((i, r), 0) * x[i] for i in item_refs)) <= cap_ledger[r], name=f'cap_{r}')
    for c in categories:
        items_in_c = cat_to_items[c]
        minqty = cat_to_minqty[c]
        maxqty = cat_to_maxqty[c]
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= minqty * z[c], name=f'catmin_{c}')
        m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= maxqty * z[c], name=f'catmax_{c}')
        for i in items_in_c:
            m.addConstr(y[i] <= z[c], name=f'catlink_{i}_{c}')
    for i, j in incompat_pairs:
        if i in item_refs and j in item_refs:
            m.addConstr(y[i] + y[j] <= 1, name=f'incompat_{i}_{j}')
    for i, prereq in requires_pairs:
        if i in item_refs and prereq in item_refs:
            m.addConstr(y[i] <= y[prereq], name=f'requires_{i}_{prereq}')
    for bundle in bundles:
        ia = bundle_to_itema[bundle]
        ib = bundle_to_itemb[bundle]
        if ia in item_refs and ib in item_refs:
            m.addConstr(b[bundle] <= y[ia], name=f'bundle1_{bundle}')
            m.addConstr(b[bundle] <= y[ib], name=f'bundle2_{bundle}')
            m.addConstr(b[bundle] >= y[ia] + y[ib] - 1, name=f'bundle3_{bundle}')
        else:
            m.addConstr(b[bundle] == 0, name=f'bundle0_{bundle}')
    obj = gp.quicksum((item_to_unit_benefit[i] * x[i] for i in item_refs)) - gp.quicksum((item_to_item_fee[i] * y[i] for i in item_refs)) - gp.quicksum((cat_to_fee[c] * z[c] for c in categories)) + gp.quicksum((bundle_to_bonus[bundle] * b[bundle] for bundle in bundles))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()