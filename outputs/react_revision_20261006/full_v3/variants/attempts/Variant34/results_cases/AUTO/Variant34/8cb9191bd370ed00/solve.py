import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv'
f_item1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv'
f_item2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv'
f_usage1 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv'
f_usage2 = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv'
benefit_df = pd.read_csv(f_benefit, sep=',')
bundle_df = pd.read_csv(f_bundle, sep=',')
capacity_ledger_df = pd.read_csv(f_capacity_ledger, sep=',')
category_df = pd.read_csv(f_category, sep=',')
identity_df = pd.read_csv(f_identity, sep=',')
incompatible_df = pd.read_csv(f_incompatible, sep=',')
item1_df = pd.read_csv(f_item1, sep=',')
item2_df = pd.read_csv(f_item2, sep=',')
item_fee_df = pd.read_csv(f_item_fee, sep=',')
market_df = pd.read_csv(f_market, sep=',')
requires_df = pd.read_csv(f_requires, sep=',')
usage1_df = pd.read_csv(f_usage1, sep=',')
usage2_df = pd.read_csv(f_usage2, sep=',')
item_tables = [item1_df, item2_df]
item_options = []
item_option_info = dict()
for df in item_tables:
    for (_, row) in df.iterrows():
        item_ref = str(row['item_ref'])
        location_id = str(row['location_id'])
        key = (item_ref, location_id)
        item_options.append(key)
        item_option_info[key] = {'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'configuration_id': str(row['configuration_id'])}
item_options = list(dict.fromkeys(item_options))
benefit_per_itemref = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
benefit_per_option = {}
for key in item_options:
    item_ref = key[0]
    if item_ref not in benefit_per_itemref:
        raise ValueError(f'Missing benefit for item_ref {item_ref}')
    benefit_per_option[key] = int(benefit_per_itemref[item_ref])
item_fee_map = dict(zip(item_fee_df['item_ref'].astype(str), item_fee_df['activation_fee_cents']))
activation_fee_per_option = {}
for key in item_options:
    item_ref = key[0]
    if item_ref not in item_fee_map:
        raise ValueError(f'Missing item_fee for item_ref {item_ref}')
    activation_fee_per_option[key] = int(item_fee_map[item_ref])
category_df['category'] = category_df['category'].astype(str)
category_set = set(category_df['category'])
category_activation_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
category_min = dict(zip(category_df['category'], category_df['minimum_quantity']))
category_max = dict(zip(category_df['category'], category_df['maximum_quantity']))
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = int(row['bonus_cents'])
usage_df = pd.concat([usage1_df, usage2_df], ignore_index=True)
usage_map = {}
for (_, row) in usage_df.iterrows():
    item_ref = str(row['item_ref'])
    resource = str(row['resource'])
    amount = int(row['amount'])
    unit = str(row['unit']).strip().casefold()
    if unit == 'liter':
        amount_ml = amount * 1000
    elif unit == 'ml':
        amount_ml = amount
    else:
        raise ValueError(f'Unknown unit {unit} for usage')
    usage_map[item_ref, resource] = amount_ml
capacity_ledger_df['resource'] = capacity_ledger_df['resource'].astype(str)
capacity_ledger_df['unit'] = capacity_ledger_df['unit'].str.strip().str.casefold()
capacity_per_resource = {}
for (resource, group) in capacity_ledger_df.groupby('resource'):
    total = 0
    for (_, row) in group.iterrows():
        amt = int(row['amount'])
        unit = str(row['unit'])
        if unit == 'ml':
            total += amt
        elif unit == 'liter':
            total += amt * 1000
        else:
            raise ValueError(f'Unknown unit {unit} in capacity_ledger')
    capacity_per_resource[resource] = total
incompat_pairs = set()
for (_, row) in incompatible_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompat_pairs.add((a, b))
    incompat_pairs.add((b, a))
prereq_pairs = []
for (_, row) in requires_df.iterrows():
    i = str(row['item_ref'])
    prereq = str(row['prerequisite_ref'])
    prereq_pairs.append((i, prereq))
itemref_to_options = {}
for key in item_options:
    item_ref = key[0]
    itemref_to_options.setdefault(item_ref, []).append(key)
category_to_options = {}
for key in item_options:
    cat = item_option_info[key]['category']
    category_to_options.setdefault(cat, []).append(key)
resource_to_options = {}
for key in item_options:
    resource = key[1]
    resource_to_options.setdefault(resource, []).append(key)

def get_option_keys_for_itemref(item_ref):
    return itemref_to_options.get(item_ref, [])
m = gp.Model('market_square_merchandising')
m.Params.MIPGap = 0.0001
x = m.addVars(item_options, lb=0, ub=[item_option_info[i]['maximum_order'] if item_option_info[i]['authorized'] else 0 for i in item_options], vtype=gp.GRB.INTEGER, name='')
y = m.addVars(item_options, vtype=gp.GRB.BINARY, name='')
z = m.addVars(list(category_set), vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for i in item_options:
    info = item_option_info[i]
    if info['authorized'] == 0:
        m.addConstr(x[i] == 0, name='unauth_x')
        m.addConstr(y[i] == 0, name='unauth_y')
    else:
        m.addConstr(x[i] >= info['minimum_lot'] * y[i], name='minlot')
        m.addConstr(x[i] <= info['maximum_order'] * y[i], name='maxorder')
for (resource, cap) in capacity_per_resource.items():
    expr = gp.LinExpr()
    for i in resource_to_options.get(resource, []):
        item_ref = i[0]
        usage = usage_map.get((item_ref, resource), 0)
        expr += usage * x[i]
    m.addConstr(expr <= cap, name='capacity_' + resource)
for c in category_set:
    options = category_to_options.get(c, [])
    expr = gp.quicksum((x[i] for i in options))
    m.addConstr(expr >= category_min[c] * z[c], name='cat_min_' + c)
    m.addConstr(expr <= category_max[c] * z[c], name='cat_max_' + c)
    for i in options:
        m.addConstr(y[i] <= z[c], name='cat_act1_' + c)
    m.addConstr(expr <= category_max[c] * z[c], name='cat_link_' + c)
for i in item_options:
    m.addConstr(x[i] >= y[i], name='y_lb')
    m.addConstr(x[i] <= item_option_info[i]['maximum_order'] * y[i], name='y_ub')
for (a, b) in incompat_pairs:
    options_a = itemref_to_options.get(a, [])
    options_b = itemref_to_options.get(b, [])
    for ia in options_a:
        for ib in options_b:
            if ia != ib:
                m.addConstr(y[ia] + y[ib] <= 1, name='incompat')
for (i_ref, prereq_ref) in prereq_pairs:
    options_i = itemref_to_options.get(i_ref, [])
    options_prereq = itemref_to_options.get(prereq_ref, [])
    for oi in options_i:
        for opreq in options_prereq:
            m.addConstr(y[oi] <= y[opreq], name='prereq')
for (a, b) in bundle_pairs:
    options_a = itemref_to_options.get(a, [])
    options_b = itemref_to_options.get(b, [])
    m.addConstr(b[a, b] <= gp.quicksum((y[ia] for ia in options_a)), name='bundle_a')
    m.addConstr(b[a, b] <= gp.quicksum((y[ib] for ib in options_b)), name='bundle_b')
    m.addConstr(b[a, b] >= gp.quicksum((y[ia] for ia in options_a)) + gp.quicksum((y[ib] for ib in options_b)) - 1, name='bundle_link')
obj_benefit = gp.quicksum((benefit_per_option[i] * x[i] for i in item_options))
obj_item_fee = gp.quicksum((activation_fee_per_option[i] * y[i] for i in item_options))
obj_cat_fee = gp.quicksum((category_activation_fee[c] * z[c] for c in category_set))
obj_bundle = gp.quicksum((bundle_bonus[bundle] * b[bundle] for bundle in bundle_pairs))
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')