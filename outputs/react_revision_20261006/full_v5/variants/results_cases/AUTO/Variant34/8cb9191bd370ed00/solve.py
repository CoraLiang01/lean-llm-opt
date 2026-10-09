import gurobipy as gp
import pandas as pd
import numpy as np
benefit_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_01.csv', sep=',')
bundle_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_02.csv', sep=',')
capacity_ledger_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_03.csv', sep=',')
category_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_04.csv', sep=',')
identity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_05.csv', sep=',')
incompatible_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_06.csv', sep=',')
item1_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_07.csv', sep=',')
item2_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_02/export_08.csv', sep=',')
item_fee_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_03/export_09.csv', sep=',')
market_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_04/export_10.csv', sep=',')
requires_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_05/export_11.csv', sep=',')
usage2_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_06/export_12.csv', sep=',')
usage1_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant34/inputs/batch_01/export_13.csv', sep=',')
item_tables = [item1_df, item2_df]
option_rows = []
for df in item_tables:
    for (_, row) in df.iterrows():
        option_rows.append({'item_ref': str(row['item_ref']), 'category': str(row['category']), 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'location_id': str(row['location_id'])})
options = []
option_key_to_cat = {}
option_key_to_auth = {}
option_key_to_minlot = {}
option_key_to_maxorder = {}
option_key_to_loc = {}
option_key_to_itemref = {}
for row in option_rows:
    key = (row['item_ref'], row['location_id'])
    options.append(key)
    option_key_to_cat[key] = row['category']
    option_key_to_auth[key] = row['authorized']
    option_key_to_minlot[key] = row['minimum_lot']
    option_key_to_maxorder[key] = row['maximum_order']
    option_key_to_loc[key] = row['location_id']
    option_key_to_itemref[key] = row['item_ref']
options = list(dict.fromkeys(options))
benefit_df['item_ref'] = benefit_df['item_ref'].astype(str)
itemref_to_benefit = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
option_key_to_benefit = {}
for key in options:
    item_ref = key[0]
    if item_ref not in itemref_to_benefit:
        raise ValueError(f'Missing benefit for item_ref {item_ref}')
    option_key_to_benefit[key] = int(itemref_to_benefit[item_ref])
item_fee_df['item_ref'] = item_fee_df['item_ref'].astype(str)
itemref_to_fee = item_fee_df.set_index('item_ref')['activation_fee_cents'].to_dict()
option_key_to_fee = {}
for key in options:
    item_ref = key[0]
    if item_ref not in itemref_to_fee:
        raise ValueError(f'Missing item_fee for item_ref {item_ref}')
    option_key_to_fee[key] = int(itemref_to_fee[item_ref])
category_df['category'] = category_df['category'].astype(str)
categories = list(category_df['category'].unique())
cat_to_minqty = category_df.set_index('category')['minimum_quantity'].to_dict()
cat_to_maxqty = category_df.set_index('category')['maximum_quantity'].to_dict()
cat_to_fee = category_df.set_index('category')['activation_fee_cents'].to_dict()

def usage_df_to_dict(usage_df):
    usage = {}
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
            raise ValueError(f'Unknown unit {unit} in usage table')
        usage[item_ref, resource] = amount_ml
    return usage
usage1 = usage_df_to_dict(usage1_df)
usage2 = usage_df_to_dict(usage2_df)
usage = usage1.copy()
usage.update(usage2)
option_key_to_resource_usage = {}
resources = set()
for key in options:
    item_ref = key[0]
    for resource in capacity_ledger_df['resource'].unique():
        k = (item_ref, resource)
        if k in usage:
            option_key_to_resource_usage[key, resource] = usage[k]
            resources.add(resource)
resources = list(resources)
capacity_ledger_df['resource'] = capacity_ledger_df['resource'].astype(str)
capacity_ledger_df['unit'] = capacity_ledger_df['unit'].astype(str)
resource_to_capacity = {}
for resource in capacity_ledger_df['resource'].unique():
    df_r = capacity_ledger_df[capacity_ledger_df['resource'] == resource]
    total = 0
    for (_, row) in df_r.iterrows():
        amt = int(row['amount'])
        unit = row['unit'].strip().casefold()
        if unit == 'liter':
            amt_ml = amt * 1000
        elif unit == 'ml':
            amt_ml = amt
        else:
            raise ValueError(f'Unknown unit {unit} in capacity_ledger')
        total += amt_ml
    resource_to_capacity[resource] = total
incompat_pairs = []
for (_, row) in incompatible_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompat_pairs.append((a, b))
requires_pairs = []
for (_, row) in requires_df.iterrows():
    item_ref = str(row['item_ref'])
    prereq = str(row['prerequisite_ref'])
    requires_pairs.append((item_ref, prereq))
bundle_pairs = []
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = int(row['bonus_cents'])
itemref_to_optionkeys = {}
for key in options:
    item_ref = key[0]
    itemref_to_optionkeys.setdefault(item_ref, []).append(key)
m = gp.Model('market_square_merchandising')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
y = m.addVars(options, vtype=gp.GRB.BINARY, name='')
z = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
for o in options:
    auth = option_key_to_auth[o]
    minlot = option_key_to_minlot[o]
    maxorder = option_key_to_maxorder[o]
    if auth == 0:
        m.addConstr(x[o] == 0, name='auth0_%s_%s' % o)
        m.addConstr(y[o] == 0, name='auth0y_%s_%s' % o)
    else:
        m.addConstr(x[o] >= minlot * y[o], name='minlot_%s_%s' % o)
        m.addConstr(x[o] <= maxorder * y[o], name='maxorder_%s_%s' % o)
        m.addConstr(x[o] >= 0, name='nonneg_%s_%s' % o)
        m.addConstr(y[o] <= 1, name='ybin_%s_%s' % o)
for resource in resources:
    expr = gp.LinExpr()
    for o in options:
        if (o, resource) in option_key_to_resource_usage:
            expr += option_key_to_resource_usage[o, resource] * x[o]
    m.addConstr(expr <= resource_to_capacity[resource], name='cap_%s' % resource)
for c in categories:
    options_in_c = [o for o in options if option_key_to_cat[o] == c]
    minqty = cat_to_minqty[c]
    maxqty = cat_to_maxqty[c]
    m.addConstr(gp.quicksum((x[o] for o in options_in_c)) >= minqty * z[c], name='catmin_%s' % c)
    m.addConstr(gp.quicksum((x[o] for o in options_in_c)) <= maxqty * z[c], name='catmax_%s' % c)
    for o in options_in_c:
        m.addConstr(y[o] <= z[c], name='catlink_%s_%s_%s' % (c, o[0], o[1]))
for (a, b) in incompat_pairs:
    options_a = itemref_to_optionkeys.get(a, [])
    options_b = itemref_to_optionkeys.get(b, [])
    for oa in options_a:
        for ob in options_b:
            m.addConstr(y[oa] + y[ob] <= 1, name='incompat_%s_%s__%s_%s' % (oa[0], oa[1], ob[0], ob[1]))
for (item_ref, prereq) in requires_pairs:
    options_item = itemref_to_optionkeys.get(item_ref, [])
    options_prereq = itemref_to_optionkeys.get(prereq, [])
    if not options_prereq:
        for oi in options_item:
            m.addConstr(y[oi] == 0, name='req_missing_%s_%s' % (oi[0], oi[1]))
    else:
        for oi in options_item:
            m.addConstr(y[oi] <= gp.quicksum((y[op] for op in options_prereq)), name='req_%s_%s' % (oi[0], oi[1]))
for (a, b) in bundle_pairs:
    options_a = itemref_to_optionkeys.get(a, [])
    options_b = itemref_to_optionkeys.get(b, [])
    if not options_a or not options_b:
        m.addConstr(b[a, b] == 0, name='bundle_missing_%s_%s' % (a, b))
        continue
    m.addConstr(b[a, b] <= gp.quicksum((y[oa] for oa in options_a)), name='bundlea_%s_%s' % (a, b))
    m.addConstr(b[a, b] <= gp.quicksum((y[ob] for ob in options_b)), name='bundleb_%s_%s' % (a, b))
    m.addConstr(b[a, b] >= gp.quicksum((y[oa] for oa in options_a)) + gp.quicksum((y[ob] for ob in options_b)) - 1, name='bundleboth_%s_%s' % (a, b))
obj = gp.LinExpr()
for o in options:
    obj += option_key_to_benefit[o] * x[o]
    obj -= option_key_to_fee[o] * y[o]
for c in categories:
    obj -= cat_to_fee[c] * z[c]
for bundle in bundle_pairs:
    obj += bundle_bonus[bundle] * b[bundle]
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for o in options:
        print(f'x{str(o)} {x[o].VarName} {x[o].X}')
        print(f'y{str(o)} {y[o].VarName} {y[o].X}')
    for c in categories:
        print(f'z{c} {z[c].VarName} {z[c].X}')
    for bundle in bundle_pairs:
        print(f'b{bundle} {b[bundle].VarName} {b[bundle].X}')
else:
    print(f'Solver status: {m.status}')