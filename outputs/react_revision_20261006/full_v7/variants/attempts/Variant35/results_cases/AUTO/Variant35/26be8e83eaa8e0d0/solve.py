import gurobipy as gp
import pandas as pd
import numpy as np
f_benefit = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_01.csv'
f_bundle = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_02.csv'
f_capacity_ledger = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_03.csv'
f_category = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_04.csv'
f_fx = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_05.csv'
f_identity = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_06.csv'
f_incompatible = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_01/export_07.csv'
f_item = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_02/export_08.csv'
f_item_fee = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_03/export_09.csv'
f_market = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_04/export_10.csv'
f_requires = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_05/export_11.csv'
f_usage = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant35/inputs/batch_06/export_12.csv'
benefit_df = pd.read_csv(f_benefit, dtype=str, keep_default_na=False)
bundle_df = pd.read_csv(f_bundle, dtype=str, keep_default_na=False)
capacity_ledger_df = pd.read_csv(f_capacity_ledger, dtype=str, keep_default_na=False)
category_df = pd.read_csv(f_category, dtype=str, keep_default_na=False)
fx_df = pd.read_csv(f_fx, dtype=str, keep_default_na=False)
identity_df = pd.read_csv(f_identity, dtype=str, keep_default_na=False)
incompatible_df = pd.read_csv(f_incompatible, dtype=str, keep_default_na=False)
item_df = pd.read_csv(f_item, dtype=str, keep_default_na=False)
item_fee_df = pd.read_csv(f_item_fee, dtype=str, keep_default_na=False)
market_df = pd.read_csv(f_market, dtype=str, keep_default_na=False)
requires_df = pd.read_csv(f_requires, dtype=str, keep_default_na=False)
usage_df = pd.read_csv(f_usage, dtype=str, keep_default_na=False)
item_df['authorized'] = item_df['authorized'].astype(int)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(int)
item_df['maximum_order'] = item_df['maximum_order'].astype(int)
item_keys = list(item_df['item_ref'])
item_set = set(item_keys)
item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
item_to_platform = dict(zip(item_df['item_ref'], item_df['location_id']))
item_to_minlot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_to_maxorder = dict(zip(item_df['item_ref'], item_df['maximum_order']))
item_to_authorized = dict(zip(item_df['item_ref'], item_df['authorized']))
category_keys = list(category_df['category'])
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(int)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(int)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(int)
cat_minqty = dict(zip(category_df['category'], category_df['minimum_quantity']))
cat_maxqty = dict(zip(category_df['category'], category_df['maximum_quantity']))
cat_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
platforms = sorted(set(item_df['location_id']) | set(capacity_ledger_df['resource']) | set(usage_df['resource']))
fx_df['usd_cents_numerator'] = fx_df['usd_cents_numerator'].astype(int)
fx_df['denominator'] = fx_df['denominator'].astype(int)
currency_to_fx = dict()
for (_, row) in fx_df.iterrows():
    currency_to_fx[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
benefit_df['amount'] = benefit_df['amount'].astype(int)
benefit_per_item = dict()
for item in item_keys:
    df = benefit_df[benefit_df['item_ref'] == item]
    total = 0
    for (_, row) in df.iterrows():
        currency = row['currency']
        amount = row['amount']
        if currency not in currency_to_fx:
            raise ValueError(f'Missing FX rate for currency {currency}')
        (num, denom) = currency_to_fx[currency]
        val = amount * num / denom
        total += val
    benefit_per_item[item] = int(round(total))
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(int)
item_fee = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
for item in item_keys:
    if item not in item_fee:
        item_fee[item] = 0
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(int)
bundle_keys = list(bundle_df.index)
bundle_pairs = []
bundle_bonus = dict()
for (idx, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    bundle_pairs.append((a, b))
    bundle_bonus[a, b] = row['bonus_cents']
usage_df['amount'] = usage_df['amount'].astype(int)
item_usage = dict()
for (_, row) in usage_df.iterrows():
    item = row['item_ref']
    platform = row['resource']
    amount = row['amount']
    unit = row['unit']
    if unit.casefold() == 'gb':
        amount_mb = amount * 1000
    elif unit.casefold() == 'mb':
        amount_mb = amount
    else:
        raise ValueError(f'Unknown unit {unit} for item {item}')
    item_usage[item] = (platform, amount_mb)
for item in item_keys:
    if item not in item_usage:
        item_usage[item] = (item_to_platform[item], 0)
capacity_ledger_df['amount'] = capacity_ledger_df['amount'].astype(int)
platform_capacity = dict()
for platform in platforms:
    df = capacity_ledger_df[capacity_ledger_df['resource'] == platform]
    total = df['amount'].sum()
    platform_capacity[platform] = total
incompat_pairs = []
for (_, row) in incompatible_df.iterrows():
    i = row['item_a']
    j = row['item_b']
    if i in item_set and j in item_set:
        incompat_pairs.append((i, j))
prereq_pairs = []
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    if i in item_set and prereq in item_set:
        prereq_pairs.append((i, prereq))
cat_to_items = {c: [] for c in category_keys}
for item in item_keys:
    c = item_to_category[item]
    cat_to_items[c].append(item)
plat_to_items = {p: [] for p in platforms}
for item in item_keys:
    p = item_to_platform[item]
    plat_to_items[p].append(item)

def solve_problem():
    m = gp.Model('GameEditionAllocation')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(item_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    activation_vars = m.addVars(item_keys, vtype=gp.GRB.BINARY, name='')
    category_activation_vars = m.addVars(category_keys, vtype=gp.GRB.BINARY, name='')
    bundle_activation_vars = m.addVars(bundle_pairs, vtype=gp.GRB.BINARY, name='')
    for i in item_keys:
        auth = item_to_authorized[i]
        minlot = item_to_minlot[i]
        maxorder = item_to_maxorder[i]
        if auth == 0:
            m.addConstr(quantity_vars[i] == 0, name=f'auth_{i}')
            m.addConstr(activation_vars[i] == 0, name=f'authz_{i}')
        else:
            m.addConstr(quantity_vars[i] >= minlot * activation_vars[i], name=f'minlot_{i}')
            m.addConstr(quantity_vars[i] <= maxorder * activation_vars[i], name=f'maxorder_{i}')
            m.addConstr(quantity_vars[i] <= maxorder, name=f'maxorder2_{i}')
            m.addConstr(quantity_vars[i] >= 0, name=f'nonneg_{i}')
    for c in category_keys:
        items_in_c = cat_to_items[c]
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) >= cat_minqty[c] * category_activation_vars[c], name=f'cat_min_{c}')
        m.addConstr(gp.quicksum((quantity_vars[i] for i in items_in_c)) <= cat_maxqty[c] * category_activation_vars[c], name=f'cat_max_{c}')
        for i in items_in_c:
            m.addConstr(category_activation_vars[c] >= activation_vars[i], name=f'cat_act_{c}_{i}')
    for p in platforms:
        items_in_p = plat_to_items[p]
        m.addConstr(gp.quicksum((item_usage[i][1] * quantity_vars[i] for i in items_in_p)) <= platform_capacity[p], name=f'plat_cap_{p}')
    for (i, j) in incompat_pairs:
        m.addConstr(activation_vars[i] + activation_vars[j] <= 1, name=f'incompat_{i}_{j}')
    for (i, prereq) in prereq_pairs:
        m.addConstr(activation_vars[i] <= activation_vars[prereq], name=f'prereq_{i}_{prereq}')
    for (a, b) in bundle_pairs:
        m.addConstr(bundle_activation_vars[a, b] <= activation_vars[a], name=f'bundle1_{a}_{b}')
        m.addConstr(bundle_activation_vars[a, b] <= activation_vars[b], name=f'bundle2_{a}_{b}')
        m.addConstr(bundle_activation_vars[a, b] >= activation_vars[a] + activation_vars[b] - 1, name=f'bundle3_{a}_{b}')
    obj = gp.LinExpr()
    obj += gp.quicksum((benefit_per_item[i] * quantity_vars[i] for i in item_keys))
    obj -= gp.quicksum((item_fee[i] * activation_vars[i] for i in item_keys))
    obj -= gp.quicksum((cat_fee[c] * category_activation_vars[c] for c in category_keys))
    obj += gp.quicksum((bundle_bonus[a, b] * bundle_activation_vars[a, b] for (a, b) in bundle_pairs))
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')