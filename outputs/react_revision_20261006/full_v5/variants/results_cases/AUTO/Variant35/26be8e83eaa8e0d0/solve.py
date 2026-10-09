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
df_benefit = pd.read_csv(f_benefit, sep=',')
df_bundle = pd.read_csv(f_bundle, sep=',')
df_capacity_ledger = pd.read_csv(f_capacity_ledger, sep=',')
df_category = pd.read_csv(f_category, sep=',')
df_fx = pd.read_csv(f_fx, sep=',')
df_identity = pd.read_csv(f_identity, sep=',')
df_incompatible = pd.read_csv(f_incompatible, sep=',')
df_item = pd.read_csv(f_item, sep=',')
df_item_fee = pd.read_csv(f_item_fee, sep=',')
df_market = pd.read_csv(f_market, sep=',')
df_requires = pd.read_csv(f_requires, sep=',')
df_usage = pd.read_csv(f_usage, sep=',')
item_rows = df_item.copy()
item_refs = list(item_rows['item_ref'])
platforms = sorted(df_item['location_id'].unique())
categories = sorted(df_category['category'].unique())
bundle_keys = list(df_bundle[['item_a', 'item_b']].itertuples(index=False, name=None))
incompat_pairs = list(df_incompatible[['item_a', 'item_b']].itertuples(index=False, name=None))
prereq_pairs = list(df_requires[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
fx_map = df_fx.set_index('currency')[['usd_cents_numerator', 'denominator']].to_dict(orient='index')

def benefit_usd_cents(row):
    currency = row['currency']
    fx = fx_map[currency]
    return row['amount'] * fx['usd_cents_numerator'] / fx['denominator']
benefit_df = df_benefit.copy()
benefit_df['usd_cents'] = benefit_df.apply(benefit_usd_cents, axis=1)
benefit_per_item = benefit_df.groupby('item_ref')['usd_cents'].sum().to_dict()
for i in item_refs:
    if i not in benefit_per_item:
        benefit_per_item[i] = 0.0
item_fee_map = df_item_fee.set_index('item_ref')['activation_fee_cents'].to_dict()
for i in item_refs:
    if i not in item_fee_map:
        item_fee_map[i] = 0
category_fee_map = df_category.set_index('category')['activation_fee_cents'].to_dict()
bundle_bonus_map = {}
for (_, row) in df_bundle.iterrows():
    bundle_bonus_map[row['item_a'], row['item_b']] = row['bonus_cents']
usage_map = {}
for (_, row) in df_usage.iterrows():
    item = row['item_ref']
    resource = row['resource']
    amount = row['amount']
    unit = row['unit']
    if unit.casefold() == 'gb':
        amount_mb = amount * 1000
    elif unit.casefold() == 'mb':
        amount_mb = amount
    else:
        raise ValueError(f'Unknown unit {unit} in usage table')
    usage_map[item, resource] = amount_mb
capacity_ledger_df = df_capacity_ledger.copy()
capacity_per_platform = capacity_ledger_df.groupby('resource')['amount'].sum().to_dict()
for p in platforms:
    if p not in capacity_per_platform:
        capacity_per_platform[p] = 0
item_param_map = {}
for (_, row) in item_rows.iterrows():
    item_param_map[row['item_ref']] = {'category': row['category'], 'authorized': int(row['authorized']), 'minimum_lot': int(row['minimum_lot']), 'maximum_order': int(row['maximum_order']), 'platform': row['location_id']}
category_bounds = {}
for (_, row) in df_category.iterrows():
    category_bounds[row['category']] = {'min': int(row['minimum_quantity']), 'max': int(row['maximum_quantity'])}
m = gp.Model('GameEditionAllocation')
x = m.addVars(item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
w = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b = m.addVars(bundle_keys, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    p = item_param_map[i]
    if p['authorized'] == 0:
        m.addConstr(x[i] == 0, name=f'auth_{i}')
        m.addConstr(z[i] == 0, name=f'authz_{i}')
    else:
        m.addConstr(x[i] <= p['maximum_order'] * z[i], name=f'link_ub_{i}')
        m.addConstr(x[i] >= p['minimum_lot'] * z[i], name=f'link_lb_{i}')
        m.addConstr(x[i] <= p['maximum_order'], name=f'maxorder_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
for c in categories:
    items_in_c = [i for i in item_refs if item_param_map[i]['category'] == c]
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) >= category_bounds[c]['min'], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_c)) <= category_bounds[c]['max'], name=f'cat_max_{c}')
    for i in items_in_c:
        m.addConstr(z[i] <= w[c], name=f'cat_link_{c}_{i}')
    m.addConstr(w[c] <= gp.quicksum((z[i] for i in items_in_c)), name=f'cat_link2_{c}')
for p in platforms:
    items_on_p = [i for i in item_refs if item_param_map[i]['platform'] == p]
    m.addConstr(gp.quicksum((x[i] * usage_map.get((i, p), 0) for i in items_on_p)) <= capacity_per_platform[p], name=f'cap_{p}')
for (i, j) in incompat_pairs:
    if i in item_refs and j in item_refs:
        m.addConstr(z[i] + z[j] <= 1, name=f'incomp_{i}_{j}')
for (i, prereq) in prereq_pairs:
    if i in item_refs and prereq in item_refs:
        m.addConstr(z[i] <= z[prereq], name=f'prereq_{i}_{prereq}')
for (a, b) in bundle_keys:
    if a in item_refs and b in item_refs:
        m.addConstr(b[a, b] <= z[a], name=f'bundle_a_{a}_{b}')
        m.addConstr(b[a, b] <= z[b], name=f'bundle_b_{a}_{b}')
        m.addConstr(b[a, b] >= z[a] + z[b] - 1, name=f'bundle_link_{a}_{b}')
    else:
        m.addConstr(b[a, b] == 0, name=f'bundle_invalid_{a}_{b}')
obj = gp.LinExpr()
obj += gp.quicksum((benefit_per_item[i] * x[i] for i in item_refs))
obj -= gp.quicksum((item_fee_map[i] * z[i] for i in item_refs))
obj -= gp.quicksum((category_fee_map[c] * w[c] for c in categories))
obj += gp.quicksum((bundle_bonus_map[a, b] * b[a, b] for (a, b) in bundle_keys))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()