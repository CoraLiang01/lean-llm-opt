import gurobipy as gp
import pandas as pd
import numpy as np
import re
UNIT_CONV = {('liter', 'ml'): 1000, ('ml', 'ml'): 1, ('hour', 'minute'): 60, ('minute', 'minute'): 1, ('kwh', 'wh'): 1000, ('wh', 'wh'): 1}
CUTOFF_DATE = '2026-03-12'
BUSINESS_UNIT = 'NORTH'
csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in csv_paths]

def get_table(table_name):
    for df in dfs:
        if 'table' in df.columns and df['table'].str.casefold().str.strip().eq(table_name.casefold()).any():
            return df
    raise ValueError(f'Table {table_name} not found in loaded CSVs.')

def filter_latest(df, date_col='effective_date'):
    df = df[df[date_col] <= CUTOFF_DATE]
    df['revision'] = df['revision'].astype(int)
    idx = df.groupby(['tenant', 'table', 'record_id'])['revision'].transform('max') == df['revision']
    df_latest = df[idx].copy()
    if 'action' in df_latest.columns:
        df_latest = df_latest[~df_latest['action'].str.casefold().eq('delete')]
    return df_latest
item_tables = []
for tname in ['item', 'item']:
    try:
        item_tables.append(get_table(tname))
    except ValueError:
        pass
item_df = pd.concat(item_tables, ignore_index=True)
item_df = item_df[item_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
item_df = filter_latest(item_df)
item_df = item_df[item_df['item_ref'] != '']
cat_tables = []
for tname in ['category', 'category']:
    try:
        cat_tables.append(get_table(tname))
    except ValueError:
        pass
cat_df = pd.concat(cat_tables, ignore_index=True)
cat_df = cat_df[cat_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
cat_df = filter_latest(cat_df)
cat_df = cat_df[cat_df['category'] != '']
item_fee_tables = []
for tname in ['item_fee', 'item_fee']:
    try:
        item_fee_tables.append(get_table(tname))
    except ValueError:
        pass
item_fee_df = pd.concat(item_fee_tables, ignore_index=True)
item_fee_df = item_fee_df[item_fee_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
item_fee_df = filter_latest(item_fee_df)
item_fee_df = item_fee_df[item_fee_df['item_ref'] != '']
cat_fee_df = cat_df[['category', 'activation_fee_cents']].copy()
cat_fee_df['activation_fee_cents'] = pd.to_numeric(cat_fee_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
benefit_tables = []
for tname in ['benefit', 'benefit']:
    try:
        benefit_tables.append(get_table(tname))
    except ValueError:
        pass
benefit_df = pd.concat(benefit_tables, ignore_index=True)
benefit_df = benefit_df[benefit_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
benefit_df = filter_latest(benefit_df)
benefit_df = benefit_df[benefit_df['item_ref'] != '']
benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='coerce').fillna(0).astype(int)
bundle_tables = []
for tname in ['bundle', 'bundle']:
    try:
        bundle_tables.append(get_table(tname))
    except ValueError:
        pass
bundle_df = pd.concat(bundle_tables, ignore_index=True)
bundle_df = bundle_df[bundle_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
bundle_df = filter_latest(bundle_df)
bundle_df = bundle_df[(bundle_df['item_a'] != '') & (bundle_df['item_b'] != '')]
bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'], errors='coerce').fillna(0).astype(int)
incomp_tables = []
for tname in ['incompatible', 'incompatible']:
    try:
        incomp_tables.append(get_table(tname))
    except ValueError:
        pass
incomp_df = pd.concat(incomp_tables, ignore_index=True)
incomp_df = incomp_df[incomp_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
incomp_df = filter_latest(incomp_df)
incomp_df = incomp_df[(incomp_df['item_a'] != '') & (incomp_df['item_b'] != '')]
requires_tables = []
for tname in ['requires', 'requires']:
    try:
        requires_tables.append(get_table(tname))
    except ValueError:
        pass
requires_df = pd.concat(requires_tables, ignore_index=True)
requires_df = requires_df[requires_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
requires_df = filter_latest(requires_df)
requires_df = requires_df[(requires_df['item_ref'] != '') & (requires_df['prerequisite_ref'] != '')]
usage_tables = []
for tname in ['usage', 'usage']:
    try:
        usage_tables.append(get_table(tname))
    except ValueError:
        pass
usage_df = pd.concat(usage_tables, ignore_index=True)
usage_df = usage_df[usage_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
usage_df = filter_latest(usage_df)
usage_df = usage_df[(usage_df['item_ref'] != '') & (usage_df['resource'] != '') & (usage_df['amount'] != '') & (usage_df['unit'] != '')]
usage_df['amount'] = pd.to_numeric(usage_df['amount'], errors='coerce').fillna(0)
cap_tables = []
for tname in ['capacity_ledger', 'capacity_ledger']:
    try:
        cap_tables.append(get_table(tname))
    except ValueError:
        pass
cap_df = pd.concat(cap_tables, ignore_index=True)
cap_df = cap_df[cap_df['tenant'].str.casefold() == BUSINESS_UNIT.casefold()]
cap_df = filter_latest(cap_df)
cap_df = cap_df[(cap_df['resource'] != '') & (cap_df['amount'] != '') & (cap_df['unit'] != '')]
cap_df['amount'] = pd.to_numeric(cap_df['amount'], errors='coerce').fillna(0)
item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='coerce').fillna(0).astype(int)
item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'], errors='coerce').fillna(0).astype(int)
item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'], errors='coerce').fillna(0).astype(int)
items = item_df['item_ref'].unique().tolist()
item_to_cat = dict(zip(item_df['item_ref'], item_df['category']))
item_authorized = dict(zip(item_df['item_ref'], item_df['authorized']))
item_minlot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_maxorder = dict(zip(item_df['item_ref'], item_df['maximum_order']))
categories = cat_df['category'].unique().tolist()
cat_minqty = dict(zip(cat_df['category'], cat_df['minimum_quantity'].astype(int)))
cat_maxqty = dict(zip(cat_df['category'], cat_df['maximum_quantity'].astype(int)))
cat_actfee = dict(zip(cat_df['category'], cat_df['activation_fee_cents'].astype(int)))
benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
for i in items:
    if i not in benefit_per_item:
        benefit_per_item[i] = 0
item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
item_actfee = {}
for (_, row) in item_fee_df.iterrows():
    item_actfee[row['item_ref']] = row['activation_fee_cents']
for i in items:
    if i not in item_actfee:
        item_actfee[i] = 0
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    bundle_bonus[a, b] = int(row['bonus_cents'])
bundles = list(bundle_bonus.keys())
incompat_pairs = set()
for (_, row) in incomp_df.iterrows():
    (a, b) = (row['item_a'], row['item_b'])
    if a != '' and b != '':
        incompat_pairs.add(frozenset([a, b]))
requires_pairs = []
for (_, row) in requires_df.iterrows():
    (i, k) = (row['item_ref'], row['prerequisite_ref'])
    if i != '' and k != '':
        requires_pairs.append((i, k))
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = float(row['amount'])
    unit = row['unit'].strip().lower()
    usage_per_item_resource.setdefault(i, {})
    usage_per_item_resource[i][r] = (amt, unit)
resource_ledger_unit = {}
resource_capacity = {}
for r in cap_df['resource'].unique():
    units = cap_df[cap_df['resource'] == r]['unit']
    ledger_unit = units.mode().iloc[0].strip().lower()
    resource_ledger_unit[r] = ledger_unit
    amt = cap_df[cap_df['resource'] == r]['amount'].sum()
    resource_capacity[r] = amt
usage_per_item_resource_ledger = {}
for i in items:
    usage_per_item_resource_ledger[i] = {}
    for r in resource_capacity:
        if i in usage_per_item_resource and r in usage_per_item_resource[i]:
            (amt, unit) = usage_per_item_resource[i][r]
            ledger_unit = resource_ledger_unit[r]
            key = (unit, ledger_unit)
            if key in UNIT_CONV:
                conv = UNIT_CONV[key]
            else:
                raise ValueError(f'Unknown unit conversion from {unit} to {ledger_unit} for resource {r}')
            usage_per_item_resource_ledger[i][r] = amt * conv
        else:
            usage_per_item_resource_ledger[i][r] = 0.0
m = gp.Model('ReplenishmentNetBenefit')
x_vars = m.addVars(items, vtype=gp.GRB.INTEGER, lb=0, name='')
y_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in items:
    if item_authorized[i] > 0:
        m.addConstr(x_vars[i] <= item_maxorder[i] * y_vars[i], name=f'xmax_{i}')
        m.addConstr(x_vars[i] >= item_minlot[i] * y_vars[i], name=f'xmin_{i}')
        m.addConstr(x_vars[i] <= item_maxorder[i], name=f'xmax2_{i}')
        m.addConstr(x_vars[i] >= 0, name=f'xnonneg_{i}')
    else:
        m.addConstr(x_vars[i] == 0, name=f'xzero_{i}')
        m.addConstr(y_vars[i] == 0, name=f'yzero_{i}')
for g in categories:
    items_in_g = [i for i in items if item_to_cat[i] == g]
    if items_in_g:
        m.addConstr(gp.quicksum((y_vars[i] for i in items_in_g)) <= len(items_in_g) * z_vars[g], name=f'catlink1_{g}')
        for i in items_in_g:
            m.addConstr(y_vars[i] <= z_vars[g], name=f'catlink2_{g}_{i}')
    else:
        m.addConstr(z_vars[g] == 0, name=f'catzero_{g}')
for g in categories:
    items_in_g = [i for i in items if item_to_cat[i] == g]
    if items_in_g:
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) >= cat_minqty[g], name=f'catmin_{g}')
        m.addConstr(gp.quicksum((x_vars[i] for i in items_in_g)) <= cat_maxqty[g], name=f'catmax_{g}')
    else:
        m.addConstr(z_vars[g] == 0, name=f'catzero2_{g}')
for r in resource_capacity:
    m.addConstr(gp.quicksum((usage_per_item_resource_ledger[i][r] * x_vars[i] for i in items)) <= resource_capacity[r], name=f'rescap_{r}')
for pair in incompat_pairs:
    pair = list(pair)
    if len(pair) == 2:
        (i, j) = pair
        if i in y_vars and j in y_vars:
            m.addConstr(y_vars[i] + y_vars[j] <= 1, name=f'incomp_{i}_{j}')
for (i, k) in requires_pairs:
    if i in y_vars and k in y_vars:
        m.addConstr(y_vars[i] <= y_vars[k], name=f'req_{i}_{k}')
for (a, b) in bundles:
    if a in y_vars and b in y_vars:
        m.addConstr(w_vars[a, b] <= y_vars[a], name=f'wleya_{a}_{b}')
        m.addConstr(w_vars[a, b] <= y_vars[b], name=f'wleyb_{a}_{b}')
        m.addConstr(w_vars[a, b] >= y_vars[a] + y_vars[b] - 1, name=f'wgeya_yb_{a}_{b}')
obj_benefit = gp.quicksum((benefit_per_item[i] * x_vars[i] for i in items))
obj_itemfee = gp.quicksum((item_actfee[i] * y_vars[i] for i in items))
obj_catfee = gp.quicksum((cat_actfee[g] * z_vars[g] for g in categories))
obj_bundle = gp.quicksum((bundle_bonus[a, b] * w_vars[a, b] for (a, b) in bundles))
m.setObjective(obj_benefit - obj_itemfee - obj_catfee + obj_bundle, gp.GRB.MAXIMIZE)
m.optimize()