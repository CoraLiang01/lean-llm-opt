import gurobipy as gp
import pandas as pd
import numpy as np
import re
paths = ['/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_01/export_01.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_02/export_02.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_03/export_03.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_04/export_04.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_05/export_05.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_06/export_06.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_01/export_07.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_02/export_08.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_03/export_09.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_04/export_10.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_05/export_11.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_06/export_12.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_01/export_13.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_02/export_14.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_03/export_15.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_04/export_16.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_05/export_17.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_06/export_18.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_01/export_19.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_02/export_20.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_03/export_21.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_04/export_22.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_05/export_23.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_06/export_24.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA12/inputs/batch_01/export_25.csv']

def get_table_df(table_name, all_dfs):
    dfs = []
    for df in all_dfs:
        if 'table' in df.columns:
            if df['table'].str.casefold().str.strip().eq(table_name.casefold().strip()).any():
                dfs.append(df[df['table'].str.casefold().str.strip() == table_name.casefold().strip()])
    return dfs

def select_latest_records(df, *args, **kwargs):
    df = df[(df['store_id'] == 'CENTRAL_FRESH') & (df['effective_date'] <= '2026-07-09')].copy()
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values('revision', ascending=False).drop_duplicates(['table', 'record_id'])
    return df[df['action'].str.upper() != 'DELETE'].copy()

def select_latest_records_no_date(df, key_cols, revision_col, action_col, store_id_col='store_id', store_id_val='CENTRAL_FRESH'):
    df = df[df[store_id_col] == store_id_val]
    df = df.sort_values(key_cols + [revision_col], ascending=[True] * len(key_cols) + [False])
    df = df.drop_duplicates(subset=key_cols, keep='first')
    df = df[df[action_col].str.casefold() != 'delete']
    return df

def select_store_records(df, *args, **kwargs):
    df = df[(df['store_id'] == 'CENTRAL_FRESH') & (df['effective_date'] <= '2026-07-09')].copy()
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values('revision', ascending=False).drop_duplicates(['table', 'record_id'])
    return df[df['action'].str.upper() != 'DELETE'].copy()

def convert_unit(amount, unit):
    if unit == 'liter':
        return amount * 1000.0
    elif unit == 'hour':
        return amount * 60.0
    elif unit == 'kwh':
        return amount * 1000.0
    else:
        return amount

def get_fx_dict(fx_df):
    fx = {}
    for _, row in fx_df.iterrows():
        if pd.isnull(row['currency']):
            continue
        fx[row['currency']] = (float(row['usd_cents_numerator']), float(row['denominator']))
    return fx
all_dfs = [pd.read_csv(p, sep=',') for p in paths]
planning_date = '2026-07-09'
store_id = 'CENTRAL_FRESH'
item_dfs = []
for df in all_dfs:
    if 'item_ref' in df.columns and 'category' in df.columns and ('authorized' in df.columns) and ('minimum_lot' in df.columns) and ('maximum_order' in df.columns):
        item_dfs.append(df)
item_df = pd.concat(item_dfs, ignore_index=True)
item_df = select_latest_records(item_df, ['item_ref'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
item_df = item_df.dropna(subset=['item_ref', 'category', 'authorized', 'minimum_lot', 'maximum_order'])
item_df['authorized'] = item_df['authorized'].astype(float)
item_df['minimum_lot'] = item_df['minimum_lot'].astype(float)
item_df['maximum_order'] = item_df['maximum_order'].astype(float)
item_df['category'] = item_df['category'].astype(str)
item_df['item_ref'] = item_df['item_ref'].astype(str)
item_df = item_df[item_df['authorized'] == 1.0]
category_dfs = []
for df in all_dfs:
    if 'category' in df.columns and 'minimum_quantity' in df.columns and ('maximum_quantity' in df.columns) and ('activation_fee_cents' in df.columns):
        category_dfs.append(df)
category_df = pd.concat(category_dfs, ignore_index=True)
category_df = select_latest_records(category_df, ['category'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
category_df = category_df.dropna(subset=['category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'])
category_df['category'] = category_df['category'].astype(str)
category_df['minimum_quantity'] = category_df['minimum_quantity'].astype(float)
category_df['maximum_quantity'] = category_df['maximum_quantity'].astype(float)
category_df['activation_fee_cents'] = category_df['activation_fee_cents'].astype(float)
benefit_dfs = []
for df in all_dfs:
    if 'component' in df.columns and 'amount' in df.columns and ('currency' in df.columns) and ('item_ref' in df.columns):
        benefit_dfs.append(df)
benefit_df = pd.concat(benefit_dfs, ignore_index=True)
benefit_df = select_latest_records(benefit_df, ['record_id'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
benefit_df = benefit_df.dropna(subset=['item_ref', 'component', 'amount', 'currency'])
benefit_df['item_ref'] = benefit_df['item_ref'].astype(str)
benefit_df['amount'] = benefit_df['amount'].astype(float)
benefit_df['currency'] = benefit_df['currency'].astype(str)
fx_dfs = []
for df in all_dfs:
    if 'currency' in df.columns and 'usd_cents_numerator' in df.columns and ('denominator' in df.columns):
        fx_dfs.append(df)
fx_df = pd.concat(fx_dfs, ignore_index=True)
fx_df = select_latest_records(fx_df, ['currency'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
fx_df = fx_df.dropna(subset=['currency', 'usd_cents_numerator', 'denominator'])
fx_dict = get_fx_dict(fx_df)
item_fee_dfs = []
for df in all_dfs:
    if 'item_ref' in df.columns and 'activation_fee_cents' in df.columns and ('table' in df.columns) and df['table'].str.contains('item_fee').any():
        item_fee_dfs.append(df)
item_fee_df = pd.concat(item_fee_dfs, ignore_index=True)
item_fee_df = select_latest_records(item_fee_df, ['item_ref'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
item_fee_df = item_fee_df.dropna(subset=['item_ref', 'activation_fee_cents'])
item_fee_df['item_ref'] = item_fee_df['item_ref'].astype(str)
item_fee_df['activation_fee_cents'] = item_fee_df['activation_fee_cents'].astype(float)
item_fee_map = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
usage_dfs = []
for df in all_dfs:
    if 'item_ref' in df.columns and 'resource' in df.columns and ('amount' in df.columns) and ('unit' in df.columns) and ('table' in df.columns) and df['table'].str.contains('usage').any():
        usage_dfs.append(df)
usage_df = pd.concat(usage_dfs, ignore_index=True)
usage_df = select_store_records(usage_df, store_id_col='store_id', store_id_val=store_id)
usage_df = usage_df.dropna(subset=['item_ref', 'resource', 'amount', 'unit'])
usage_df['item_ref'] = usage_df['item_ref'].astype(str)
usage_df['resource'] = usage_df['resource'].astype(str)
usage_df['amount'] = usage_df.apply(lambda row: convert_unit(row['amount'], row['unit']), axis=1)
capacity_dfs = []
for df in all_dfs:
    if 'resource' in df.columns and 'amount' in df.columns and ('unit' in df.columns) and ('table' in df.columns) and df['table'].str.contains('capacity_ledger').any():
        capacity_dfs.append(df)
capacity_df = pd.concat(capacity_dfs, ignore_index=True)
capacity_df = select_latest_records(capacity_df, ['record_id'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
capacity_df = capacity_df.dropna(subset=['resource', 'amount', 'unit'])
capacity_df['resource'] = capacity_df['resource'].astype(str)
capacity_df['amount'] = capacity_df.apply(lambda row: convert_unit(row['amount'], row['unit']), axis=1)
resource_capacity = capacity_df.groupby('resource')['amount'].sum().to_dict()
incompat_dfs = []
for df in all_dfs:
    if 'item_a' in df.columns and 'item_b' in df.columns and ('table' in df.columns) and df['table'].str.contains('incompatible').any():
        incompat_dfs.append(df)
incompat_df = pd.concat(incompat_dfs, ignore_index=True)
incompat_df = select_latest_records(incompat_df, ['record_id'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
incompat_df = incompat_df.dropna(subset=['item_a', 'item_b'])
incompat_pairs = set()
for _, row in incompat_df.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a != b:
        incompat_pairs.add(tuple(sorted((a, b))))
requires_dfs = []
for df in all_dfs:
    if 'item_ref' in df.columns and 'prerequisite_ref' in df.columns and ('table' in df.columns) and df['table'].str.contains('requires').any():
        requires_dfs.append(df)
requires_df = pd.concat(requires_dfs, ignore_index=True)
requires_df = select_latest_records(requires_df, ['record_id'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
requires_df = requires_df.dropna(subset=['item_ref', 'prerequisite_ref'])
requires_pairs = set()
for _, row in requires_df.iterrows():
    i = str(row['item_ref'])
    p = str(row['prerequisite_ref'])
    if i != p:
        requires_pairs.add((i, p))
bundle_dfs = []
for df in all_dfs:
    if 'item_a' in df.columns and 'item_b' in df.columns and ('bonus_cents' in df.columns) and ('table' in df.columns) and df['table'].str.contains('bundle').any():
        bundle_dfs.append(df)
bundle_df = pd.concat(bundle_dfs, ignore_index=True)
bundle_df = select_latest_records(bundle_df, ['record_id'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
bundle_df = bundle_df.dropna(subset=['item_a', 'item_b', 'bonus_cents'])
bundle_df['item_a'] = bundle_df['item_a'].astype(str)
bundle_df['item_b'] = bundle_df['item_b'].astype(str)
bundle_df['bonus_cents'] = bundle_df['bonus_cents'].astype(float)
bundle_tuples = []
for _, row in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    bonus = row['bonus_cents']
    if a != b:
        bundle_tuples.append((a, b, bonus))
identity_dfs = []
for df in all_dfs:
    if 'ref' in df.columns and 'entity_id' in df.columns and ('kind' in df.columns) and ('table' in df.columns) and df['table'].str.contains('identity').any():
        identity_dfs.append(df)
identity_df = pd.concat(identity_dfs, ignore_index=True)
identity_df = select_latest_records(identity_df, ['ref'], 'effective_date', 'revision', 'action', store_id_col='store_id', store_id_val=store_id)
identity_df = identity_df.dropna(subset=['ref', 'entity_id'])
identity_map = dict(zip(identity_df['ref'].astype(str), identity_df['entity_id'].astype(str)))
item_list = sorted(item_df['item_ref'].unique())
category_list = sorted(category_df['category'].unique())
resource_list = sorted(resource_capacity.keys())
item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
category_to_items = {g: [] for g in category_list}
for i, g in item_to_category.items():
    if g in category_to_items:
        category_to_items[g].append(i)
item_benefit = {i: 0.0 for i in item_list}
for i in item_list:
    rows = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for _, row in rows.iterrows():
        amt = float(row['amount'])
        curr = row['currency']
        if curr in fx_dict:
            num, denom = fx_dict[curr]
            amt_usd_cents = amt * num / denom
            total += amt_usd_cents
        else:
            continue
    item_benefit[i] = total
item_fee = {i: item_fee_map[i] if i in item_fee_map else 0.0 for i in item_list}
category_min = dict(zip(category_df['category'], category_df['minimum_quantity']))
category_max = dict(zip(category_df['category'], category_df['maximum_quantity']))
category_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
item_min = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_max = dict(zip(item_df['item_ref'], item_df['maximum_order']))
usage_map = {}
for _, row in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = float(row['amount'])
    usage_map[i, r] = amt

def get_usage(i, r):
    return usage_map.get((i, r), 0.0)
m = gp.Model('ProduceReplenishment')
x = {}
y = {}
for i in item_list:
    x[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, name=f'x_{i}')
    y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
z = {}
for g in category_list:
    z[g] = m.addVar(vtype=gp.GRB.BINARY, name=f'z_{g}')
for i in item_list:
    m.addConstr(x[i] >= item_min[i] * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= item_max[i] * y[i], name=f'maxorder_{i}')
for g in category_list:
    items_in_g = category_to_items[g]
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= category_min[g], name=f'catmin_{g}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= category_max[g], name=f'catmax_{g}')
for g in category_list:
    items_in_g = category_to_items[g]
    for i in items_in_g:
        m.addConstr(z[g] >= y[i], name=f'catfee_logic_{g}_{i}')
for r in resource_list:
    m.addConstr(gp.quicksum((get_usage(i, r) * x[i] for i in item_list)) <= resource_capacity[r], name=f'res_{r}')
for a, b in incompat_pairs:
    if a in y and b in y:
        m.addConstr(y[a] + y[b] <= 1, name=f'incompat_{a}_{b}')
for i, p in requires_pairs:
    if i in y and p in y:
        m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
obj = gp.quicksum((item_benefit[i] * x[i] for i in item_list))
obj -= gp.quicksum((item_fee[i] * y[i] for i in item_list))
obj -= gp.quicksum((category_fee[g] * z[g] for g in category_list))
for a, b, bonus in bundle_tuples:
    if a in y and b in y:
        obj += bonus * y[a] * y[b]
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')