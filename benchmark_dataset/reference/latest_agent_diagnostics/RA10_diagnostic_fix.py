import gurobipy as gp
import pandas as pd
import numpy as np
import re
paths = ['/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_01/export_01.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_02/export_02.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_03/export_03.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_04/export_04.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_05/export_05.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_06/export_06.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_01/export_07.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_02/export_08.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_03/export_09.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_04/export_10.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_05/export_11.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_06/export_12.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_01/export_13.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_02/export_14.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_03/export_15.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_04/export_16.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_05/export_17.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_06/export_18.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_01/export_19.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_02/export_20.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_03/export_21.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_04/export_22.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_05/export_23.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_06/export_24.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA10/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, sep=',') for p in paths]

def select_latest_valid(df, *args, **kwargs):
    df = df[(df['store_id'] == 'MARKET_SQUARE') & (df['effective_date'] <= '2026-08-13')].copy()
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values('revision', ascending=False).drop_duplicates(['table', 'record_id'])
    return df[df['action'].str.upper() != 'DELETE'].copy()
planning_date = '2026-08-13'
store_id = 'MARKET_SQUARE'

def find_table_df(table_name):
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == table_name.casefold()).any():
            return df
    return None
item_dfs = []
for t in ['item']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            item_dfs.append(df)
item_rows = pd.concat([select_latest_valid(df, store_id, 'item', planning_date) for df in item_dfs], ignore_index=True)
item_rows = item_rows[item_rows['authorized'] == 1]
item_rows = item_rows.reset_index(drop=True)
item_keys = []
item_info = {}
for idx, row in item_rows.iterrows():
    item_ref = str(row['item_ref'])
    location_id = str(row['location_id'])
    category = str(row['category'])
    min_lot = int(row['minimum_lot'])
    max_order = int(row['maximum_order'])
    item_keys.append((item_ref, location_id))
    item_info[item_ref, location_id] = {'category': category, 'minimum_lot': min_lot, 'maximum_order': max_order, 'item_ref': item_ref, 'location_id': location_id}
cat_dfs = []
for t in ['category']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            cat_dfs.append(df)
cat_rows = pd.concat([select_latest_valid(df, store_id, 'category', planning_date) for df in cat_dfs], ignore_index=True)
cat_rows = cat_rows.reset_index(drop=True)
categories = set(cat_rows['category'].dropna().astype(str))
category_info = {}
for idx, row in cat_rows.iterrows():
    c = str(row['category'])
    category_info[c] = {'minimum_quantity': int(row['minimum_quantity']), 'maximum_quantity': int(row['maximum_quantity']), 'activation_fee_cents': int(row['activation_fee_cents'])}
item_fee_dfs = []
for t in ['item_fee']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            item_fee_dfs.append(df)
item_fee_rows = pd.concat([select_latest_valid(df, store_id, 'item_fee', planning_date) for df in item_fee_dfs], ignore_index=True)
item_fee_rows = item_fee_rows.reset_index(drop=True)
item_fee_map = {}
for idx, row in item_fee_rows.iterrows():
    if pd.isnull(row['item_ref']):
        continue
    item_fee_map[str(row['item_ref'])] = int(row['activation_fee_cents'])
benefit_dfs = []
for t in ['benefit']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            benefit_dfs.append(df)
benefit_rows = pd.concat([select_latest_valid(df, store_id, 'benefit', planning_date) for df in benefit_dfs], ignore_index=True)
benefit_rows = benefit_rows.reset_index(drop=True)
fx_dfs = []
for t in ['fx']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            fx_dfs.append(df)
fx_rows = pd.concat([select_latest_valid(df, store_id, 'fx', planning_date) for df in fx_dfs], ignore_index=True)
fx_rows = fx_rows.reset_index(drop=True)
fx_map = {}
for idx, row in fx_rows.iterrows():
    if pd.isnull(row['currency']):
        continue
    currency = str(row['currency'])
    fx_map[currency] = (float(row['usd_cents_numerator']), float(row['denominator']))
benefit_per_itemref = {}
for item_ref in benefit_rows['item_ref'].dropna().unique():
    rows = benefit_rows[benefit_rows['item_ref'] == item_ref]
    total = 0.0
    for idx, row in rows.iterrows():
        amt = float(row['amount'])
        currency = str(row['currency'])
        if currency not in fx_map:
            continue
        num, denom = fx_map[currency]
        if denom == 0:
            continue
        usd_cents = amt * num / denom
        total += usd_cents
    benefit_per_itemref[str(item_ref)] = total
usage_dfs = []
for t in ['usage']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            usage_dfs.append(df)
usage_rows = pd.concat([select_latest_valid(df, store_id, 'usage', planning_date) for df in usage_dfs], ignore_index=True)
usage_rows = usage_rows.reset_index(drop=True)
usage_map = {}
for idx, row in usage_rows.iterrows():
    if pd.isnull(row['item_ref']) or pd.isnull(row['resource']):
        continue
    item_ref = str(row['item_ref'])
    resource = str(row['resource'])
    amt = float(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        amt_ml = amt * 1000
    elif unit == 'ml':
        amt_ml = amt
    else:
        continue
    usage_map[item_ref, resource] = amt_ml
cap_dfs = []
for t in ['capacity_ledger']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            cap_dfs.append(df)
cap_rows = pd.concat([select_latest_valid(df, store_id, 'capacity_ledger', planning_date) for df in cap_dfs], ignore_index=True)
cap_rows = cap_rows.reset_index(drop=True)
resource_caps = {}
for idx, row in cap_rows.iterrows():
    if pd.isnull(row['resource']):
        continue
    resource = str(row['resource'])
    amt = float(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        amt_ml = amt * 1000
    elif unit == 'ml':
        amt_ml = amt
    else:
        continue
    resource_caps.setdefault(resource, 0.0)
    resource_caps[resource] += amt_ml
incomp_dfs = []
for t in ['incompatible']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            incomp_dfs.append(df)
incomp_rows = pd.concat([select_latest_valid(df, store_id, 'incompatible', planning_date) for df in incomp_dfs], ignore_index=True)
incomp_rows = incomp_rows.reset_index(drop=True)
incompat_pairs = set()
for idx, row in incomp_rows.iterrows():
    if pd.isnull(row['item_a']) or pd.isnull(row['item_b']):
        continue
    a = str(row['item_a'])
    b = str(row['item_b'])
    incompat_pairs.add((a, b))
    incompat_pairs.add((b, a))
req_dfs = []
for t in ['requires']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            req_dfs.append(df)
req_rows = pd.concat([select_latest_valid(df, store_id, 'requires', planning_date) for df in req_dfs], ignore_index=True)
req_rows = req_rows.reset_index(drop=True)
prereq_pairs = []
for idx, row in req_rows.iterrows():
    if pd.isnull(row['item_ref']) or pd.isnull(row['prerequisite_ref']):
        continue
    prereq_pairs.append((str(row['item_ref']), str(row['prerequisite_ref'])))
bundle_dfs = []
for t in ['bundle']:
    for df in dfs:
        if 'table' in df.columns and (df['table'].astype(str).str.casefold() == t.casefold()).any():
            bundle_dfs.append(df)
bundle_rows = pd.concat([select_latest_valid(df, store_id, 'bundle', planning_date) for df in bundle_dfs], ignore_index=True)
bundle_rows = bundle_rows.reset_index(drop=True)
bundle_list = []
for idx, row in bundle_rows.iterrows():
    if pd.isnull(row['item_a']) or pd.isnull(row['item_b']) or pd.isnull(row['bonus_cents']):
        continue
    a = str(row['item_a'])
    b = str(row['item_b'])
    bonus = float(row['bonus_cents'])
    bundle_list.append({'item_a': a, 'item_b': b, 'bonus_cents': bonus})
itemref_to_keys = {}
for k in item_keys:
    itemref_to_keys.setdefault(k[0], []).append(k)
cat_to_itemkeys = {}
for k in item_keys:
    c = item_info[k]['category']
    cat_to_itemkeys.setdefault(c, []).append(k)
m = gp.Model('SupermarketDisplayPlan')
m.Params.OutputFlag = 0
x = {}
y = {}
for k in item_keys:
    min_lot = item_info[k]['minimum_lot']
    max_order = item_info[k]['maximum_order']
    x[k] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=max_order, name=f'x_{k[0]}_{k[1]}')
    y[k] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{k[0]}_{k[1]}')
z = {}
for c in categories:
    z[c] = m.addVar(vtype=gp.GRB.BINARY, name=f'z_{c}')
bvar = {}
for bundle in bundle_list:
    a = bundle['item_a']
    b = bundle['item_b']
    bvar[a, b] = m.addVar(vtype=gp.GRB.BINARY, name=f'b_{a}_{b}')
m.update()
for k in item_keys:
    min_lot = item_info[k]['minimum_lot']
    max_order = item_info[k]['maximum_order']
    m.addConstr(x[k] >= min_lot * y[k], name=f'minlot_{k[0]}_{k[1]}')
    m.addConstr(x[k] <= max_order * y[k], name=f'maxorder_{k[0]}_{k[1]}')
for resource in resource_caps:
    relevant_keys = [k for k in item_keys if (k[0], resource) in usage_map]
    if not relevant_keys:
        continue
    m.addConstr(gp.quicksum((usage_map[k[0], resource] * x[k] for k in relevant_keys)) <= resource_caps[resource], name=f'cap_{resource}')
for c in categories:
    itemks = cat_to_itemkeys.get(c, [])
    if not itemks:
        m.addConstr(z[c] == 0, name=f'cat_empty_{c}')
        continue
    minq = category_info[c]['minimum_quantity']
    maxq = category_info[c]['maximum_quantity']
    m.addConstr(gp.quicksum((x[k] for k in itemks)) >= minq * z[c], name=f'cat_min_{c}')
    m.addConstr(gp.quicksum((x[k] for k in itemks)) <= maxq * z[c], name=f'cat_max_{c}')
    for k in itemks:
        m.addConstr(y[k] <= z[c], name=f'cat_link_{k[0]}_{k[1]}')
for a, b in incompat_pairs:
    a_keys = itemref_to_keys.get(a, [])
    b_keys = itemref_to_keys.get(b, [])
    for ka in a_keys:
        for kb in b_keys:
            if ka == kb:
                continue
            m.addConstr(y[ka] + y[kb] <= 1, name=f'incomp_{ka[0]}_{ka[1]}_{kb[0]}_{kb[1]}')
for item_ref, prereq_ref in prereq_pairs:
    item_keys_ = itemref_to_keys.get(item_ref, [])
    prereq_keys_ = itemref_to_keys.get(prereq_ref, [])
    for ki in item_keys_:
        for kp in prereq_keys_:
            m.addConstr(y[ki] <= y[kp], name=f'prereq_{ki[0]}_{ki[1]}_{kp[0]}_{kp[1]}')
for bundle in bundle_list:
    a = bundle['item_a']
    b = bundle['item_b']
    a_keys = itemref_to_keys.get(a, [])
    b_keys = itemref_to_keys.get(b, [])
    for ka in a_keys:
        for kb in b_keys:
            m.addConstr(bvar[a, b] <= y[ka], name=f'bundle_le_a_{ka[0]}_{ka[1]}_{kb[0]}_{kb[1]}')
            m.addConstr(bvar[a, b] <= y[kb], name=f'bundle_le_b_{ka[0]}_{ka[1]}_{kb[0]}_{kb[1]}')
            m.addConstr(bvar[a, b] >= y[ka] + y[kb] - 1, name=f'bundle_ge_sum_{ka[0]}_{ka[1]}_{kb[0]}_{kb[1]}')
obj_benefit = gp.LinExpr()
for k in item_keys:
    per_unit = benefit_per_itemref.get(k[0], 0.0)
    obj_benefit += per_unit * x[k]
obj_item_fee = gp.LinExpr()
for k in item_keys:
    fee = item_fee_map.get(k[0], 0.0)
    obj_item_fee += fee * y[k]
obj_cat_fee = gp.LinExpr()
for c in categories:
    fee = category_info[c]['activation_fee_cents']
    obj_cat_fee += fee * z[c]
obj_bundle_bonus = gp.LinExpr()
for bundle in bundle_list:
    obj_bundle_bonus += bundle['bonus_cents'] * bvar[bundle['item_a'], bundle['item_b']]
m.setObjective(obj_benefit - obj_item_fee - obj_cat_fee + obj_bundle_bonus, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'{int(round(m.objVal))}')
else:
    print('No optimal solution found.')