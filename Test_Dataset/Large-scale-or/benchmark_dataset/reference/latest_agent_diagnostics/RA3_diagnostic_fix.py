import gurobipy as gp
import pandas as pd
import numpy as np
import re
PLANNING_DATE = '2026-04-16'
DEALERSHIP_ID = 'RIVERSIDE_AUTO'
paths = ['/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_01/export_01.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_02/export_02.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_03/export_03.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_04/export_04.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_05/export_05.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_06/export_06.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_01/export_07.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_02/export_08.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_03/export_09.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_04/export_10.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_05/export_11.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_06/export_12.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_01/export_13.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_02/export_14.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_03/export_15.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_04/export_16.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_05/export_17.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_06/export_18.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_01/export_19.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_02/export_20.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_03/export_21.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_04/export_22.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_05/export_23.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_06/export_24.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA3/inputs/batch_01/export_25.csv']

def parse_date(s):
    return tuple(map(int, s.split('-')))
planning_date_tuple = parse_date(PLANNING_DATE)

def get_surviving(df, key_cols, date_col, revision_col, action_col, dealership_col, table_name):
    df = df[(df[dealership_col].astype(str).str.casefold() == DEALERSHIP_ID.casefold()) & (df['table'].astype(str).str.casefold() == table_name.casefold())]
    df = df[df[date_col].apply(lambda d: parse_date(str(d)) <= planning_date_tuple)]
    df['_sortkey'] = df[date_col].apply(parse_date)
    df = df.sort_values(key_cols + ['_sortkey', revision_col], ascending=[True] * len(key_cols) + [True, False])
    df = df.groupby(key_cols, as_index=False).last()
    if action_col in df.columns:
        df = df[df[action_col].str.casefold() != 'delete']
    df = df.drop(columns=['_sortkey'])
    return df
dfs = [pd.read_csv(p, sep=',') for p in paths]

def concat_tables(table_name, dfs):
    return pd.concat([df for df in dfs if 'table' in df.columns and any(df['table'].astype(str).str.casefold() == table_name.casefold())], ignore_index=True)
item_df = concat_tables('item', dfs)
item_surv = get_surviving(item_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'item')
item_surv = item_surv[item_surv['authorized'].astype(float) == 1]
item_surv = item_surv[item_surv['item_ref'].notnull() & item_surv['category'].notnull()]
item_ids = item_surv['item_ref'].astype(str).tolist()
item_to_cat = dict(zip(item_surv['item_ref'].astype(str), item_surv['category'].astype(str)))
item_minlot = dict(zip(item_surv['item_ref'].astype(str), item_surv['minimum_lot'].astype(float)))
item_maxorder = dict(zip(item_surv['item_ref'].astype(str), item_surv['maximum_order'].astype(float)))
cat_df = concat_tables('category', dfs)
cat_surv = get_surviving(cat_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'category')
cat_surv = cat_surv[cat_surv['category'].notnull()]
cat_ids = cat_surv['category'].astype(str).tolist()
cat_minqty = dict(zip(cat_surv['category'].astype(str), cat_surv['minimum_quantity'].astype(float)))
cat_maxqty = dict(zip(cat_surv['category'].astype(str), cat_surv['maximum_quantity'].astype(float)))
cat_fee = dict(zip(cat_surv['category'].astype(str), cat_surv['activation_fee_cents'].astype(float)))
usage_df = concat_tables('usage', dfs)
usage_df = usage_df[usage_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
usage_df = usage_df[usage_df['item_ref'].notnull() & usage_df['resource'].notnull()]
usage_surv = get_surviving(usage_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'usage')

def convert_usage(row):
    amt = float(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'hour':
        return amt * 60
    elif unit == 'kwh':
        return amt * 1000
    else:
        return amt
usage_dict = {}
for _, row in usage_surv.iterrows():
    i = str(row['item_ref'])
    r = str(row['resource'])
    usage_dict[i, r] = convert_usage(row)
resource_set = set((r for _, r in usage_dict.keys()))
cap_df = concat_tables('capacity_ledger', dfs)
cap_df = cap_df[cap_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
cap_surv = get_surviving(cap_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'capacity_ledger')
cap_surv = cap_surv[cap_surv['resource'].notnull()]

def convert_cap(row):
    amt = float(row['amount'])
    unit = str(row['unit']).strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'hour':
        return amt * 60
    elif unit == 'kwh':
        return amt * 1000
    else:
        return amt
resource_caps = {}
for r in resource_set:
    rows = cap_surv[cap_surv['resource'].astype(str) == r]
    resource_caps[r] = rows.apply(convert_cap, axis=1).sum()
benefit_df = concat_tables('benefit', dfs)
benefit_df = benefit_df[benefit_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
benefit_surv = get_surviving(benefit_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'benefit')
benefit_surv = benefit_surv[benefit_surv['item_ref'].notnull() & benefit_surv['component'].notnull() & benefit_surv['amount'].notnull() & benefit_surv['currency'].notnull()]
fx_df = concat_tables('fx', dfs)
fx_df = fx_df[fx_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
fx_surv = get_surviving(fx_df, ['currency'], 'effective_date', 'revision', 'action', 'dealership_id', 'fx')
fx_rates = {}
for _, row in fx_surv.iterrows():
    c = str(row['currency'])
    num = float(row['usd_cents_numerator'])
    denom = float(row['denominator'])
    fx_rates[c] = (num, denom)
item_benefit = {i: 0.0 for i in item_ids}
for i in item_ids:
    rows = benefit_surv[benefit_surv['item_ref'].astype(str) == i]
    total = 0.0
    for _, row in rows.iterrows():
        amt = float(row['amount'])
        curr = str(row['currency'])
        if curr in fx_rates:
            num, denom = fx_rates[curr]
            if denom != 0:
                amt_usd_cents = amt * num / denom
            else:
                amt_usd_cents = 0.0
        else:
            amt_usd_cents = 0.0
        total += amt_usd_cents
    item_benefit[i] = total
item_fee_df = concat_tables('item_fee', dfs)
item_fee_df = item_fee_df[item_fee_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
item_fee_surv = get_surviving(item_fee_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'item_fee')
item_fee_surv = item_fee_surv[item_fee_surv['item_ref'].notnull() & item_fee_surv['activation_fee_cents'].notnull()]
item_fee_dict = dict(zip(item_fee_surv['item_ref'].astype(str), item_fee_surv['activation_fee_cents'].astype(float)))
bundle_df = concat_tables('bundle', dfs)
bundle_df = bundle_df[bundle_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
bundle_surv = get_surviving(bundle_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'bundle')
bundle_surv = bundle_surv[bundle_surv['item_a'].notnull() & bundle_surv['item_b'].notnull() & bundle_surv['bonus_cents'].notnull()]
bundle_list = []
for _, row in bundle_surv.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_ids and b in item_ids:
        bundle_list.append((a, b, float(row['bonus_cents'])))
bundle_indices = list(range(len(bundle_list)))
incomp_df = concat_tables('incompatible', dfs)
incomp_df = incomp_df[incomp_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
incomp_surv = get_surviving(incomp_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'incompatible')
incomp_surv = incomp_surv[incomp_surv['item_a'].notnull() & incomp_surv['item_b'].notnull()]
incomp_pairs = []
for _, row in incomp_surv.iterrows():
    a = str(row['item_a'])
    b = str(row['item_b'])
    if a in item_ids and b in item_ids:
        incomp_pairs.append((a, b))
req_df = concat_tables('requires', dfs)
req_df = req_df[req_df['dealership_id'].astype(str).str.casefold() == DEALERSHIP_ID.casefold()]
req_surv = get_surviving(req_df, ['record_id'], 'effective_date', 'revision', 'action', 'dealership_id', 'requires')
req_surv = req_surv[req_surv['item_ref'].notnull() & req_surv['prerequisite_ref'].notnull()]
req_pairs = []
for _, row in req_surv.iterrows():
    i = str(row['item_ref'])
    p = str(row['prerequisite_ref'])
    if i in item_ids and p in item_ids:
        req_pairs.append((i, p))
m = gp.Model('RA3_Replenishment')
x = {}
y = {}
for i in item_ids:
    x[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, name=f'x_{i}')
    y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
z = {}
for g in cat_ids:
    z[g] = m.addVar(vtype=gp.GRB.BINARY, name=f'z_{g}')
w = {}
for bidx, (a, b, bonus) in enumerate(bundle_list):
    w[bidx] = m.addVar(vtype=gp.GRB.BINARY, name=f'w_{bidx}')
m.update()
for i in item_ids:
    minlot = item_minlot[i]
    maxorder = item_maxorder[i]
    m.addConstr(x[i] >= minlot * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= maxorder * y[i], name=f'maxorder_{i}')
for r in resource_set:
    m.addConstr(gp.quicksum((usage_dict.get((i, r), 0.0) * x[i] for i in item_ids)) <= resource_caps.get(r, 0.0), name=f'resource_{r}')
for g in cat_ids:
    items_in_g = [i for i in item_ids if item_to_cat[i] == g]
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) >= cat_minqty[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x[i] for i in items_in_g)) <= cat_maxqty[g], name=f'cat_max_{g}')
    for i in items_in_g:
        m.addConstr(z[g] >= y[i], name=f'cat_z_lb_{g}_{i}')
    m.addConstr(z[g] <= gp.quicksum((y[i] for i in items_in_g)), name=f'cat_z_ub_{g}')
for i, j in incomp_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
for i, p in req_pairs:
    m.addConstr(y[i] <= y[p], name=f'req_{i}_{p}')
for bidx, (a, b, bonus) in enumerate(bundle_list):
    m.addConstr(w[bidx] <= y[a], name=f'bundle_w_a_{bidx}')
    m.addConstr(w[bidx] <= y[b], name=f'bundle_w_b_{bidx}')
    m.addConstr(w[bidx] >= y[a] + y[b] - 1, name=f'bundle_w_lb_{bidx}')
for i in item_ids:
    m.addConstr(x[i] <= item_maxorder[i] * y[i], name=f'y_x_link_ub_{i}')
obj = gp.LinExpr()
for i in item_ids:
    obj += item_benefit[i] * x[i]
for i in item_ids:
    fee = item_fee_dict.get(i, 0.0)
    obj -= fee * y[i]
for g in cat_ids:
    obj -= cat_fee[g] * z[g]
for bidx, (a, b, bonus) in enumerate(bundle_list):
    obj += bonus * w[bidx]
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')