def canonical_records(df, *args, **kwargs):
    df = df[(df['portfolio_id'] == 'NYC_REDEVELOPMENT') & (df['effective_date'] <= '2026-06-18')].copy()
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values('revision', ascending=False).drop_duplicates(['table', 'record_id'])
    return df[df['action'].str.upper() != 'DELETE'].copy()
import gurobipy as gp
import pandas as pd
import numpy as np
import re
paths = ['/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_01/export_01.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_02/export_02.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_03/export_03.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_04/export_04.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_05/export_05.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_06/export_06.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_01/export_07.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_02/export_08.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_03/export_09.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_04/export_10.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_05/export_11.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_06/export_12.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_01/export_13.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_02/export_14.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_03/export_15.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_04/export_16.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_05/export_17.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_06/export_18.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_01/export_19.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_02/export_20.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_03/export_21.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_04/export_22.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_05/export_23.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_06/export_24.csv', '/Users/xiaomao/Desktop/Gemini/benchmark_dataset/RA7/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, sep=',') for p in paths]

def get_latest_records(df, table_name, key_cols, date_col='effective_date', portfolio_id='NYC_REDEVELOPMENT', as_of='2026-06-18'):
    df = df[df['table'].str.casefold() == table_name.casefold()]
    df = df[df['portfolio_id'].str.casefold() == portfolio_id.casefold()]
    df = df[df[date_col] <= as_of]
    df = df.copy()
    df['revision'] = pd.to_numeric(df['revision'], errors='coerce')
    df['_sort'] = df['action'].apply(lambda x: 1 if str(x).strip().upper() == 'DELETE' else 0)
    df = df.sort_values(key_cols + ['revision', '_sort'], ascending=[True] * len(key_cols) + [False, False])
    latest = df.groupby(key_cols, as_index=False).first()
    latest = latest[latest['action'].str.strip().str.upper() != 'DELETE']
    latest = latest.drop(columns=['_sort'])
    return latest
item_tables = []
for df in dfs:
    if 'item_ref' in df.columns and 'category' in df.columns and ('authorized' in df.columns) and ('minimum_lot' in df.columns) and ('maximum_order' in df.columns):
        item_tables.append(df)
item_df = pd.concat(item_tables, ignore_index=True)
item_latest = get_latest_records(item_df, 'item', ['record_id'], date_col='effective_date')
item_latest = item_latest[item_latest['authorized'].astype(float) == 1]
item_latest = item_latest[item_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
item_latest['item_ref'] = item_latest['item_ref'].astype(str)
item_latest['category'] = item_latest['category'].astype(str)
item_latest['minimum_lot'] = item_latest['minimum_lot'].astype(float).astype(int)
item_latest['maximum_order'] = item_latest['maximum_order'].astype(float).astype(int)
items = sorted(item_latest['item_ref'].unique())
item2cat = dict(zip(item_latest['item_ref'], item_latest['category']))
item2minlot = dict(zip(item_latest['item_ref'], item_latest['minimum_lot']))
item2maxorder = dict(zip(item_latest['item_ref'], item_latest['maximum_order']))
cat_tables = []
for df in dfs:
    if 'category' in df.columns and 'minimum_quantity' in df.columns and ('maximum_quantity' in df.columns) and ('activation_fee_cents' in df.columns):
        cat_tables.append(df)
cat_df = pd.concat(cat_tables, ignore_index=True)
cat_latest = get_latest_records(cat_df, 'category', ['record_id'], date_col='effective_date')
cat_latest = cat_latest[cat_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
cat_latest['category'] = cat_latest['category'].astype(str)
cat_latest['minimum_quantity'] = cat_latest['minimum_quantity'].astype(float).astype(int)
cat_latest['maximum_quantity'] = cat_latest['maximum_quantity'].astype(float).astype(int)
cat_latest['activation_fee_cents'] = cat_latest['activation_fee_cents'].astype(float).astype(int)
categories = sorted(cat_latest['category'].unique())
cat2minqty = dict(zip(cat_latest['category'], cat_latest['minimum_quantity']))
cat2maxqty = dict(zip(cat_latest['category'], cat_latest['maximum_quantity']))
cat2fee = dict(zip(cat_latest['category'], cat_latest['activation_fee_cents']))
usage_tables = []
for df in dfs:
    if 'resource' in df.columns and 'item_ref' in df.columns and ('amount' in df.columns) and ('unit' in df.columns) and ('table' in df.columns) and df['table'].str.contains('usage', case=False).any():
        usage_tables.append(df)
usage_df = pd.concat(usage_tables, ignore_index=True)
usage_df = canonical_records(usage_df)
usage_df = usage_df[usage_df['portfolio_id'].str.casefold() == 'nyc_redevelopment']
usage_df = usage_df[usage_df['action'].str.strip().str.upper() != 'DELETE']
usage_df['item_ref'] = usage_df['item_ref'].astype(str)
usage_df['resource'] = usage_df['resource'].astype(str)
usage_df['amount'] = usage_df['amount'].astype(float)
usage_df['unit'] = usage_df['unit'].astype(str)

def convert_usage(row):
    amt = row['amount']
    unit = row['unit'].strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'hour':
        return amt * 60
    elif unit == 'kwh':
        return amt * 1000
    else:
        return amt
usage_df['amount_converted'] = usage_df.apply(convert_usage, axis=1)
usage = {}
for _, row in usage_df.iterrows():
    usage[row['item_ref'], row['resource']] = usage.get((row['item_ref'], row['resource']), 0) + row['amount_converted']
resources = sorted(set([r for _, r in usage.keys()]))
cap_tables = []
for df in dfs:
    if 'resource' in df.columns and 'amount' in df.columns and ('unit' in df.columns) and ('table' in df.columns) and df['table'].str.contains('capacity_ledger', case=False).any():
        cap_tables.append(df)
cap_df = pd.concat(cap_tables, ignore_index=True)
cap_df = canonical_records(cap_df)
cap_df = cap_df[cap_df['portfolio_id'].str.casefold() == 'nyc_redevelopment']
cap_df = cap_df[cap_df['action'].str.strip().str.upper() != 'DELETE']
cap_df['resource'] = cap_df['resource'].astype(str)
cap_df['amount'] = cap_df['amount'].astype(float)
cap_df['unit'] = cap_df['unit'].astype(str)

def convert_cap(row):
    amt = row['amount']
    unit = row['unit'].strip().lower()
    if unit == 'liter':
        return amt * 1000
    elif unit == 'hour':
        return amt * 60
    elif unit == 'kwh':
        return amt * 1000
    else:
        return amt
cap_df['amount_converted'] = cap_df.apply(convert_cap, axis=1)
cap_by_resource = cap_df.groupby('resource')['amount_converted'].sum().to_dict()
itemfee_tables = []
for df in dfs:
    if 'item_ref' in df.columns and 'activation_fee_cents' in df.columns and ('table' in df.columns) and df['table'].str.contains('item_fee', case=False).any():
        itemfee_tables.append(df)
itemfee_df = pd.concat(itemfee_tables, ignore_index=True)
itemfee_latest = get_latest_records(itemfee_df, 'item_fee', ['record_id'], date_col='effective_date')
itemfee_latest = itemfee_latest[itemfee_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
itemfee_latest['item_ref'] = itemfee_latest['item_ref'].astype(str)
itemfee_latest['activation_fee_cents'] = itemfee_latest['activation_fee_cents'].astype(float).astype(int)
item2fee = dict(zip(itemfee_latest['item_ref'], itemfee_latest['activation_fee_cents']))
benefit_tables = []
for df in dfs:
    if 'item_ref' in df.columns and 'component' in df.columns and ('amount' in df.columns) and ('currency' in df.columns) and ('table' in df.columns) and df['table'].str.contains('benefit', case=False).any():
        benefit_tables.append(df)
benefit_df = pd.concat(benefit_tables, ignore_index=True)
benefit_df = canonical_records(benefit_df)
benefit_df = benefit_df[benefit_df['portfolio_id'].str.casefold() == 'nyc_redevelopment']
benefit_df = benefit_df[benefit_df['action'].str.strip().str.upper() != 'DELETE']
benefit_df['item_ref'] = benefit_df['item_ref'].astype(str)
benefit_df['currency'] = benefit_df['currency'].astype(str)
benefit_df['amount'] = benefit_df['amount'].astype(float)
fx_tables = []
for df in dfs:
    if 'currency' in df.columns and 'usd_cents_numerator' in df.columns and ('denominator' in df.columns) and ('table' in df.columns) and df['table'].str.contains('fx', case=False).any():
        fx_tables.append(df)
fx_df = pd.concat(fx_tables, ignore_index=True)
fx_df = canonical_records(fx_df)
fx_df = fx_df[fx_df['portfolio_id'].str.casefold() == 'nyc_redevelopment']
fx_df = fx_df[fx_df['action'].str.strip().str.upper() != 'DELETE']
fx_df['currency'] = fx_df['currency'].astype(str)
fx_df['usd_cents_numerator'] = fx_df['usd_cents_numerator'].astype(float)
fx_df['denominator'] = fx_df['denominator'].astype(float)
fx_df = fx_df[fx_df['effective_date'] <= '2026-06-18']
fx_df = fx_df.sort_values(['currency', 'revision'], ascending=[True, False])
fx_latest = fx_df.groupby('currency', as_index=False).first()
fx_rates = {}
for _, row in fx_latest.iterrows():
    fx_rates[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
item2benefit = {}
for item in items:
    df = benefit_df[benefit_df['item_ref'] == item]
    total = 0.0
    for _, row in df.iterrows():
        currency = row['currency']
        amt = row['amount']
        if currency in fx_rates:
            num, denom = fx_rates[currency]
            if denom != 0:
                amt_usd_cents = amt * num / denom
            else:
                amt_usd_cents = 0.0
        else:
            amt_usd_cents = 0.0
        total += amt_usd_cents
    item2benefit[item] = total
bundle_tables = []
for df in dfs:
    if 'item_a' in df.columns and 'item_b' in df.columns and ('bonus_cents' in df.columns) and ('table' in df.columns) and df['table'].str.contains('bundle', case=False).any():
        bundle_tables.append(df)
bundle_df = pd.concat(bundle_tables, ignore_index=True)
bundle_latest = get_latest_records(bundle_df, 'bundle', ['record_id'], date_col='effective_date')
bundle_latest = bundle_latest[bundle_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
bundle_latest['item_a'] = bundle_latest['item_a'].astype(str)
bundle_latest['item_b'] = bundle_latest['item_b'].astype(str)
bundle_latest['bonus_cents'] = bundle_latest['bonus_cents'].astype(float).astype(int)
bundles = []
bundle2bonus = {}
for _, row in bundle_latest.iterrows():
    a, b = (row['item_a'], row['item_b'])
    if a in items and b in items:
        bundles.append((a, b))
        bundle2bonus[a, b] = row['bonus_cents']
incomp_tables = []
for df in dfs:
    if 'item_a' in df.columns and 'item_b' in df.columns and ('table' in df.columns) and df['table'].str.contains('incompatible', case=False).any():
        incomp_tables.append(df)
incomp_df = pd.concat(incomp_tables, ignore_index=True)
incomp_latest = get_latest_records(incomp_df, 'incompatible', ['record_id'], date_col='effective_date')
incomp_latest = incomp_latest[incomp_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
incomp_latest['item_a'] = incomp_latest['item_a'].astype(str)
incomp_latest['item_b'] = incomp_latest['item_b'].astype(str)
incompat_pairs = []
for _, row in incomp_latest.iterrows():
    a, b = (row['item_a'], row['item_b'])
    if a in items and b in items:
        incompat_pairs.append((a, b))
req_tables = []
for df in dfs:
    if 'item_ref' in df.columns and 'prerequisite_ref' in df.columns and ('table' in df.columns) and df['table'].str.contains('requires', case=False).any():
        req_tables.append(df)
req_df = pd.concat(req_tables, ignore_index=True)
req_latest = get_latest_records(req_df, 'requires', ['record_id'], date_col='effective_date')
req_latest = req_latest[req_latest['portfolio_id'].str.casefold() == 'nyc_redevelopment']
req_latest['item_ref'] = req_latest['item_ref'].astype(str)
req_latest['prerequisite_ref'] = req_latest['prerequisite_ref'].astype(str)
prereq_pairs = []
for _, row in req_latest.iterrows():
    i, p = (row['item_ref'], row['prerequisite_ref'])
    if i in items and p in items:
        prereq_pairs.append((i, p))
cat2items = {g: [] for g in categories}
for i in items:
    g = item2cat[i]
    if g in cat2items:
        cat2items[g].append(i)
m = gp.Model('NYC_Redevelopment_Portfolio')
x = {}
y = {}
for i in items:
    x[i] = m.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=item2maxorder[i], name=f'x_{i}')
    y[i] = m.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}')
z = {}
for g in categories:
    z[g] = m.addVar(vtype=gp.GRB.BINARY, name=f'z_{g}')
w = {}
for a, b in bundles:
    w[a, b] = m.addVar(vtype=gp.GRB.BINARY, name=f'w_{a}_{b}')
m.update()
for i in items:
    m.addConstr(x[i] >= item2minlot[i] * y[i], name=f'minlot_{i}')
    m.addConstr(x[i] <= item2maxorder[i] * y[i], name=f'maxorder_{i}')
for r in resources:
    m.addConstr(gp.quicksum((usage.get((i, r), 0.0) * x[i] for i in items)) <= cap_by_resource.get(r, 0.0), name=f'resource_{r}')
for g in categories:
    m.addConstr(gp.quicksum((x[i] for i in cat2items[g])) >= cat2minqty[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((x[i] for i in cat2items[g])) <= cat2maxqty[g], name=f'cat_max_{g}')
for g in categories:
    for i in cat2items[g]:
        m.addConstr(z[g] >= y[i], name=f'catact1_{g}_{i}')
    m.addConstr(z[g] <= gp.quicksum((y[i] for i in cat2items[g])), name=f'catact2_{g}')
for i in items:
    m.addConstr(x[i] <= item2maxorder[i] * y[i], name=f'ylogic1_{i}')
    m.addConstr(x[i] >= y[i], name=f'ylogic2_{i}')
for i, j in incompat_pairs:
    m.addConstr(y[i] + y[j] <= 1, name=f'incomp_{i}_{j}')
for i, p in prereq_pairs:
    m.addConstr(y[i] <= y[p], name=f'prereq_{i}_{p}')
for a, b in bundles:
    m.addConstr(w[a, b] <= y[a], name=f'bundle1_{a}_{b}')
    m.addConstr(w[a, b] <= y[b], name=f'bundle2_{a}_{b}')
    m.addConstr(w[a, b] >= y[a] + y[b] - 1, name=f'bundle3_{a}_{b}')
obj = gp.quicksum((item2benefit.get(i, 0.0) * x[i] for i in items))
obj -= gp.quicksum((item2fee.get(i, 0) * y[i] for i in items))
obj -= gp.quicksum((cat2fee.get(g, 0) * z[g] for g in categories))
obj += gp.quicksum((bundle2bonus.get((a, b), 0) * w[a, b] for a, b in bundles))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal net benefit (USD cents): {int(round(m.objVal))}')
else:
    print(f'No optimal solution found. Status: {m.status}')