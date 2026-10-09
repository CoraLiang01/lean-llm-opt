import gurobipy as gp
import pandas as pd
import numpy as np
import re

def select_latest_valid(df, table, dealership, as_of_date):
    df = df[(df['table'].str.casefold() == table.casefold()) & (df['dealership_id'].str.casefold() == dealership.casefold())]
    df = df[df['effective_date'] <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values(['dealership_id', 'table', 'record_id', 'revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(['dealership_id', 'table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.casefold() != 'delete']
    return df.reset_index(drop=True)

def select_latest_valid_no_dealer(df, table, as_of_date):
    df = df[df['table'].str.casefold() == table.casefold()]
    df = df[df['effective_date'] <= as_of_date]
    df['revision'] = pd.to_numeric(df['revision'])
    df = df.sort_values(['table', 'record_id', 'revision'], ascending=[True, True, False])
    df = df.drop_duplicates(['table', 'record_id'], keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.casefold() != 'delete']
    return df.reset_index(drop=True)

def select_all_valid(df, table, dealership):
    df = df[(df['table'].str.casefold() == table.casefold()) & (df['dealership_id'].str.casefold() == dealership.casefold())]
    return df.reset_index(drop=True)

def select_all_valid_table(df, table):
    df = df[df['table'].str.casefold() == table.casefold()]
    return df.reset_index(drop=True)

def convert_to_base_unit(amount, unit):
    if unit == 'liter':
        return float(amount) * 1000.0
    elif unit == 'ml':
        return float(amount)
    elif unit == 'hour':
        return float(amount) * 60.0
    elif unit == 'minute':
        return float(amount)
    elif unit == 'kwh':
        return float(amount) * 1000.0
    elif unit == 'wh':
        return float(amount)
    else:
        raise ValueError(f'Unknown unit: {unit}')

def convert_to_usd_cents(amount, currency, fx_dict):
    if currency not in fx_dict:
        raise KeyError(f'Missing FX rate for currency {currency}')
    (num, denom) = fx_dict[currency]
    return float(amount) * float(num) / float(denom)
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_06/export_24.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant32/inputs/batch_01/export_25.csv']
dfs = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths]
as_of_date = '2026-05-07'
dealership = 'OSLO_NEW_CARS'

def collect_table(table, dealership=None, as_of_date=None):
    rows = []
    for df in dfs:
        if 'dealership_id' in df.columns:
            if dealership is not None:
                df2 = df[df['dealership_id'].str.casefold() == dealership.casefold()]
            else:
                df2 = df
        else:
            df2 = df
        if 'table' in df2.columns:
            df2 = df2[df2['table'].str.casefold() == table.casefold()]
        else:
            continue
        if 'effective_date' in df2.columns and as_of_date is not None:
            df2 = df2[df2['effective_date'] <= as_of_date]
        rows.append(df2)
    if rows:
        df_all = pd.concat(rows, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    if not df_all.empty and 'revision' in df_all.columns:
        df_all['revision'] = pd.to_numeric(df_all['revision'])
        keys = []
        if 'dealership_id' in df_all.columns:
            keys.append('dealership_id')
        if 'table' in df_all.columns:
            keys.append('table')
        keys.append('record_id')
        df_all = df_all.sort_values(keys + ['revision'], ascending=[True] * len(keys) + [False])
        df_all = df_all.drop_duplicates(keys, keep='first')
    if not df_all.empty and 'action' in df_all.columns:
        df_all = df_all[df_all['action'].str.casefold() != 'delete']
    return df_all.reset_index(drop=True)
item_df = collect_table('item', dealership, as_of_date)
item_df['authorized'] = pd.to_numeric(item_df['authorized'])
item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'])
item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'])
item_df = item_df[item_df['authorized'] > 0]
item_refs = item_df['item_ref'].unique().tolist()
item_df_all = collect_table('item', dealership, as_of_date)
item_df_all['authorized'] = pd.to_numeric(item_df_all['authorized'])
item_df_all['minimum_lot'] = pd.to_numeric(item_df_all['minimum_lot'])
item_df_all['maximum_order'] = pd.to_numeric(item_df_all['maximum_order'])
all_item_refs = item_df_all['item_ref'].unique().tolist()
cat_df = collect_table('category', dealership, as_of_date)
cat_df['minimum_quantity'] = pd.to_numeric(cat_df['minimum_quantity'])
cat_df['maximum_quantity'] = pd.to_numeric(cat_df['maximum_quantity'])
cat_df['activation_fee_cents'] = pd.to_numeric(cat_df['activation_fee_cents'])
categories = cat_df['category'].unique().tolist()
item_to_cat = dict(zip(item_df_all['item_ref'], item_df_all['category']))
cat_to_items = {g: item_df_all[item_df_all['category'] == g]['item_ref'].tolist() for g in categories}
usage_df = collect_table('usage', dealership, as_of_date)
resources = usage_df['resource'].dropna().unique().tolist()
bundle_df = collect_table('bundle', dealership, as_of_date)
bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'])
bundle_df = bundle_df[bundle_df['item_a'].astype(bool) & bundle_df['item_b'].astype(bool)]
bundles = list(bundle_df[['item_a', 'item_b']].itertuples(index=False, name=None))
incomp_df = collect_table('incompatible', dealership, as_of_date)
incomp_df = incomp_df[incomp_df['item_a'].astype(bool) & incomp_df['item_b'].astype(bool)]
incompat_pairs = list(incomp_df[['item_a', 'item_b']].itertuples(index=False, name=None))
req_df = collect_table('requires', dealership, as_of_date)
req_df = req_df[req_df['item_ref'].astype(bool) & req_df['prerequisite_ref'].astype(bool)]
requires_pairs = list(req_df[['item_ref', 'prerequisite_ref']].itertuples(index=False, name=None))
benefit_df = collect_table('benefit', dealership, as_of_date)
benefit_df['amount'] = pd.to_numeric(benefit_df['amount'])
benefit_df = benefit_df[benefit_df['item_ref'].astype(bool) & benefit_df['component'].astype(bool)]
fx_df = collect_table('fx', dealership, as_of_date)
fx_df = fx_df[fx_df['currency'].astype(bool)]
fx_df['usd_cents_numerator'] = pd.to_numeric(fx_df['usd_cents_numerator'])
fx_df['denominator'] = pd.to_numeric(fx_df['denominator'])
fx_dict = {}
for (_, row) in fx_df.iterrows():
    fx_dict[row['currency']] = (row['usd_cents_numerator'], row['denominator'])
item_fee_df = collect_table('item_fee', dealership, as_of_date)
item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'])
item_fee_dict = dict(zip(item_fee_df['item_ref'], item_fee_df['activation_fee_cents']))
cat_fee_dict = dict(zip(cat_df['category'], cat_df['activation_fee_cents']))
cat_min_qty = dict(zip(cat_df['category'], cat_df['minimum_quantity']))
cat_max_qty = dict(zip(cat_df['category'], cat_df['maximum_quantity']))
item_auth = dict(zip(item_df_all['item_ref'], item_df_all['authorized']))
item_min_lot = dict(zip(item_df_all['item_ref'], item_df_all['minimum_lot']))
item_max_order = dict(zip(item_df_all['item_ref'], item_df_all['maximum_order']))
usage_df['amount'] = pd.to_numeric(usage_df['amount'])
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = convert_to_base_unit(row['amount'], row['unit'])
    usage_per_item_resource.setdefault(i, {})[r] = amt
cap_df = collect_table('capacity_ledger', dealership, as_of_date)
cap_df = cap_df[cap_df['resource'].astype(bool) & cap_df['amount'].astype(bool)]
cap_df['amount'] = pd.to_numeric(cap_df['amount'])
cap_df['amount_base'] = cap_df.apply(lambda row: convert_to_base_unit(row['amount'], row['unit']), axis=1)
resource_capacity = cap_df.groupby('resource')['amount_base'].sum().to_dict()
bundle_bonus = {}
for (_, row) in bundle_df.iterrows():
    bundle_bonus[row['item_a'], row['item_b']] = row['bonus_cents']
incompat_set = set()
for (a, b) in incompat_pairs:
    incompat_set.add(frozenset([a, b]))
requires_list = requires_pairs
item_benefit_usd_cents = {}
for i in all_item_refs:
    df_i = benefit_df[benefit_df['item_ref'] == i]
    total = 0.0
    for (_, row) in df_i.iterrows():
        amt = row['amount']
        curr = row['currency']
        usd_cents = convert_to_usd_cents(amt, curr, fx_dict)
        total += usd_cents
    item_benefit_usd_cents[i] = total
item_fee = {i: item_fee_dict.get(i, 0.0) for i in all_item_refs}
item_min_lot_full = {i: item_min_lot.get(i, 0) for i in all_item_refs}
item_max_order_full = {i: item_max_order.get(i, 0) for i in all_item_refs}
item_auth_full = {i: item_auth.get(i, 0) for i in all_item_refs}
item_cat_full = {i: item_to_cat.get(i, None) for i in all_item_refs}
item_resource_usage = {}
for i in all_item_refs:
    item_resource_usage[i] = {}
    for r in resources:
        item_resource_usage[i][r] = usage_per_item_resource.get(i, {}).get(r, 0.0)
m = gp.Model('Oslo_New_Cars_MaxNetBenefit')
q_vars = m.addVars(all_item_refs, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(all_item_refs, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in all_item_refs:
    if item_auth_full[i] <= 0:
        m.addConstr(q_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q_vars[i] >= int(item_min_lot_full[i]) * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= int(item_max_order_full[i]) * z_vars[i], name=f'maxorder_{i}')
for i in all_item_refs:
    m.addConstr(q_vars[i] <= 100000.0 * z_vars[i], name=f'link_z_{i}')
for g in categories:
    items_in_g = cat_to_items[g]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= int(cat_min_qty[g]), name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= int(cat_max_qty[g]), name=f'cat_max_{g}')
    for i in items_in_g:
        m.addConstr(z_vars[i] <= y_vars[g], name=f'cat_link_{g}_{i}')
    m.addConstr(gp.quicksum((z_vars[i] for i in items_in_g)) >= y_vars[g], name=f'cat_y_lb_{g}')
for r in resources:
    cap = resource_capacity.get(r, 0.0)
    m.addConstr(gp.quicksum((item_resource_usage[i][r] * q_vars[i] for i in all_item_refs)) <= cap, name=f'res_{r}')
for (a, b) in incompat_pairs:
    m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incomp_{a}_{b}')
for (i, k) in requires_list:
    m.addConstr(q_vars[k] >= int(item_min_lot_full[k]) * z_vars[i], name=f'req_{i}_{k}')
for (a, b) in bundles:
    m.addConstr(w_vars[a, b] <= z_vars[a], name=f'bundle1_{a}_{b}')
    m.addConstr(w_vars[a, b] <= z_vars[b], name=f'bundle2_{a}_{b}')
    m.addConstr(w_vars[a, b] >= z_vars[a] + z_vars[b] - 1, name=f'bundle3_{a}_{b}')
benefit_expr = gp.quicksum((item_benefit_usd_cents[i] * q_vars[i] for i in all_item_refs))
item_fee_expr = gp.quicksum((item_fee[i] * z_vars[i] for i in all_item_refs))
cat_fee_expr = gp.quicksum((cat_fee_dict[g] * y_vars[g] for g in categories))
bundle_bonus_expr = gp.quicksum((bundle_bonus[a, b] * w_vars[a, b] for (a, b) in bundles))
net_benefit = benefit_expr - item_fee_expr - cat_fee_expr + bundle_bonus_expr
m.setObjective(net_benefit, gp.GRB.MAXIMIZE)
m.optimize()