import gurobipy as gp
import pandas as pd
import numpy as np
import re

def filter_table(df, asof_date, tenant='NORTH', record_keys=None):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    if record_keys is not None:
        df = df[df['record_id'].isin(record_keys)]
    return df

def filter_itemref_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_category_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_resource_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_pair_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_requires_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def filter_identity_table(df, asof_date, tenant='NORTH'):
    df = df[df['tenant'].str.strip().str.casefold() == tenant.casefold()]
    df = df[df['effective_date'] <= asof_date]
    idx_cols = ['table', 'tenant', 'record_id']
    df['revision'] = pd.to_numeric(df['revision'], errors='raise')
    df = df.sort_values(idx_cols + ['revision'], ascending=[True, True, True, False])
    df = df.drop_duplicates(idx_cols, keep='first')
    if 'action' in df.columns:
        df = df[df['action'].str.strip().str.casefold() != 'delete']
    return df

def convert_resource_amount(amount, unit, resource):
    if resource == 'space':
        if unit.strip().casefold() == 'ml':
            return float(amount)
        elif unit.strip().casefold() == 'liter':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for space: {unit}')
    elif resource == 'labor':
        if unit.strip().casefold() == 'minute':
            return float(amount)
        elif unit.strip().casefold() == 'hour':
            return float(amount) * 60.0
        else:
            raise ValueError(f'Unknown unit for labor: {unit}')
    elif resource == 'power':
        if unit.strip().casefold() == 'wh':
            return float(amount)
        elif unit.strip().casefold() == 'kwh':
            return float(amount) * 1000.0
        else:
            raise ValueError(f'Unknown unit for power: {unit}')
    else:
        raise ValueError(f'Unknown resource: {resource}')
asof_date = '2026-03-12'
tenant = 'NORTH'
paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_01.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_02.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_03.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_04.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_05.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_06.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_07.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_08.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_09.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_10.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_11.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_12.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_13.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_14.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_15.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_16.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_17.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_18.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_01/export_19.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_02/export_20.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_03/export_21.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_04/export_22.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_05/export_23.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant30/inputs/batch_06/export_24.csv']
dfs = {}
for path in paths:
    key = re.sub('^.*/', '', path)
    dfs[key] = pd.read_csv(path, dtype=str, keep_default_na=False)
identity_dfs = []
for k in dfs:
    if 'identity' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'identity').any():
        identity_dfs.append(dfs[k])
identity_df = pd.concat(identity_dfs, ignore_index=True)
identity_df = filter_identity_table(identity_df, asof_date, tenant)
identity_df = identity_df[identity_df['kind'].str.strip().str.casefold() == 'item']
ref_to_itemid = dict(zip(identity_df['ref'], identity_df['entity_id']))
item_dfs = []
for k in dfs:
    if 'item_ref' in dfs[k].columns and 'category' in dfs[k].columns and ('authorized' in dfs[k].columns):
        item_dfs.append(dfs[k])
item_df = pd.concat(item_dfs, ignore_index=True)
item_df = filter_itemref_table(item_df, asof_date, tenant)
item_df = item_df[item_df['item_ref'].notnull() & item_df['category'].notnull()]
item_df['authorized'] = pd.to_numeric(item_df['authorized'], errors='raise')
item_df['minimum_lot'] = pd.to_numeric(item_df['minimum_lot'], errors='raise')
item_df['maximum_order'] = pd.to_numeric(item_df['maximum_order'], errors='raise')
item_ref_set = set(item_df['item_ref'])
category_dfs = []
for k in dfs:
    if 'category' in dfs[k].columns and 'activation_fee_cents' in dfs[k].columns:
        category_dfs.append(dfs[k])
category_df = pd.concat(category_dfs, ignore_index=True)
category_df = filter_category_table(category_df, asof_date, tenant)
category_df = category_df[category_df['category'].notnull()]
category_df['minimum_quantity'] = pd.to_numeric(category_df['minimum_quantity'], errors='coerce').fillna(0).astype(int)
category_df['maximum_quantity'] = pd.to_numeric(category_df['maximum_quantity'], errors='coerce').fillna(0).astype(int)
category_df['activation_fee_cents'] = pd.to_numeric(category_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
category_set = set(category_df['category'])
benefit_dfs = []
for k in dfs:
    if 'benefit' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'benefit').any():
        benefit_dfs.append(dfs[k])
    elif 'amount_cents' in dfs[k].columns and 'item_ref' in dfs[k].columns:
        benefit_dfs.append(dfs[k])
benefit_df = pd.concat(benefit_dfs, ignore_index=True)
benefit_df = filter_itemref_table(benefit_df, asof_date, tenant)
benefit_df = benefit_df[benefit_df['item_ref'].notnull()]
benefit_df['amount_cents'] = pd.to_numeric(benefit_df['amount_cents'], errors='raise')
benefit_per_item = benefit_df.groupby('item_ref')['amount_cents'].sum().to_dict()
item_fee_dfs = []
for k in dfs:
    if 'item_fee' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'item_fee').any():
        item_fee_dfs.append(dfs[k])
    elif 'activation_fee_cents' in dfs[k].columns and 'item_ref' in dfs[k].columns:
        item_fee_dfs.append(dfs[k])
item_fee_df = pd.concat(item_fee_dfs, ignore_index=True)
item_fee_df = filter_itemref_table(item_fee_df, asof_date, tenant)
item_fee_df = item_fee_df[item_fee_df['item_ref'].notnull()]
item_fee_df['activation_fee_cents'] = pd.to_numeric(item_fee_df['activation_fee_cents'], errors='coerce').fillna(0).astype(int)
item_fee_per_item = item_fee_df.groupby('item_ref')['activation_fee_cents'].max().to_dict()
usage_dfs = []
for k in dfs:
    if 'usage' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'usage').any():
        usage_dfs.append(dfs[k])
    elif 'item_ref' in dfs[k].columns and 'resource' in dfs[k].columns and ('amount' in dfs[k].columns) and ('unit' in dfs[k].columns):
        usage_dfs.append(dfs[k])
usage_df = pd.concat(usage_dfs, ignore_index=True)
usage_df = filter_itemref_table(usage_df, asof_date, tenant)
usage_df = usage_df[usage_df['item_ref'].notnull() & usage_df['resource'].notnull() & usage_df['amount'].notnull() & usage_df['unit'].notnull()]
usage_df['amount'] = pd.to_numeric(usage_df['amount'], errors='raise')
usage_per_item_resource = {}
for (_, row) in usage_df.iterrows():
    item = row['item_ref']
    resource = row['resource'].strip()
    amount = row['amount']
    unit = row['unit']
    amount_canonical = convert_resource_amount(amount, unit, resource)
    usage_per_item_resource[item, resource] = amount_canonical
capacity_dfs = []
for k in dfs:
    if 'capacity_ledger' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'capacity_ledger').any():
        capacity_dfs.append(dfs[k])
    elif 'resource' in dfs[k].columns and 'amount' in dfs[k].columns and ('unit' in dfs[k].columns):
        capacity_dfs.append(dfs[k])
capacity_df = pd.concat(capacity_dfs, ignore_index=True)
capacity_df = filter_resource_table(capacity_df, asof_date, tenant)
capacity_df = capacity_df[capacity_df['resource'].notnull() & capacity_df['amount'].notnull() & capacity_df['unit'].notnull()]
capacity_df['amount'] = pd.to_numeric(capacity_df['amount'], errors='raise')
capacity_df = capacity_df[capacity_df['entry'].isin(['opening', 'reservation'])]
resource_capacity = {}
for resource in capacity_df['resource'].unique():
    sub = capacity_df[capacity_df['resource'] == resource]
    total = 0.0
    for (_, row) in sub.iterrows():
        amt = row['amount']
        unit = row['unit']
        amt_canonical = convert_resource_amount(amt, unit, resource)
        total += amt_canonical
    resource_capacity[resource] = total
incompatible_dfs = []
for k in dfs:
    if 'incompatible' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'incompatible').any():
        incompatible_dfs.append(dfs[k])
    elif 'item_a' in dfs[k].columns and 'item_b' in dfs[k].columns:
        incompatible_dfs.append(dfs[k])
incompatible_df = pd.concat(incompatible_dfs, ignore_index=True)
incompatible_df = filter_pair_table(incompatible_df, asof_date, tenant)
incompatible_df = incompatible_df[incompatible_df['item_a'].notnull() & incompatible_df['item_b'].notnull()]
incompatible_pairs = set()
for (_, row) in incompatible_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a != '' and b != '':
        incompatible_pairs.add(tuple(sorted((a, b))))
requires_dfs = []
for k in dfs:
    if 'requires' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'requires').any():
        requires_dfs.append(dfs[k])
    elif 'item_ref' in dfs[k].columns and 'prerequisite_ref' in dfs[k].columns:
        requires_dfs.append(dfs[k])
requires_df = pd.concat(requires_dfs, ignore_index=True)
requires_df = filter_requires_table(requires_df, asof_date, tenant)
requires_df = requires_df[requires_df['item_ref'].notnull() & requires_df['prerequisite_ref'].notnull()]
requires_pairs = set()
for (_, row) in requires_df.iterrows():
    i = row['item_ref']
    k = row['prerequisite_ref']
    if i != '' and k != '':
        requires_pairs.add((i, k))
bundle_dfs = []
for k in dfs:
    if 'bundle' in dfs[k].columns.get('table', '') or (dfs[k]['table'].str.strip().str.casefold() == 'bundle').any():
        bundle_dfs.append(dfs[k])
    elif 'item_a' in dfs[k].columns and 'item_b' in dfs[k].columns and ('bonus_cents' in dfs[k].columns):
        bundle_dfs.append(dfs[k])
bundle_df = pd.concat(bundle_dfs, ignore_index=True)
bundle_df = filter_pair_table(bundle_df, asof_date, tenant)
bundle_df = bundle_df[bundle_df['item_a'].notnull() & bundle_df['item_b'].notnull()]
bundle_df['bonus_cents'] = pd.to_numeric(bundle_df['bonus_cents'], errors='coerce').fillna(0).astype(int)
bundle_list = []
bundle_bonus = {}
for (idx, row) in bundle_df.iterrows():
    a = row['item_a']
    b = row['item_b']
    key = tuple(sorted((a, b)))
    bundle_list.append(key)
    bundle_bonus[key] = row['bonus_cents']
item_list = sorted(item_df['item_ref'])
item_set = set(item_list)
category_list = sorted(category_df['category'])
category_set = set(category_list)
item_to_category = dict(zip(item_df['item_ref'], item_df['category']))
category_to_items = {g: [] for g in category_list}
for i in item_list:
    g = item_to_category[i]
    if g in category_to_items:
        category_to_items[g].append(i)
item_authorized = dict(zip(item_df['item_ref'], item_df['authorized']))
item_min_lot = dict(zip(item_df['item_ref'], item_df['minimum_lot']))
item_max_order = dict(zip(item_df['item_ref'], item_df['maximum_order']))
category_min_qty = dict(zip(category_df['category'], category_df['minimum_quantity']))
category_max_qty = dict(zip(category_df['category'], category_df['maximum_quantity']))
category_fee = dict(zip(category_df['category'], category_df['activation_fee_cents']))
item_fee = {i: item_fee_per_item.get(i, 0) for i in item_list}
item_benefit = {i: benefit_per_item.get(i, 0) for i in item_list}
resources = set()
for (i, r) in usage_per_item_resource:
    if i in item_set:
        resources.add(r)
resources = sorted(resources)
resource_cap = {r: resource_capacity.get(r, 0.0) for r in resources}
item_resource_usage = {}
for i in item_list:
    for r in resources:
        item_resource_usage[i, r] = usage_per_item_resource.get((i, r), 0.0)
m = gp.Model('vehicle_dealer_replenishment')
q_vars = m.addVars(item_list, vtype=gp.GRB.INTEGER, lb=0, name='')
z_vars = m.addVars(item_list, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(category_list, vtype=gp.GRB.BINARY, name='')
w_vars = m.addVars(bundle_list, vtype=gp.GRB.BINARY, name='')
for i in item_list:
    if item_authorized[i] <= 0:
        m.addConstr(q_vars[i] == 0, name=f'unauth_{i}')
        m.addConstr(z_vars[i] == 0, name=f'unauth_z_{i}')
    else:
        m.addConstr(q_vars[i] >= item_min_lot[i] * z_vars[i], name=f'minlot_{i}')
        m.addConstr(q_vars[i] <= item_max_order[i] * z_vars[i], name=f'maxorder_{i}')
for g in category_list:
    items_in_g = category_to_items[g]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) >= category_min_qty[g], name=f'cat_min_{g}')
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_g)) <= category_max_qty[g], name=f'cat_max_{g}')
for g in category_list:
    items_in_g = category_to_items[g]
    for i in items_in_g:
        m.addConstr(z_vars[i] <= y_vars[g], name=f'catflag1_{g}_{i}')
    m.addConstr(y_vars[g] <= gp.quicksum((z_vars[i] for i in items_in_g)), name=f'catflag2_{g}')
for r in resources:
    m.addConstr(gp.quicksum((item_resource_usage[i, r] * q_vars[i] for i in item_list)) <= resource_cap[r], name=f'rescap_{r}')
for (a, b) in incompatible_pairs:
    if a in item_set and b in item_set:
        m.addConstr(z_vars[a] + z_vars[b] <= 1, name=f'incomp_{a}_{b}')
for (i, k) in requires_pairs:
    if i in item_set and k in item_set:
        m.addConstr(z_vars[i] <= z_vars[k], name=f'reqflag_{i}_{k}')
        m.addConstr(q_vars[k] >= item_min_lot[k] * z_vars[i], name=f'reqqty_{i}_{k}')
for b in bundle_list:
    (a, b_) = b
    if a in item_set and b_ in item_set:
        m.addConstr(w_vars[b] <= z_vars[a], name=f'bundle1_{a}_{b_}')
        m.addConstr(w_vars[b] <= z_vars[b_], name=f'bundle2_{a}_{b_}')
        m.addConstr(w_vars[b] >= z_vars[a] + z_vars[b_] - 1, name=f'bundle3_{a}_{b_}')
    else:
        m.addConstr(w_vars[b] == 0, name=f'bundle0_{a}_{b_}')
obj = gp.quicksum((item_benefit[i] * q_vars[i] for i in item_list))
obj -= gp.quicksum((item_fee[i] * z_vars[i] for i in item_list))
obj -= gp.quicksum((category_fee[g] * y_vars[g] for g in category_list))
obj += gp.quicksum((bundle_bonus[b] * w_vars[b] for b in bundle_list))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()