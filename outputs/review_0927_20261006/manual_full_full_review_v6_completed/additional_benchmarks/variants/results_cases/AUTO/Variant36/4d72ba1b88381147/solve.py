import gurobipy as gp
import pandas as pd
import numpy as np

def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
df_bundle = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_01.csv')
df_bundle['bonus_cents'] = df_bundle['bonus_cents'].astype(int)
df_cap = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_02/export_02.csv')
df_cap['amount'] = df_cap['amount'].astype(int)
df_cat = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_03.csv')
df_cat['minimum_quantity'] = df_cat['minimum_quantity'].astype(int)
df_cat['maximum_quantity'] = df_cat['maximum_quantity'].astype(int)
df_cat['activation_fee_cents'] = df_cat['activation_fee_cents'].astype(int)
df_identity = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_04.csv')
df_incomp = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_05.csv')
df_item1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_06/export_06.csv')
df_item2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_01/export_07.csv')
for df in [df_item1, df_item2]:
    df['authorized'] = df['authorized'].astype(int)
    df['minimum_lot'] = df['minimum_lot'].astype(int)
    df['maximum_order'] = df['maximum_order'].astype(int)
    df['unit_benefit_cents'] = df['unit_benefit_cents'].astype(int)
    df['item_fee_cents'] = df['item_fee_cents'].astype(int)
df_items = pd.concat([df_item1, df_item2], ignore_index=True)
df_requires = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_03/export_09.csv')
df_usage1 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_04/export_10.csv')
df_usage2 = read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant36/inputs/batch_05/export_11.csv')
for df in [df_usage1, df_usage2]:
    df['amount'] = df['amount'].astype(int)
df_usage = pd.concat([df_usage1, df_usage2], ignore_index=True)
items_df = df_items[df_items['authorized'] == 1].copy()
item_refs = items_df['item_ref'].unique()
item_refs_set = set(item_refs)
item_params = items_df.set_index('item_ref').to_dict(orient='index')
categories = df_cat['category'].unique()
cat_params = df_cat.set_index('category').to_dict(orient='index')
item_to_cat = items_df.set_index('item_ref')['category'].to_dict()
cat_to_items = {c: set(items_df[items_df['category'] == c]['item_ref']) for c in categories}
resources = df_cap['resource'].unique()
cap_by_resource = df_cap.groupby('resource')['amount'].sum().to_dict()
usage_dict = {}
for _, row in df_usage.iterrows():
    i = row['item_ref']
    r = row['resource']
    amt = row['amount']
    if i in item_refs_set:
        usage_dict[i, r] = int(amt)
for i in item_refs:
    for r in resources:
        if (i, r) not in usage_dict:
            usage_dict[i, r] = 0
bundles = []
bundle_bonus = {}
for _, row in df_bundle.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs_set and b in item_refs_set:
        bundles.append((a, b))
        bundle_bonus[a, b] = row['bonus_cents']
incomp_pairs = []
for _, row in df_incomp.iterrows():
    a = row['item_a']
    b = row['item_b']
    if a in item_refs_set and b in item_refs_set:
        incomp_pairs.append((a, b))
requires_pairs = []
for _, row in df_requires.iterrows():
    i = row['item_ref']
    prereq = row['prerequisite_ref']
    if i in item_refs_set and prereq in item_refs_set:
        requires_pairs.append((i, prereq))
m = gp.Model('FC_EAST_HVAC_AC_Placement')
q_vars = m.addVars(item_refs, vtype=gp.GRB.INTEGER, name='')
z_vars = m.addVars(item_refs, vtype=gp.GRB.BINARY, name='')
g_vars = m.addVars(categories, vtype=gp.GRB.BINARY, name='')
b_vars = m.addVars(bundles, vtype=gp.GRB.BINARY, name='')
for i in item_refs:
    min_lot = item_params[i]['minimum_lot']
    max_order = item_params[i]['maximum_order']
    m.addConstr(q_vars[i] >= min_lot * z_vars[i])
    m.addConstr(q_vars[i] <= max_order * z_vars[i])
    m.addConstr(q_vars[i] >= 0)
    m.addConstr(z_vars[i] <= 1)
    m.addConstr(q_vars[i] <= max_order)
for r in resources:
    m.addConstr(gp.quicksum((usage_dict[i, r] * q_vars[i] for i in item_refs)) <= cap_by_resource[r])
for c in categories:
    items_in_c = cat_to_items[c]
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) >= cat_params[c]['minimum_quantity'])
    m.addConstr(gp.quicksum((q_vars[i] for i in items_in_c)) <= cat_params[c]['maximum_quantity'])
    for i in items_in_c:
        m.addConstr(z_vars[i] <= g_vars[c])
    m.addConstr(g_vars[c] <= gp.quicksum((z_vars[i] for i in items_in_c)))
for i, j in incomp_pairs:
    m.addConstr(z_vars[i] + z_vars[j] <= 1)
for i, prereq in requires_pairs:
    m.addConstr(z_vars[i] <= z_vars[prereq])
for a, b in bundles:
    m.addConstr(b_vars[a, b] <= z_vars[a])
    m.addConstr(b_vars[a, b] <= z_vars[b])
    m.addConstr(b_vars[a, b] >= z_vars[a] + z_vars[b] - 1)
item_benefit = gp.quicksum((item_params[i]['unit_benefit_cents'] * q_vars[i] for i in item_refs))
item_fees = gp.quicksum((item_params[i]['item_fee_cents'] * z_vars[i] for i in item_refs))
cat_fees = gp.quicksum((cat_params[c]['activation_fee_cents'] * g_vars[c] for c in categories))
bundle_bonuses = gp.quicksum((bundle_bonus[a, b] * b_vars[a, b] for a, b in bundles))
m.setObjective(item_benefit - item_fees - cat_fees + bundle_bonuses, gp.GRB.MAXIMIZE)
m.optimize()