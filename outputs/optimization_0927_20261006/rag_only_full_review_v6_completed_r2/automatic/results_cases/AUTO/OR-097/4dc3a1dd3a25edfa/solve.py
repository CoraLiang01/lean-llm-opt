import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_char_df = pd.read_csv(opt_char_path, sep=',', dtype=str, keep_default_na=False)
opt_char_df['Option'] = opt_char_df['Option'].str.strip()
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    opt_char_df[col] = pd.to_numeric(opt_char_df[col], errors='raise')
option_ids = list(opt_char_df['Option'])
cost = dict(zip(option_ids, opt_char_df['Cost']))
delta = dict(zip(option_ids, opt_char_df['Delta']))
gamma = dict(zip(option_ids, opt_char_df['Gamma']))
vega = dict(zip(option_ids, opt_char_df['Vega']))
maxlong = dict(zip(option_ids, opt_char_df['MaxLong']))
maxshort = dict(zip(option_ids, opt_char_df['MaxShort']))
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].str.strip()
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
asset_ids = asset_cols
if set(option_ids) != set(asset_ref_df['Unnamed: 0']):
    raise ValueError('Mismatch between OptionCharacteristics and Option_AssetReferenceMatrix Option IDs.')
A = {}
for (_, row) in asset_ref_df.iterrows():
    opt = row['Unnamed: 0']
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(row[asset])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
m = Model('option_hedging')
x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb=[maxshort[oid] for oid in option_ids], ub=[maxlong[oid] for oid in option_ids], name='')
abs_x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, name='')
for oid in option_ids:
    m.addConstr(abs_x_vars[oid] >= x_vars[oid])
    m.addConstr(abs_x_vars[oid] >= -x_vars[oid])
m.setObjective(quicksum((cost[oid] * abs_x_vars[oid] for oid in option_ids)), GRB.MINIMIZE)
for greek in greeks:
    if greek == 'Delta':
        G = delta
    elif greek == 'Gamma':
        G = gamma
    elif greek == 'Vega':
        G = vega
    else:
        raise ValueError(f'Unknown Greek: {greek}')
    greek_expr = quicksum((G[oid] * A[oid][asset] * x_vars[oid] for oid in option_ids for asset in asset_ids))
    m.addConstr(G_initial[greek] + greek_expr <= tolerance[greek])
    m.addConstr(G_initial[greek] + greek_expr >= -tolerance[greek])
m.optimize()