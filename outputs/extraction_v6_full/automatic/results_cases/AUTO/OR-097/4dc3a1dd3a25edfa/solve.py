import gurobipy as gp
import pandas as pd
import numpy as np
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_df = pd.read_csv(option_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
option_ids = option_df['Option'].astype(str).tolist()
n_options = len(option_ids)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
asset_ids = [col for col in asset_cols]
n_assets = len(asset_ids)
option_df['Option'] = option_df['Option'].astype(str).str.strip()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
if set(option_ids) != set(asset_ref_df['Unnamed: 0']):
    raise ValueError('Mismatch between OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv option IDs.')
option_to_assetrow = {oid: asset_ref_df.index[asset_ref_df['Unnamed: 0'] == oid][0] for oid in option_ids}
cost = option_df.set_index('Option')['Cost'].astype(float).to_dict()
delta = option_df.set_index('Option')['Delta'].astype(float).to_dict()
gamma = option_df.set_index('Option')['Gamma'].astype(float).to_dict()
vega = option_df.set_index('Option')['Vega'].astype(float).to_dict()
maxlong = option_df.set_index('Option')['MaxLong'].astype(int).to_dict()
maxshort = option_df.set_index('Option')['MaxShort'].astype(int).to_dict()
A = {}
for oid in option_ids:
    row = asset_ref_df.loc[option_to_assetrow[oid], asset_cols]
    A[oid] = {aid: int(row[aid]) for aid in asset_ids}
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, name='x')
z = m.addVars(option_ids, vtype=gp.GRB.INTEGER, name='z')
m.setObjective(gp.quicksum((cost[oid] * z[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
for oid in option_ids:
    m.addConstr(z[oid] >= x[oid], name=f'abs_pos_{oid}')
    m.addConstr(z[oid] >= -x[oid], name=f'abs_neg_{oid}')
for oid in option_ids:
    m.addConstr(x[oid] >= maxshort[oid], name=f'min_{oid}')
    m.addConstr(x[oid] <= maxlong[oid], name=f'max_{oid}')
for G in greeks:
    expr = G_initial[G]
    for oid in option_ids:
        for aid in asset_ids:
            if A[oid][aid] == 1:
                expr += G_coeff[G][oid] * x[oid]
    m.addConstr(expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(expr >= -G_tol[G], name=f'{G}_lower')
m.optimize()