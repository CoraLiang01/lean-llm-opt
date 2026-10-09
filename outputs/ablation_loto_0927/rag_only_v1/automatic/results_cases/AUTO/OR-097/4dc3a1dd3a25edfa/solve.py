import pandas as pd
import numpy as np
from gurobipy import Model, GRB
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_char = pd.read_csv(opt_char_path, sep=',')
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
asset_ref = pd.read_csv(asset_ref_path, sep=',')
opt_char['Option'] = opt_char['Option'].astype(str).str.strip()
asset_ref['Unnamed: 0'] = asset_ref['Unnamed: 0'].astype(str).str.strip()
options = list(opt_char['Option'])
assets = [col for col in asset_ref.columns if col.startswith('Asset_')]
if set(options) != set(asset_ref['Unnamed: 0']):
    raise ValueError('Mismatch between options in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv')
opt_to_assetref_idx = {opt: idx for (idx, opt) in enumerate(asset_ref['Unnamed: 0'])}
cost = dict(zip(opt_char['Option'], opt_char['Cost']))
delta = dict(zip(opt_char['Option'], opt_char['Delta']))
gamma = dict(zip(opt_char['Option'], opt_char['Gamma']))
vega = dict(zip(opt_char['Option'], opt_char['Vega']))
maxlong = dict(zip(opt_char['Option'], opt_char['MaxLong']))
maxshort = dict(zip(opt_char['Option'], opt_char['MaxShort']))
A = {}
for opt in options:
    row = asset_ref.loc[asset_ref['Unnamed: 0'] == opt]
    if row.empty:
        raise ValueError(f'Option {opt} not found in Option_AssetReferenceMatrix.csv')
    row = row.iloc[0]
    A[opt] = {asset: int(row[asset]) for asset in assets}
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = Model('option_hedging')
x = m.addVars(options, vtype=GRB.INTEGER, name='')
z = m.addVars(options, vtype=GRB.INTEGER, name='')
for opt in options:
    m.addConstr(x[opt] >= maxshort[opt], name=f'minpos_{opt}')
    m.addConstr(x[opt] <= maxlong[opt], name=f'maxpos_{opt}')
for opt in options:
    m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
for G in greeks:
    expr = 0
    for opt in options:
        coeff = G_coeff[G][opt]
        for asset in assets:
            if A[opt][asset] == 1:
                expr += coeff * x[opt]
    m.addConstr(G_initial[G] + expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(G_initial[G] + expr >= -G_tol[G], name=f'{G}_lower')
m.setObjective(sum((cost[opt] * z[opt] for opt in options)), GRB.MINIMIZE)
m.optimize()