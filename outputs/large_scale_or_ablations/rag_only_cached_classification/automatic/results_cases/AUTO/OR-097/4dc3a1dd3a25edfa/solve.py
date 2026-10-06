import pandas as pd
import numpy as np
from gurobipy import Model, GRB
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
df_opt = pd.read_csv(opt_char_path, sep=',')
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
df_asset = pd.read_csv(asset_ref_path, sep=',')
options = df_opt['Option'].astype(str).tolist()
if len(options) != 120:
    raise ValueError(f'Expected 120 options, got {len(options)}')
asset_cols = [col for col in df_asset.columns if col.startswith('Asset_')]
assets = asset_cols
if len(assets) != 6:
    raise ValueError(f'Expected 6 assets, got {len(assets)}')
opt_idx_map = {opt: i for i, opt in enumerate(options)}
asset_opts = df_asset['Unnamed: 0'].astype(str).tolist()
if set(options) != set(asset_opts):
    raise ValueError('Option identifiers do not match between OptionCharacteristics and AssetReferenceMatrix.')
df_asset = df_asset.set_index('Unnamed: 0').reindex(options)
cost = df_opt.set_index('Option')['Cost'].astype(float).to_dict()
delta = df_opt.set_index('Option')['Delta'].astype(float).to_dict()
gamma = df_opt.set_index('Option')['Gamma'].astype(float).to_dict()
vega = df_opt.set_index('Option')['Vega'].astype(float).to_dict()
maxlong = df_opt.set_index('Option')['MaxLong'].astype(int).to_dict()
maxshort = df_opt.set_index('Option')['MaxShort'].astype(int).to_dict()
A = {}
for opt in options:
    A[opt] = {}
    for asset in assets:
        A[opt][asset] = int(df_asset.loc[opt, asset])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = Model('option_hedging')
x = m.addVars(options, vtype=GRB.INTEGER, name='')
z = m.addVars(options, vtype=GRB.CONTINUOUS, lb=0.0, name='')
for opt in options:
    m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
for opt in options:
    m.addConstr(x[opt] >= maxshort[opt], name=f'min_{opt}')
    m.addConstr(x[opt] <= maxlong[opt], name=f'max_{opt}')
for G in greeks:
    expr = G_initial[G]
    for opt in options:
        coeff = 0.0
        for asset in assets:
            coeff += G_coeff[G][opt] * A[opt][asset]
        expr += coeff * x[opt]
    m.addConstr(expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(expr >= -G_tol[G], name=f'{G}_lower')
m.setObjective(sum((cost[opt] * z[opt] for opt in options)), GRB.MINIMIZE)
m.optimize()