import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
options = list(opt_df['Option'])
n_options = len(options)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
assets = asset_cols
n_assets = len(assets)
asset_ref_options = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
opt_df_options = pd.Series(options).astype(str).str.strip()
if not set(asset_ref_options) == set(opt_df_options):
    raise ValueError('Mismatch between Option identifiers in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv.')
option_to_idx = {opt: i for (i, opt) in enumerate(options)}
asset_ref_df = asset_ref_df.set_index('Unnamed: 0')
asset_ref_df.index = asset_ref_df.index.astype(str).str.strip()
opt_df = opt_df.set_index('Option')
opt_df.index = opt_df.index.astype(str).str.strip()
cost = opt_df['Cost'].to_dict()
delta = opt_df['Delta'].to_dict()
gamma = opt_df['Gamma'].to_dict()
vega = opt_df['Vega'].to_dict()
maxlong = opt_df['MaxLong'].to_dict()
maxshort = opt_df['MaxShort'].to_dict()
A = {}
for opt in options:
    for asset in assets:
        A[opt, asset] = int(asset_ref_df.loc[opt, asset])
greeks = ['Delta', 'Gamma', 'Vega']
greek_param = {'Delta': {'col': delta, 'init': 0.25, 'tol': 0.06}, 'Gamma': {'col': gamma, 'init': 0.08, 'tol': 0.05}, 'Vega': {'col': vega, 'init': 0.17, 'tol': 0.07}}
m = gp.Model('OptionHedging')
m.Params.MIPGap = 0.0001
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in options}, ub={opt: maxlong[opt] for opt in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in options:
    m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
for greek in greeks:
    g_col = greek_param[greek]['col']
    g_init = greek_param[greek]['init']
    g_tol = greek_param[greek]['tol']
    expr = g_init + gp.quicksum((g_col[opt] * A[opt, asset] * x[opt] for opt in options for asset in assets))
    m.addConstr(expr <= g_tol, name=f'{greek}_upper')
    m.addConstr(expr >= -g_tol, name=f'{greek}_lower')
m.setObjective(gp.quicksum((cost[opt] * z[opt] for opt in options)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for opt in options:
        print(f'{x[opt].VarName} {x[opt].X}')
        print(f'{z[opt].VarName} {z[opt].X}')
else:
    print(f'Solver status: {m.status}')