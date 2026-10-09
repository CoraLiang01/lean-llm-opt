import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
options = list(opt_df['Option'].astype(str))
if len(options) != 120:
    raise ValueError(f'Expected 120 options, got {len(options)}')
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
assets = asset_cols
if len(assets) != 6:
    raise ValueError(f'Expected 6 assets, got {len(assets)}')
opt_df_idx = opt_df.set_index('Option')
asset_ref_df_idx = asset_ref_df.set_index('Unnamed: 0')
if not set(options).issubset(set(asset_ref_df_idx.index)):
    missing = set(options) - set(asset_ref_df_idx.index)
    raise ValueError(f'Options missing from asset reference matrix: {missing}')
cost = opt_df_idx['Cost'].to_dict()
delta = opt_df_idx['Delta'].to_dict()
gamma = opt_df_idx['Gamma'].to_dict()
vega = opt_df_idx['Vega'].to_dict()
maxlong = opt_df_idx['MaxLong'].to_dict()
maxshort = opt_df_idx['MaxShort'].to_dict()
A = {}
for i in options:
    for j in assets:
        A[i, j] = int(asset_ref_df_idx.loc[i, j])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
m.Params.MIPGap = 0.0001
x = m.addVars(options, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, vtype=gp.GRB.INTEGER, name='')
z = m.addVars(options, lb=0, vtype=gp.GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for i in options:
    m.addConstr(z[i] >= x[i], name=f'z_ge_x_{i}')
    m.addConstr(z[i] >= -x[i], name=f'z_ge_negx_{i}')
for G in greeks:
    expr = G_initial[G] + gp.quicksum((G_coeff[G][i] * A[i, j] * x[i] for i in options for j in assets))
    m.addConstr(expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(expr >= -G_tol[G], name=f'{G}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
        print(f'{z[i].VarName} {z[i].X}')
else:
    print(f'Solver status: {m.status}')