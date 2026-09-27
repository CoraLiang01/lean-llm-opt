import gurobipy as gp
import pandas as pd
import numpy as np
import re
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
cost = dict(zip(opt_df['Option'].astype(str), opt_df['Cost']))
delta = dict(zip(opt_df['Option'].astype(str), opt_df['Delta']))
gamma = dict(zip(opt_df['Option'].astype(str), opt_df['Gamma']))
vega = dict(zip(opt_df['Option'].astype(str), opt_df['Vega']))
maxlong = dict(zip(opt_df['Option'].astype(str), opt_df['MaxLong']))
maxshort = dict(zip(opt_df['Option'].astype(str), opt_df['MaxShort']))
asset_ref_df['Option'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
asset_ref_df = asset_ref_df.set_index('Option')
A = {}
for i in options:
    if i not in asset_ref_df.index:
        raise ValueError(f'Option {i} not found in asset reference matrix')
    A[i] = {}
    for j in assets:
        A[i][j] = int(asset_ref_df.loc[i, j])
greeks = ['Delta', 'Gamma', 'Vega']
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
for G in greeks:
    expr = initial_exposure[G] + gp.quicksum((greek_coeff[G][i] * A[i][j] * x[i] for i in options for j in assets))
    m.addConstr(expr <= tolerance[G], name=f'{G}_upper')
    m.addConstr(expr >= -tolerance[G], name=f'{G}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.2f}')
    print('--- Option Positions (x[i]) ---')
    for i in options:
        xi = x[i].X
        if abs(xi) > 1e-06:
            print(f'{i}: {int(round(xi))} contracts (Cost per contract: {cost[i]}, |x|={int(round(z[i].X))})')
    print('------------------------------')
    for G in greeks:
        exposure = initial_exposure[G]
        for i in options:
            for j in assets:
                exposure += greek_coeff[G][i] * A[i][j] * x[i].X
        print(f'Final {G} exposure: {exposure:.6f} (tolerance ±{tolerance[G]})')
else:
    print(f'No optimal solution found. Status: {m.status}')