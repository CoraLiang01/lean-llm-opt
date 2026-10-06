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
asset_cols = [col for col in asset_ref_df.columns if re.fullmatch('Asset_\\d+', col)]
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
if set(asset_ref_df['Option']) != set(options):
    raise ValueError('Option identifiers do not match between OptionCharacteristics and Option_AssetReferenceMatrix.')
A = {}
for i in options:
    row = asset_ref_df.loc[asset_ref_df['Option'] == i]
    if row.empty:
        raise ValueError(f'Option {i} not found in Option_AssetReferenceMatrix.')
    A[i] = {}
    for j in assets:
        A[i][j] = int(row.iloc[0][j])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
for G in greeks:
    greek_sum = gp.LinExpr()
    for i in options:
        for j in assets:
            greek_sum.addTerms(G_coeff[G][i] * A[i][j], x[i])
    net_exposure = G_initial[G] + greek_sum
    m.addConstr(net_exposure <= G_tolerance[G], name=f'{G}_upper')
    m.addConstr(net_exposure >= -G_tolerance[G], name=f'{G}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.2f}')
    print('--- Option Positions (x_i) ---')
    for i in options:
        xi = x[i].X
        if abs(xi) > 1e-06:
            print(f'{i}: {int(round(xi))} contracts (|x|={int(round(z[i].X))}, Cost={cost[i]})')
    print('-----------------------------')
    for G in greeks:
        greek_sum = 0.0
        for i in options:
            for j in assets:
                greek_sum += G_coeff[G][i] * A[i][j] * x[i].X
        net = G_initial[G] + greek_sum
        print(f'Net {G} after hedging: {net:.5f} (tolerance ±{G_tolerance[G]})')
else:
    print(f'No optimal solution found. Status: {m.status}')