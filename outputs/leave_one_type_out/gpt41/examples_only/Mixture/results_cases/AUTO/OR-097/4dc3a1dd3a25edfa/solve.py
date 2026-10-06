import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
options = list(opt_df['Option'].astype(str))
n_options = len(options)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
assets = asset_cols
n_assets = len(assets)
greeks = ['Delta', 'Gamma', 'Vega']
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
asset_ref_df['Option'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
opt_df['Option'] = opt_df['Option'].astype(str).str.strip()
if not set(options) <= set(asset_ref_df['Option']):
    missing = set(options) - set(asset_ref_df['Option'])
    raise ValueError(f'Missing asset reference rows for options: {missing}')
cost = dict(zip(opt_df['Option'], opt_df['Cost']))
maxlong = dict(zip(opt_df['Option'], opt_df['MaxLong']))
maxshort = dict(zip(opt_df['Option'], opt_df['MaxShort']))
greek_val = {g: dict(zip(opt_df['Option'], opt_df[g])) for g in greeks}
A = {}
for i in options:
    row = asset_ref_df.loc[asset_ref_df['Option'] == i]
    if row.empty:
        raise ValueError(f'Option {i} not found in asset reference matrix.')
    A[i] = {a: int(row.iloc[0][a]) for a in assets}
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
for g in greeks:
    exposure_expr = initial_exposure[g] + gp.quicksum((greek_val[g][i] * A[i][a] * x[i] for i in options for a in assets))
    m.addConstr(exposure_expr <= tolerance[g], name=f'{g}_upper')
    m.addConstr(exposure_expr >= -tolerance[g], name=f'{g}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.2f}')
    print('\n--- Optimal Option Positions (x_i) ---')
    for i in options:
        xi = x[i].X
        if abs(xi) > 1e-06:
            print(f'  {i}: {int(round(xi)):+d} contracts (|x|={int(round(z[i].X))})')
    print('\n--- Post-Hedging Exposures ---')
    for g in greeks:
        exposure = initial_exposure[g] + sum((greek_val[g][i] * sum((A[i][a] * x[i].X for a in assets)) for i in options))
        print(f'  {g}: {exposure:.5f} (tolerance ±{tolerance[g]:.5f})')
else:
    print(f'No optimal solution found. Status: {m.status}')