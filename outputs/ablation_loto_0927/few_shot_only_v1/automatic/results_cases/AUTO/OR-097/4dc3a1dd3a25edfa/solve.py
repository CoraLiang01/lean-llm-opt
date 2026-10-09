import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
if 'Option' not in opt_df.columns:
    raise KeyError("OptionCharacteristics.csv must have 'Option' column.")
options = list(opt_df['Option'])
if len(options) != 120:
    raise ValueError(f'Expected 120 options, got {len(options)}.')
asset_cols = [col for col in asset_ref_df.columns if re.match('Asset_\\d+', col)]
if len(asset_cols) != 6:
    raise ValueError(f'Expected 6 asset columns, got {len(asset_cols)}.')
assets = asset_cols
if asset_ref_df.shape[0] != 120:
    raise ValueError(f'Expected 120 rows in Option_AssetReferenceMatrix.csv, got {asset_ref_df.shape[0]}.')

def get_col_dict(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing column '{col}' in OptionCharacteristics.csv")
    return dict(zip(df['Option'], df[col]))
cost = get_col_dict(opt_df, 'Cost')
delta = get_col_dict(opt_df, 'Delta')
gamma = get_col_dict(opt_df, 'Gamma')
vega = get_col_dict(opt_df, 'Vega')
maxlong = get_col_dict(opt_df, 'MaxLong')
maxshort = get_col_dict(opt_df, 'MaxShort')
option_to_idx = {opt: idx for (idx, opt) in enumerate(opt_df['Option'])}
A = {}
for opt in options:
    idx = option_to_idx[opt]
    for asset in assets:
        A[opt, asset] = int(asset_ref_df.loc[idx, asset])
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greeks = ['Delta', 'Gamma', 'Vega']
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={opt: int(maxshort[opt]) for opt in options}, ub={opt: int(maxlong[opt]) for opt in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[opt] * z[opt] for opt in options)), gp.GRB.MINIMIZE)
for opt in options:
    m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
    m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
for greek in greeks:
    if greek == 'Delta':
        G = delta
    elif greek == 'Gamma':
        G = gamma
    elif greek == 'Vega':
        G = vega
    else:
        raise ValueError(f'Unknown greek: {greek}')
    greek_sum = gp.LinExpr()
    for opt in options:
        for asset in assets:
            if A[opt, asset] == 1:
                greek_sum.addTerms(G[opt], x[opt])
    net_exposure = initial_exposure[greek] + greek_sum
    m.addConstr(net_exposure <= tolerance[greek], name=f'{greek}_upper')
    m.addConstr(net_exposure >= -tolerance[greek], name=f'{greek}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.6f}')
    print('--- Option Positions (x_i) ---')
    for opt in options:
        xi = x[opt].X
        if abs(xi) > 1e-06:
            print(f'Option {opt}: {int(round(xi))} contracts (|x|={int(round(z[opt].X))}, Cost={cost[opt]})')
    for greek in greeks:
        if greek == 'Delta':
            G = delta
        elif greek == 'Gamma':
            G = gamma
        elif greek == 'Vega':
            G = vega
        else:
            continue
        total = initial_exposure[greek]
        for opt in options:
            xi = x[opt].X
            for asset in assets:
                if A[opt, asset] == 1:
                    total += G[opt] * xi
        print(f'Net {greek} after hedging: {total:.6f} (tolerance ±{tolerance[greek]})')
else:
    print(f'No optimal solution found. Status: {m.status}')