import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
if 'Option' not in optchar_df.columns:
    raise KeyError("OptionCharacteristics.csv must have an 'Option' column.")
options = list(optchar_df['Option'])
asset_cols = [col for col in assetref_df.columns if re.fullmatch('Asset_\\d+', col)]
assets = asset_cols

def to_float_series(df, col, idx_col):
    return pd.Series(df[col].astype(float).values, index=df[idx_col])

def to_int_series(df, col, idx_col):
    return pd.Series(df[col].astype(int).values, index=df[idx_col])
cost = to_float_series(optchar_df, 'Cost', 'Option').to_dict()
delta = to_float_series(optchar_df, 'Delta', 'Option').to_dict()
gamma = to_float_series(optchar_df, 'Gamma', 'Option').to_dict()
vega = to_float_series(optchar_df, 'Vega', 'Option').to_dict()
maxlong = to_int_series(optchar_df, 'MaxLong', 'Option').to_dict()
maxshort = to_int_series(optchar_df, 'MaxShort', 'Option').to_dict()
if 'Unnamed: 0' not in assetref_df.columns:
    raise KeyError("Option_AssetReferenceMatrix.csv must have an 'Unnamed: 0' column for Option IDs.")
assetref_df = assetref_df.set_index('Unnamed: 0')
if not set(options).issubset(set(assetref_df.index)):
    raise ValueError('Some Option IDs in OptionCharacteristics.csv are missing from Option_AssetReferenceMatrix.csv.')
A = {}
for opt in options:
    for asset in assets:
        val = assetref_df.loc[opt, asset]
        try:
            A[opt, asset] = int(val)
        except Exception:
            raise ValueError(f'Non-integer value in asset reference matrix at Option {opt}, Asset {asset}: {val}')
greeks = ['Delta', 'Gamma', 'Vega']
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in options}, ub={opt: maxlong[opt] for opt in options}, name='')
z_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
for opt in options:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs_pos_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs_neg_{opt}')
for greek in greeks:
    greek_contrib = gp.quicksum((greek_param[greek][opt] * A[opt, asset] * x_vars[opt] for opt in options for asset in assets))
    net_exposure = initial_exposure[greek] + greek_contrib
    m.addConstr(net_exposure <= tolerance[greek], name=f'{greek}_upper')
    m.addConstr(net_exposure >= -tolerance[greek], name=f'{greek}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.6f}')
    print('--- Option Positions ---')
    for opt in options:
        xval = x_vars[opt].X
        if abs(xval) > 1e-06:
            print(f'Option {opt}: {xval:.0f} contracts (|x|={z_vars[opt].X:.0f}, Cost/contract={cost[opt]:.4f})')
    print('-----------------------')
    for greek in greeks:
        exposure = initial_exposure[greek] + sum((greek_param[greek][opt] * sum((A[opt, asset] for asset in assets)) * x_vars[opt].X for opt in options))
        print(f'Net {greek} after hedging: {exposure:.6f} (tolerance ±{tolerance[greek]:.6f})')
else:
    print(f'No optimal solution found. Status: {m.status}')