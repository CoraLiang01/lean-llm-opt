import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
optchar_df['Option'] = optchar_df['Option'].str.strip()
assetref_df['Unnamed: 0'] = assetref_df['Unnamed: 0'].str.strip()
option_ids = list(optchar_df['Option'])
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_ids)}')
asset_cols = [col for col in assetref_df.columns if col != 'Unnamed: 0']
if len(asset_cols) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_cols)}')
optchar_df.set_index('Option', inplace=True)
assetref_df.set_index('Unnamed: 0', inplace=True)
cost = optchar_df['Cost'].astype(float).to_dict()
delta = optchar_df['Delta'].astype(float).to_dict()
gamma = optchar_df['Gamma'].astype(float).to_dict()
vega = optchar_df['Vega'].astype(float).to_dict()
maxlong = optchar_df['MaxLong'].astype(int).to_dict()
maxshort = optchar_df['MaxShort'].astype(int).to_dict()
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_cols:
        A[opt][asset] = int(assetref_df.loc[opt, asset])
greeks = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in option_ids}, ub={opt: maxlong[opt] for opt in option_ids}, name='')
y_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[opt] * y_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
for opt in option_ids:
    m.addConstr(y_vars[opt] >= x_vars[opt], name=f'abs_pos_{opt}')
    m.addConstr(y_vars[opt] >= -x_vars[opt], name=f'abs_neg_{opt}')
for greek in greeks:
    expr = greek_initial[greek] + gp.quicksum((greek_coeff[greek][opt] * A[opt][asset] * x_vars[opt] for opt in option_ids for asset in asset_cols))
    tol = greek_tol[greek]
    m.addConstr(expr <= tol, name=f'{greek}_upper')
    m.addConstr(expr >= -tol, name=f'{greek}_lower')
m.optimize()