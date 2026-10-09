import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
optasset_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
optasset_df = pd.read_csv(optasset_path, sep=',', dtype=str, keep_default_na=False)
option_ids = optchar_df['Option'].tolist()
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_ids)}')
asset_cols = [col for col in optasset_df.columns if re.match('Asset_\\d+$', col)]
asset_ids = asset_cols
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_ids)}')
optchar_df = optchar_df.set_index('Option', drop=False)
cost = optchar_df['Cost'].astype(float).to_dict()
delta = optchar_df['Delta'].astype(float).to_dict()
gamma = optchar_df['Gamma'].astype(float).to_dict()
vega = optchar_df['Vega'].astype(float).to_dict()
maxlong = optchar_df['MaxLong'].astype(int).to_dict()
maxshort = optchar_df['MaxShort'].astype(int).to_dict()
optasset_df = optasset_df.set_index('Unnamed: 0', drop=False)
if not set(option_ids).issubset(set(optasset_df.index)):
    missing = set(option_ids) - set(optasset_df.index)
    raise ValueError(f'Options missing in Option_AssetReferenceMatrix: {missing}')
A = {}
for i in option_ids:
    A[i] = {}
    for j in asset_ids:
        A[i][j] = int(optasset_df.loc[i, j])
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[i] for i in option_ids], ub=[maxlong[i] for i in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in option_ids:
    m.addConstr(z_vars[i] >= x_vars[i], name=f'abs1_{i}')
    m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs2_{i}')
for G in greek_names:
    expr = greek_initial[G]
    for i in option_ids:
        for j in asset_ids:
            if A[i][j] != 0:
                expr += greek_coeff[G][i] * A[i][j] * x_vars[i]
    m.addConstr(expr <= greek_tol[G], name=f'{G}_upper')
    m.addConstr(expr >= -greek_tol[G], name=f'{G}_lower')
m.setObjective(gp.quicksum((cost[i] * z_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.optimize()