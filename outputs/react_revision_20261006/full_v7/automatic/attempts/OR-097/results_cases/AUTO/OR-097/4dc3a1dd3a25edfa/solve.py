import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
opt_df['Option'] = opt_df['Option'].str.strip()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].str.strip()
option_ids = list(opt_df['Option'])
asset_ids = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
if len(asset_ids) != 6:
    raise ValueError('Expected 6 asset columns in Option_AssetReferenceMatrix.csv')
if not set(asset_ref_df['Unnamed: 0']) == set(option_ids):
    raise ValueError('Mismatch between options in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv')
asset_ref_df = asset_ref_df.set_index('Unnamed: 0').loc[option_ids]
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    opt_df[col] = pd.to_numeric(opt_df[col], errors='raise')
for col in asset_ids:
    asset_ref_df[col] = pd.to_numeric(asset_ref_df[col], errors='raise')
cost = dict(zip(option_ids, opt_df['Cost']))
delta = dict(zip(option_ids, opt_df['Delta']))
gamma = dict(zip(option_ids, opt_df['Gamma']))
vega = dict(zip(option_ids, opt_df['Vega']))
maxlong = dict(zip(option_ids, opt_df['MaxLong']))
maxshort = dict(zip(option_ids, opt_df['MaxShort']))
A = {}
for i in option_ids:
    for j in asset_ids:
        A[i, j] = int(asset_ref_df.loc[i, j])
greek_names = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=None, ub=None, name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in option_ids:
    m.addConstr(x_vars[i] >= maxshort[i], name=f'min_{i}')
    m.addConstr(x_vars[i] <= maxlong[i], name=f'max_{i}')
for i in option_ids:
    m.addConstr(z_vars[i] >= x_vars[i], name=f'abs1_{i}')
    m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs2_{i}')
for G in greek_names:
    expr = gp.LinExpr()
    for i in option_ids:
        for j in asset_ids:
            coeff = G_coeff[G][i] * A[i, j]
            if coeff != 0:
                expr.addTerms(coeff, x_vars[i])
    m.addConstr(G_initial[G] + expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(G_initial[G] + expr >= -G_tol[G], name=f'{G}_lower')
obj = gp.quicksum((cost[i] * z_vars[i] for i in option_ids))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()