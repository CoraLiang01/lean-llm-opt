import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
option_ids = list(opt_df['Option'].astype(str))
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_ids)}')
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
asset_ids = [col for col in asset_cols]
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_ids)}')
asset_ref_df = asset_ref_df.set_index(asset_ref_df['Unnamed: 0'].astype(str))
if not set(option_ids).issubset(set(asset_ref_df.index)):
    missing = set(option_ids) - set(asset_ref_df.index)
    raise ValueError(f'Options missing in asset reference matrix: {missing}')
cost = dict(zip(option_ids, opt_df['Cost']))
delta = dict(zip(option_ids, opt_df['Delta']))
gamma = dict(zip(option_ids, opt_df['Gamma']))
vega = dict(zip(option_ids, opt_df['Vega']))
maxlong = dict(zip(option_ids, opt_df['MaxLong']))
maxshort = dict(zip(option_ids, opt_df['MaxShort']))
A = {}
for i in option_ids:
    A[i] = {}
    for j in asset_ids:
        A[i][j] = int(asset_ref_df.loc[i, j])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in option_ids}, ub={i: maxlong[i] for i in option_ids}, name='')
z = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in option_ids:
    m.addConstr(z[i] >= x[i], name=f'abs_pos_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs_neg_{i}')
for G in greeks:
    expr = gp.LinExpr()
    for i in option_ids:
        coeff = G_coeff[G][i]
        for j in asset_ids:
            if A[i][j] != 0:
                expr.addTerms(coeff * A[i][j], x[i])
    m.addConstr(G_initial[G] + expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(G_initial[G] + expr >= -G_tol[G], name=f'{G}_lower')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.optimize()