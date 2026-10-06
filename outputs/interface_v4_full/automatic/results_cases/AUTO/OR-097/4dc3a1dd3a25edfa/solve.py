import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(opt_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
opt_df['Option'] = opt_df['Option'].astype(str).str.strip()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
opt_ids = list(opt_df['Option'])
asset_ids = list(asset_ref_df['Unnamed: 0'])
if set(opt_ids) != set(asset_ids):
    raise ValueError('Mismatch between Option IDs in OptionCharacteristics and Option_AssetReferenceMatrix.')
opt_df = opt_df.set_index('Option').loc[opt_ids]
asset_ref_df = asset_ref_df.set_index('Unnamed: 0').loc[opt_ids]
options = opt_ids
assets = ['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4', 'Asset_5', 'Asset_6']
cost = opt_df['Cost'].to_dict()
delta = opt_df['Delta'].to_dict()
gamma = opt_df['Gamma'].to_dict()
vega = opt_df['Vega'].to_dict()
maxlong = opt_df['MaxLong'].to_dict()
maxshort = opt_df['MaxShort'].to_dict()
A = {}
for i in options:
    A[i] = {}
    for j in assets:
        A[i][j] = int(asset_ref_df.loc[i, j])
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
for G in greeks:
    expr = gp.LinExpr()
    for i in options:
        coeff = G_coeff[G][i]
        for j in assets:
            if A[i][j] != 0:
                expr.addTerms(coeff * A[i][j], x[i])
    net_exposure = G_initial[G] + expr
    m.addConstr(net_exposure <= G_tol[G], name=f'{G}_plus')
    m.addConstr(net_exposure >= -G_tol[G], name=f'{G}_minus')
m.optimize()