import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
df_opt = pd.read_csv(option_char_path, sep=',')
df_ref = pd.read_csv(asset_ref_path, sep=',')
df_opt['Option'] = df_opt['Option'].astype(str).str.strip()
df_ref['Unnamed: 0'] = df_ref['Unnamed: 0'].astype(str).str.strip()
options = list(df_opt['Option'])
assets = [col for col in df_ref.columns if col.startswith('Asset_')]
if len(assets) != 6:
    raise ValueError('Expected 6 asset columns in Option_AssetReferenceMatrix.csv')
if set(options) != set(df_ref['Unnamed: 0']):
    missing_in_ref = set(options) - set(df_ref['Unnamed: 0'])
    missing_in_opt = set(df_ref['Unnamed: 0']) - set(options)
    raise ValueError(f'Option identifier mismatch between files. Missing in ref: {missing_in_ref}, missing in opt: {missing_in_opt}')
df_ref = df_ref.set_index('Unnamed: 0').loc[options].reset_index()
cost = dict(zip(df_opt['Option'], df_opt['Cost']))
delta = dict(zip(df_opt['Option'], df_opt['Delta']))
gamma = dict(zip(df_opt['Option'], df_opt['Gamma']))
vega = dict(zip(df_opt['Option'], df_opt['Vega']))
maxlong = dict(zip(df_opt['Option'], df_opt['MaxLong']))
maxshort = dict(zip(df_opt['Option'], df_opt['MaxShort']))
A = {}
for (i, row) in df_ref.iterrows():
    opt = row['Unnamed: 0']
    for asset in assets:
        A[opt, asset] = int(row[asset])
GREEKS = ['Delta', 'Gamma', 'Vega']
G_init = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
m.Params.MIPGap = 0.0001
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in options}, ub={i: maxlong[i] for i in options}, name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, name='')
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
for G in GREEKS:
    expr = gp.LinExpr()
    for i in options:
        for asset in assets:
            coeff = G_coeff[G][i] * A[i, asset]
            if coeff != 0:
                expr.add(x[i], coeff)
    m.addConstr(G_init[G] + expr <= G_tol[G], name=f'{G}_upper')
    m.addConstr(G_init[G] + expr >= -G_tol[G], name=f'{G}_lower')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
        print(f'{z[i].VarName} {z[i].X}')
else:
    print(f'Solver status: {m.status}')