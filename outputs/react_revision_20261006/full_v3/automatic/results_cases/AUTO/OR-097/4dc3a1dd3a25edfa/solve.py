import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_df = pd.read_csv(option_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
option_df['Option'] = option_df['Option'].astype(str).str.strip()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
options = list(option_df['Option'])
assets = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
if len(assets) != 6:
    raise ValueError('Expected 6 asset columns in Option_AssetReferenceMatrix.csv')
if set(options) != set(asset_ref_df['Unnamed: 0']):
    missing_in_ref = set(options) - set(asset_ref_df['Unnamed: 0'])
    missing_in_char = set(asset_ref_df['Unnamed: 0']) - set(options)
    raise ValueError(f'Option identifier mismatch between files. Missing in reference: {missing_in_ref}, Missing in characteristics: {missing_in_char}')
asset_ref_df = asset_ref_df.set_index('Unnamed: 0').loc[options]
cost = option_df.set_index('Option')['Cost'].to_dict()
delta = option_df.set_index('Option')['Delta'].to_dict()
gamma = option_df.set_index('Option')['Gamma'].to_dict()
vega = option_df.set_index('Option')['Vega'].to_dict()
maxlong = option_df.set_index('Option')['MaxLong'].to_dict()
maxshort = option_df.set_index('Option')['MaxShort'].to_dict()
A = {}
for i in options:
    for j in assets:
        A[i, j] = int(asset_ref_df.loc[i, j])
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
m = gp.Model('OptionHedging')
m.setParam('MIPGap', 0.0001)
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=[maxshort[i] for i in options], ub=[maxlong[i] for i in options], name='')
z = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in options:
    m.addConstr(z[i] >= x[i], name=f'abs1_{i}')
    m.addConstr(z[i] >= -x[i], name=f'abs2_{i}')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in options)), gp.GRB.MINIMIZE)
for (greek, greek_vec) in [('Delta', delta), ('Gamma', gamma), ('Vega', vega)]:
    expr = gp.LinExpr()
    for i in options:
        for j in assets:
            if A[i, j] != 0:
                expr.add(greek_vec[i] * A[i, j] * x[i])
    m.addConstr(initial_exposure[greek] + expr <= tolerance[greek], name=f'{greek}_upper')
    m.addConstr(initial_exposure[greek] + expr >= -tolerance[greek], name=f'{greek}_lower')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
        print(f'{z[i].VarName} {z[i].X}')
else:
    print(f'Solver status: {m.status}')