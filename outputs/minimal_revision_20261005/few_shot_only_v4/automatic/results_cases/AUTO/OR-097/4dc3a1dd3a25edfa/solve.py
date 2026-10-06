import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_df = pd.read_csv(option_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
option_df['Option'] = option_df['Option'].astype(str)
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str)
option_keys = list(option_df['Option'])
asset_keys = [col for col in asset_ref_df.columns if re.fullmatch('Asset_\\d+', col)]
if len(asset_keys) != 6:
    raise ValueError('Expected 6 asset columns in Option_AssetReferenceMatrix.csv')
asset_ref_df = asset_ref_df.set_index('Unnamed: 0')
if not set(option_keys).issubset(set(asset_ref_df.index)):
    missing = set(option_keys) - set(asset_ref_df.index)
    raise ValueError(f'Options missing in asset reference matrix: {missing}')
cost = option_df.set_index('Option')['Cost'].to_dict()
delta = option_df.set_index('Option')['Delta'].to_dict()
gamma = option_df.set_index('Option')['Gamma'].to_dict()
vega = option_df.set_index('Option')['Vega'].to_dict()
maxlong = option_df.set_index('Option')['MaxLong'].to_dict()
maxshort = option_df.set_index('Option')['MaxShort'].to_dict()
A = {}
for i in option_keys:
    for j in asset_keys:
        val = asset_ref_df.loc[i, j]
        if not (val == 0 or val == 1):
            raise ValueError(f'Non-binary value in asset reference matrix at ({i},{j}): {val}')
        A[i, j] = int(val)
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
if len(option_keys) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_keys)}')
if len(asset_keys) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_keys)}')
for g in greek_names:
    if len(greek_param[g]) != 120:
        raise ValueError(f'Greek {g} missing values for some options')

def solve_problem():
    m = gp.Model('OptionHedging')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in option_keys}, ub={i: maxlong[i] for i in option_keys}, name='')
    z = m.addVars(option_keys, vtype=gp.GRB.INTEGER, name='')
    for i in option_keys:
        m.addConstr(z[i] >= x[i], name=f'z_ge_x_{i}')
        m.addConstr(z[i] >= -x[i], name=f'z_ge_negx_{i}')
    m.setObjective(gp.quicksum((cost[i] * z[i] for i in option_keys)), gp.GRB.MINIMIZE)
    for g in greek_names:
        expr = greek_initial[g] + gp.quicksum((greek_param[g][i] * A[i, j] * x[i] for i in option_keys for j in asset_keys))
        tol = greek_tol[g]
        m.addConstr(expr <= tol, name=f'{g}_upper')
        m.addConstr(expr >= -tol, name=f'{g}_lower')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')