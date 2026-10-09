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
greek_names = ['Delta', 'Gamma', 'Vega']
if not set(asset_ref_df['Unnamed: 0']) == set(option_ids):
    raise ValueError('Mismatch between OptionCharacteristics and Option_AssetReferenceMatrix option IDs')
option_to_idx = {opt: idx for (idx, opt) in enumerate(option_ids)}

def col_to_dict(df, key_col, val_col, dtype):
    s = df.set_index(key_col)[val_col]
    if dtype == float:
        return s.astype(float).to_dict()
    elif dtype == int:
        return s.astype(int).to_dict()
    else:
        return s.to_dict()
cost = col_to_dict(opt_df, 'Option', 'Cost', int)
delta = col_to_dict(opt_df, 'Option', 'Delta', float)
gamma = col_to_dict(opt_df, 'Option', 'Gamma', float)
vega = col_to_dict(opt_df, 'Option', 'Vega', float)
maxlong = col_to_dict(opt_df, 'Option', 'MaxLong', int)
maxshort = col_to_dict(opt_df, 'Option', 'MaxShort', int)
A = {}
for (_, row) in asset_ref_df.iterrows():
    opt = row['Unnamed: 0']
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(row[asset])
initial_greek = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance_greek = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}

def solve_problem():
    m = gp.Model('OptionHedging')
    x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=None, ub=None, name='')
    z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    for opt in option_ids:
        m.addConstr(x_vars[opt] >= maxshort[opt], name=f'min_{opt}')
        m.addConstr(x_vars[opt] <= maxlong[opt], name=f'max_{opt}')
    for opt in option_ids:
        m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
        m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
    for greek in greek_names:
        if greek == 'Delta':
            greek_vec = delta
        elif greek == 'Gamma':
            greek_vec = gamma
        elif greek == 'Vega':
            greek_vec = vega
        else:
            raise ValueError(f'Unknown greek {greek}')
        coeff = {}
        for opt in option_ids:
            coeff[opt] = 0.0
            for asset in asset_ids:
                coeff[opt] += greek_vec[opt] * A[opt][asset]
        expr = gp.quicksum((coeff[opt] * x_vars[opt] for opt in option_ids))
        m.addConstr(initial_greek[greek] + expr <= tolerance_greek[greek], name=f'{greek}_upper')
        m.addConstr(initial_greek[greek] + expr >= -tolerance_greek[greek], name=f'{greek}_lower')
    m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')