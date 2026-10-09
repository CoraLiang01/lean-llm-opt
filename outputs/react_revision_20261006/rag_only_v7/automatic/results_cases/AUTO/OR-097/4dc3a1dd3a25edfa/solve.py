import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    option_char_df[col] = pd.to_numeric(option_char_df[col], errors='raise')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
for col in asset_cols:
    asset_ref_df[col] = pd.to_numeric(asset_ref_df[col], errors='raise')
option_ids = option_char_df['Option'].tolist()
asset_ids = asset_cols
option_ids_set = set(option_ids)
asset_ref_option_ids = asset_ref_df['Unnamed: 0'].tolist()
asset_ref_option_ids_set = set(asset_ref_option_ids)
if option_ids_set != asset_ref_option_ids_set:
    raise ValueError('Mismatch between OptionCharacteristics and Option_AssetReferenceMatrix option IDs.')
option_to_assetref_idx = {oid: idx for (idx, oid) in enumerate(asset_ref_option_ids)}
cost = option_char_df.set_index('Option')['Cost'].to_dict()
delta = option_char_df.set_index('Option')['Delta'].to_dict()
gamma = option_char_df.set_index('Option')['Gamma'].to_dict()
vega = option_char_df.set_index('Option')['Vega'].to_dict()
maxlong = option_char_df.set_index('Option')['MaxLong'].to_dict()
maxshort = option_char_df.set_index('Option')['MaxShort'].to_dict()
A = {}
for oid in option_ids:
    row = asset_ref_df.loc[option_to_assetref_idx[oid]]
    for asset in asset_ids:
        A[oid, asset] = int(row[asset])
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_names = ['Delta', 'Gamma', 'Vega']

def solve_problem():
    m = gp.Model('option_hedging')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb={oid: maxshort[oid] for oid in option_ids}, ub={oid: maxlong[oid] for oid in option_ids}, name='')
    z_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb=0, name='')
    for oid in option_ids:
        m.addConstr(z_vars[oid] >= x_vars[oid], name=f'z_ge_x_{oid}')
        m.addConstr(z_vars[oid] >= -x_vars[oid], name=f'z_ge_negx_{oid}')
    for greek in greek_names:
        if greek == 'Delta':
            g_vec = delta
        elif greek == 'Gamma':
            g_vec = gamma
        elif greek == 'Vega':
            g_vec = vega
        else:
            raise ValueError(f'Unknown greek: {greek}')
        expr = greek_initial[greek]
        expr_var = gp.LinExpr()
        for oid in option_ids:
            for asset in asset_ids:
                if A[oid, asset] == 1:
                    expr_var += g_vec[oid] * x_vars[oid]
        total_expr = expr + expr_var
        tol = greek_tol[greek]
        m.addConstr(total_expr <= tol, name=f'{greek}_upper')
        m.addConstr(total_expr >= -tol, name=f'{greek}_lower')
    obj = gp.quicksum((cost[oid] * z_vars[oid] for oid in option_ids))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')