import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = option_char_df['Option'].tolist()
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, found {len(option_ids)} in OptionCharacteristics.csv')
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
asset_ids = [col for col in asset_cols]
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, found {len(asset_ids)} in Option_AssetReferenceMatrix.csv')
asset_ref_df = asset_ref_df.rename(columns={asset_ref_df.columns[0]: 'Option'})
asset_ref_df['Option'] = asset_ref_df['Option'].astype(str)
if set(asset_ref_df['Option']) != set(option_ids):
    raise ValueError('Mismatch between OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv option identifiers')
A = {}
for (_, row) in asset_ref_df.iterrows():
    opt = row['Option']
    for asset in asset_ids:
        try:
            A[opt, asset] = int(row[asset])
        except Exception:
            raise ValueError(f'Non-integer value in Option_AssetReferenceMatrix for option {opt}, asset {asset}')

def parse_numeric_col(df, col, keylist, allow_negative=False):
    vals = {}
    for (idx, row) in df.iterrows():
        k = row['Option']
        v = row[col]
        try:
            f = float(v)
            if not allow_negative and f < 0:
                raise ValueError(f'Negative value in {col} for option {k}')
            vals[k] = f
        except Exception:
            raise ValueError(f'Non-numeric value in {col} for option {k}: {v}')
    if set(vals.keys()) != set(keylist):
        raise ValueError(f'Missing values in {col} for some options')
    return vals
Cost = parse_numeric_col(option_char_df, 'Cost', option_ids)
Delta = parse_numeric_col(option_char_df, 'Delta', option_ids, allow_negative=True)
Gamma = parse_numeric_col(option_char_df, 'Gamma', option_ids, allow_negative=True)
Vega = parse_numeric_col(option_char_df, 'Vega', option_ids, allow_negative=True)
MaxLong = parse_numeric_col(option_char_df, 'MaxLong', option_ids, allow_negative=True)
MaxShort = parse_numeric_col(option_char_df, 'MaxShort', option_ids, allow_negative=True)
GREEKS = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
Tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
Greek_dict = {'Delta': Delta, 'Gamma': Gamma, 'Vega': Vega}

def solve_problem():
    m = gp.Model('OptionHedging')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=None, ub=None, name='')
    z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    for i in option_ids:
        m.addConstr(x_vars[i] >= MaxShort[i], name=f'short_{i}')
        m.addConstr(x_vars[i] <= MaxLong[i], name=f'long_{i}')
    for i in option_ids:
        m.addConstr(z_vars[i] >= x_vars[i], name=f'zpos_{i}')
        m.addConstr(z_vars[i] >= -x_vars[i], name=f'zneg_{i}')
    for greek in GREEKS:
        greek_vec = Greek_dict[greek]
        expr = G_initial[greek]
        linexpr = gp.LinExpr()
        for i in option_ids:
            coeff = 0.0
            for j in asset_ids:
                coeff += greek_vec[i] * A[i, j]
            if abs(coeff) > 1e-12:
                linexpr.addTerms(coeff, x_vars[i])
        expr_total = expr + linexpr
        tol = Tolerance[greek]
        m.addConstr(expr_total <= tol, name=f'{greek}_ub')
        m.addConstr(expr_total >= -tol, name=f'{greek}_lb')
    obj = gp.quicksum((Cost[i] * z_vars[i] for i in option_ids))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')