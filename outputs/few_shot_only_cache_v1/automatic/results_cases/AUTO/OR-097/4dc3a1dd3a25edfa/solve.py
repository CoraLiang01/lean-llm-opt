import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, dtype=str, keep_default_na=False)
asset_ref_df = pd.read_csv(asset_ref_path, dtype=str, keep_default_na=False)
option_ids = option_char_df['Option'].tolist()
n_options = len(option_ids)
if n_options != 120:
    raise ValueError(f'Expected 120 options, got {n_options}')
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
asset_ids = asset_cols
n_assets = len(asset_ids)
if n_assets != 6:
    raise ValueError(f'Expected 6 assets, got {n_assets}')
option_char_df['Option_norm'] = option_char_df['Option'].str.strip()
asset_ref_df['Option_norm'] = asset_ref_df['Unnamed: 0'].str.strip()
if not set(option_ids) <= set(asset_ref_df['Option_norm']):
    missing = set(option_ids) - set(asset_ref_df['Option_norm'])
    raise ValueError(f'Options missing in asset reference matrix: {missing}')

def to_float_series(df, col, idx_col):
    return pd.Series(df[col].astype(float).values, index=df[idx_col]).to_dict()

def to_int_series(df, col, idx_col):
    return pd.Series(df[col].astype(int).values, index=df[idx_col]).to_dict()
cost = to_float_series(option_char_df, 'Cost', 'Option')
delta = to_float_series(option_char_df, 'Delta', 'Option')
gamma = to_float_series(option_char_df, 'Gamma', 'Option')
vega = to_float_series(option_char_df, 'Vega', 'Option')
maxlong = to_int_series(option_char_df, 'MaxLong', 'Option')
maxshort = to_int_series(option_char_df, 'MaxShort', 'Option')
A = {}
for _, row in asset_ref_df.iterrows():
    opt = row['Option_norm']
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(row[asset])
greeks = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={i: maxshort[i] for i in option_ids}, ub={i: maxlong[i] for i in option_ids}, name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in option_ids:
    m.addConstr(z_vars[i] >= x_vars[i], name=f'abs_pos_{i}')
    m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs_neg_{i}')
for greek in greeks:
    expr = gp.LinExpr()
    for i in option_ids:
        for j in asset_ids:
            if A[i][j] != 0:
                expr.addTerms(greek_param[greek][i] * A[i][j], x_vars[i])
    expr_total = greek_initial[greek] + expr
    tol = greek_tol[greek]
    m.addConstr(expr_total <= tol, name=f'{greek}_upper')
    m.addConstr(expr_total >= -tol, name=f'{greek}_lower')
m.setObjective(gp.quicksum((cost[i] * z_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.optimize()