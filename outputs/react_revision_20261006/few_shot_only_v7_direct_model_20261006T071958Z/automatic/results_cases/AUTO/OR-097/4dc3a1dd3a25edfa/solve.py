import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = list(opt_df['Option'])
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, found {len(option_ids)} in OptionCharacteristics.csv')
asset_cols = [col for col in asset_ref_df.columns if re.fullmatch('Asset_\\d+', col)]
if len(asset_cols) != 6:
    raise ValueError(f'Expected 6 asset columns, found {len(asset_cols)} in Option_AssetReferenceMatrix.csv')
asset_ids = asset_cols

def to_float_series(df, col, idx):
    try:
        return pd.Series(df[col].astype(float).values, index=idx)
    except Exception as e:
        raise ValueError(f"Column '{col}' in OptionCharacteristics.csv could not be converted to float: {e}")
cost = to_float_series(opt_df, 'Cost', option_ids)
delta = to_float_series(opt_df, 'Delta', option_ids)
gamma = to_float_series(opt_df, 'Gamma', option_ids)
vega = to_float_series(opt_df, 'Vega', option_ids)
maxlong = to_float_series(opt_df, 'MaxLong', option_ids)
maxshort = to_float_series(opt_df, 'MaxShort', option_ids)
if asset_ref_df.shape[0] != 120:
    raise ValueError(f'Expected 120 rows in Option_AssetReferenceMatrix.csv, found {asset_ref_df.shape[0]}')
if 'Option' in asset_ref_df.columns:
    asset_ref_df = asset_ref_df.set_index('Option').reindex(option_ids)
else:
    asset_ref_df = asset_ref_df.reset_index(drop=True)
    asset_ref_df.index = option_ids
A = {}
for i in option_ids:
    for j in asset_ids:
        try:
            val = asset_ref_df.loc[i, j]
            A[i, j] = int(float(val))
        except Exception as e:
            raise ValueError(f'Error reading A[{i},{j}] from Option_AssetReferenceMatrix.csv: {e}')
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greeks = ['Delta', 'Gamma', 'Vega']
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}

def solve_option_hedging():
    m = gp.Model('OptionHedge')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(option_ids, lb=None, ub=None, vtype=gp.GRB.INTEGER, name='')
    z_vars = m.addVars(option_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    for i in option_ids:
        m.addConstr(x_vars[i] <= maxlong[i], name=f'maxlong_{i}')
        m.addConstr(x_vars[i] >= maxshort[i], name=f'maxshort_{i}')
    for i in option_ids:
        m.addConstr(z_vars[i] >= x_vars[i], name=f'abs1_{i}')
        m.addConstr(z_vars[i] >= -x_vars[i], name=f'abs2_{i}')
    for G in greeks:
        expr = gp.LinExpr()
        for i in option_ids:
            coeff_i = greek_coeff[G][i]
            for j in asset_ids:
                if A[i, j] != 0:
                    expr.addTerms(coeff_i * A[i, j], x_vars[i])
        total_expr = initial_exposure[G] + expr
        tol = tolerance[G]
        m.addConstr(total_expr <= tol, name=f'{G}_upper')
        m.addConstr(total_expr >= -tol, name=f'{G}_lower')
    obj = gp.quicksum((cost[i] * z_vars[i] for i in option_ids))
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_option_hedging()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')