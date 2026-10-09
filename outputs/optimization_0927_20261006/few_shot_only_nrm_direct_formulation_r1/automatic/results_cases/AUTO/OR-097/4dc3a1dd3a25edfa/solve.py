import gurobipy as gp
import pandas as pd
import numpy as np
import re
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_char_df = pd.read_csv(opt_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
if 'Option' not in opt_char_df.columns:
    raise KeyError("OptionCharacteristics.csv must have an 'Option' column.")
options = list(opt_char_df['Option'])
asset_cols = [col for col in asset_ref_df.columns if re.fullmatch('Asset_\\d+', col)]
assets = asset_cols
if asset_ref_df.shape[0] != len(options):
    raise ValueError('Option_AssetReferenceMatrix.csv must have the same number of rows as options.')
if asset_ref_df.index.size != len(options):
    raise ValueError('Mismatch in number of options between files.')

def to_float_series(df, col, idx):
    try:
        return pd.Series(df[col].astype(float).values, index=idx)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")

def to_int_series(df, col, idx):
    try:
        return pd.Series(df[col].astype(int).values, index=idx)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")
cost = to_float_series(opt_char_df, 'Cost', options).to_dict()
delta = to_float_series(opt_char_df, 'Delta', options).to_dict()
gamma = to_float_series(opt_char_df, 'Gamma', options).to_dict()
vega = to_float_series(opt_char_df, 'Vega', options).to_dict()
maxlong = to_int_series(opt_char_df, 'MaxLong', options).to_dict()
maxshort = to_int_series(opt_char_df, 'MaxShort', options).to_dict()
A = {}
for (i, opt) in enumerate(options):
    for asset in assets:
        val = asset_ref_df.iloc[i][asset]
        try:
            A[opt, asset] = int(val)
        except Exception:
            raise ValueError(f"Non-integer value '{val}' in asset reference matrix at option '{opt}', asset '{asset}'.")
greeks = ['Delta', 'Gamma', 'Vega']
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
G_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=None, ub=None, name='')
z_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in options:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
for opt in options:
    m.addConstr(x_vars[opt] >= maxshort[opt], name=f'min_{opt}')
    m.addConstr(x_vars[opt] <= maxlong[opt], name=f'max_{opt}')
for greek in greeks:
    coeff = G_coeff[greek]
    initial = G_initial[greek]
    tol = G_tolerance[greek]
    exposure_expr = gp.LinExpr()
    for opt in options:
        for asset in assets:
            if A[opt, asset] == 1:
                exposure_expr.addTerms(coeff[opt], x_vars[opt])
    m.addConstr(initial + exposure_expr <= tol, name=f'{greek}_upper')
    m.addConstr(initial + exposure_expr >= -tol, name=f'{greek}_lower')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
m.optimize()