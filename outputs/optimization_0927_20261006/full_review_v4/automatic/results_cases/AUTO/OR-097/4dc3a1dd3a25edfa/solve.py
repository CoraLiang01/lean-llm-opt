import gurobipy as gp
import pandas as pd
import numpy as np
import re
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_char_df = pd.read_csv(opt_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = opt_char_df['Option'].tolist()
n_options = len(option_ids)
asset_cols = [col for col in asset_ref_df.columns if re.match('Asset_\\d+$', col)]
asset_ids = asset_cols
n_assets = len(asset_ids)

def to_float_series(df, col):
    return df.set_index('Option')[col].astype(float)

def to_int_series(df, col):
    return df.set_index('Option')[col].astype(int)
cost = to_float_series(opt_char_df, 'Cost').to_dict()
delta = to_float_series(opt_char_df, 'Delta').to_dict()
gamma = to_float_series(opt_char_df, 'Gamma').to_dict()
vega = to_float_series(opt_char_df, 'Vega').to_dict()
maxlong = to_int_series(opt_char_df, 'MaxLong').to_dict()
maxshort = to_int_series(opt_char_df, 'MaxShort').to_dict()
asset_ref_df = asset_ref_df.set_index('Unnamed: 0')
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(asset_ref_df.loc[opt, asset])
greeks = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[opt] for opt in option_ids], ub=[maxlong[opt] for opt in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
for greek in greeks:
    expr = greek_initial[greek]
    expr += gp.quicksum((greek_coeff[greek][opt] * A[opt][asset] * x_vars[opt] for opt in option_ids for asset in asset_ids))
    m.addConstr(expr <= greek_tol[greek], name=f'{greek}_plus')
    m.addConstr(expr >= -greek_tol[greek], name=f'{greek}_minus')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.optimize()