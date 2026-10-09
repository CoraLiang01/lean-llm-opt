import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = optchar_df['Option'].tolist()
n_options = len(option_ids)
asset_columns = [col for col in assetref_df.columns if re.match('Asset_\\d+$', col)]
asset_ids = asset_columns
n_assets = len(asset_ids)
assetref_df = assetref_df.set_index(assetref_df.columns[0])
assetref_df.index = assetref_df.index.str.strip()
missing_opts = set(option_ids) - set(assetref_df.index)
if missing_opts:
    raise ValueError(f'Options missing in Option_AssetReferenceMatrix.csv: {missing_opts}')

def to_float_col(df, col):
    return df[col].astype(float).to_dict()

def to_int_col(df, col):
    return df[col].astype(int).to_dict()
cost = to_float_col(optchar_df.set_index('Option'), 'Cost')
delta = to_float_col(optchar_df.set_index('Option'), 'Delta')
gamma = to_float_col(optchar_df.set_index('Option'), 'Gamma')
vega = to_float_col(optchar_df.set_index('Option'), 'Vega')
maxlong = to_int_col(optchar_df.set_index('Option'), 'MaxLong')
maxshort = to_int_col(optchar_df.set_index('Option'), 'MaxShort')
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(assetref_df.loc[opt, asset])
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[opt] for opt in option_ids], ub=[maxlong[opt] for opt in option_ids], name='')
absx_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in option_ids:
    m.addConstr(absx_vars[opt] >= x_vars[opt], name=f'absx_pos_{opt}')
    m.addConstr(absx_vars[opt] >= -x_vars[opt], name=f'absx_neg_{opt}')
m.setObjective(gp.quicksum((cost[opt] * absx_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
for greek in greek_names:
    greek_sum = gp.LinExpr()
    for opt in option_ids:
        for asset in asset_ids:
            if A[opt][asset] != 0:
                greek_sum.addTerms(greek_coeff[greek][opt] * A[opt][asset], x_vars[opt])
    net_exposure = greek_initial[greek] + greek_sum
    m.addConstr(net_exposure <= greek_tol[greek], name=f'{greek}_plus')
    m.addConstr(net_exposure >= -greek_tol[greek], name=f'{greek}_minus')
m.optimize()