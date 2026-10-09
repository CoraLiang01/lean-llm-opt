import gurobipy as gp
import pandas as pd
import numpy as np
import re
optchar_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
optchar_df = pd.read_csv(optchar_path, sep=',', dtype=str, keep_default_na=False)
assetref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
assetref_df = pd.read_csv(assetref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = list(optchar_df['Option'])
if len(option_ids) != 120:
    raise ValueError(f'Expected 120 options, got {len(option_ids)}')
asset_cols = [col for col in assetref_df.columns if re.match('Asset_\\d+$', col)]
asset_ids = asset_cols
if len(asset_ids) != 6:
    raise ValueError(f'Expected 6 assets, got {len(asset_ids)}')

def col_to_dict(df, colname, keycol='Option', dtype=float):
    vals = df.set_index(keycol)[colname].astype(dtype)
    if vals.index.duplicated().any():
        raise ValueError(f'Duplicate Option IDs in {keycol}')
    return vals.to_dict()
cost = col_to_dict(optchar_df, 'Cost', dtype=float)
delta = col_to_dict(optchar_df, 'Delta', dtype=float)
gamma = col_to_dict(optchar_df, 'Gamma', dtype=float)
vega = col_to_dict(optchar_df, 'Vega', dtype=float)
maxlong = col_to_dict(optchar_df, 'MaxLong', dtype=int)
maxshort = col_to_dict(optchar_df, 'MaxShort', dtype=int)
assetref_df = assetref_df.rename(columns={assetref_df.columns[0]: 'Option'})
assetref_df = assetref_df.set_index('Option')
if not set(option_ids).issubset(set(assetref_df.index)):
    missing = set(option_ids) - set(assetref_df.index)
    raise ValueError(f'Options missing in asset reference matrix: {missing}')
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(assetref_df.loc[opt, asset])
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_coeff = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb={opt: maxshort[opt] for opt in option_ids}, ub={opt: maxlong[opt] for opt in option_ids}, name='')
y_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for opt in option_ids:
    m.addConstr(y_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(y_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
for greek in greek_names:
    expr = greek_initial[greek]
    for opt in option_ids:
        for asset in asset_ids:
            if A[opt][asset] != 0:
                expr += greek_coeff[greek][opt] * A[opt][asset] * x_vars[opt]
    m.addConstr(expr <= greek_tolerance[greek], name=f'{greek}_plus')
    m.addConstr(expr >= -greek_tolerance[greek], name=f'{greek}_minus')
m.setObjective(gp.quicksum((cost[opt] * y_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.optimize()