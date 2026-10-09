import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, dtype=str, keep_default_na=False)
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    option_char_df[col] = option_char_df[col].astype(float if col in ['Delta', 'Gamma', 'Vega'] else int)
asset_ref_df = pd.read_csv(asset_ref_path, dtype=str, keep_default_na=False)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
for col in asset_cols:
    asset_ref_df[col] = asset_ref_df[col].astype(int)
option_ids = list(option_char_df['Option'])
asset_ids = asset_cols
asset_ref_df_indexed = asset_ref_df.set_index('Unnamed: 0')
missing_opts = set(option_ids) - set(asset_ref_df_indexed.index)
if missing_opts:
    raise ValueError(f'Options missing in asset reference matrix: {missing_opts}')
cost = {row['Option']: int(row['Cost']) for (_, row) in option_char_df.iterrows()}
delta = {row['Option']: float(row['Delta']) for (_, row) in option_char_df.iterrows()}
gamma = {row['Option']: float(row['Gamma']) for (_, row) in option_char_df.iterrows()}
vega = {row['Option']: float(row['Vega']) for (_, row) in option_char_df.iterrows()}
maxlong = {row['Option']: int(row['MaxLong']) for (_, row) in option_char_df.iterrows()}
maxshort = {row['Option']: int(row['MaxShort']) for (_, row) in option_char_df.iterrows()}
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(asset_ref_df_indexed.loc[opt, asset])
GREEKS = ['Delta', 'Gamma', 'Vega']
GREEK_INIT = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
GREEK_TOL = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
GREEK_COEF = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = Model('option_hedging')
m.Params.OutputFlag = 0
x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, name='')
z_vars = m.addVars(option_ids, vtype=GRB.CONTINUOUS, name='')
for opt in option_ids:
    m.addConstr(x_vars[opt] >= maxshort[opt], name=f'lb_{opt}')
    m.addConstr(x_vars[opt] <= maxlong[opt], name=f'ub_{opt}')
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
    m.addConstr(z_vars[opt] >= 0, name=f'abs3_{opt}')
for greek in GREEKS:
    greek_coef = GREEK_COEF[greek]
    greek_init = GREEK_INIT[greek]
    greek_tol = GREEK_TOL[greek]
    exposure_expr = quicksum((greek_coef[opt] * A[opt][asset] * x_vars[opt] for opt in option_ids for asset in asset_ids))
    m.addConstr(greek_init + exposure_expr <= greek_tol, name=f'{greek}_pos')
    m.addConstr(greek_init + exposure_expr >= -greek_tol, name=f'{greek}_neg')
m.setObjective(quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), GRB.MINIMIZE)
m.optimize()