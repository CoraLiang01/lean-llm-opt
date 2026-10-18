import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_char_df = pd.read_csv(option_char_path, sep=',', dtype=str, keep_default_na=False)
option_char_df['Option'] = option_char_df['Option'].str.strip()
for col in ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']:
    option_char_df[col] = option_char_df[col].astype(float if col in ['Delta', 'Gamma', 'Vega'] else int)
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].str.strip()
option_ids = list(option_char_df['Option'])
asset_option_ids = list(asset_ref_df['Unnamed: 0'])
if set(option_ids) != set(asset_option_ids):
    raise ValueError('Mismatch between OptionCharacteristics and Option_AssetReferenceMatrix option IDs.')
option_char_df = option_char_df.set_index('Option').loc[option_ids]
asset_ref_df = asset_ref_df.set_index('Unnamed: 0').loc[option_ids]
asset_ids = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
cost = option_char_df['Cost'].to_dict()
delta = option_char_df['Delta'].to_dict()
gamma = option_char_df['Gamma'].to_dict()
vega = option_char_df['Vega'].to_dict()
maxlong = option_char_df['MaxLong'].to_dict()
maxshort = option_char_df['MaxShort'].to_dict()
A = {}
for opt in option_ids:
    for asset in asset_ids:
        A[opt, asset] = int(asset_ref_df.loc[opt, asset])
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_names = ['Delta', 'Gamma', 'Vega']
m = Model('option_hedging')
m.Params.OutputFlag = 0
x_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb={opt: int(maxshort[opt]) for opt in option_ids}, ub={opt: int(maxlong[opt]) for opt in option_ids}, name='')
z_vars = m.addVars(option_ids, vtype=GRB.INTEGER, lb=0, name='')
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs1_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs2_{opt}')
for greek in greek_names:
    if greek == 'Delta':
        g_vec = delta
    elif greek == 'Gamma':
        g_vec = gamma
    elif greek == 'Vega':
        g_vec = vega
    else:
        raise ValueError(f'Unknown Greek: {greek}')
    exposure_expr = quicksum((g_vec[opt] * A[opt, asset] * x_vars[opt] for opt in option_ids for asset in asset_ids))
    total_expr = initial_exposure[greek] + exposure_expr
    m.addConstr(total_expr <= tolerance[greek], name=f'{greek}_upper')
    m.addConstr(total_expr >= -tolerance[greek], name=f'{greek}_lower')
m.setObjective(quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), GRB.MINIMIZE)
m.optimize()