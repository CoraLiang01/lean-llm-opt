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
asset_columns = [col for col in assetref_df.columns if col != 'Unnamed: 0']
asset_ids = asset_columns
n_assets = len(asset_ids)
greek_names = ['Delta', 'Gamma', 'Vega']
greek_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
greek_tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
required_optchar_cols = ['Option', 'Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']
for col in required_optchar_cols:
    if col not in optchar_df.columns:
        raise KeyError(f"Missing required column '{col}' in OptionCharacteristics.csv")
optchar_df['Cost'] = optchar_df['Cost'].astype(float)
optchar_df['Delta'] = optchar_df['Delta'].astype(float)
optchar_df['Gamma'] = optchar_df['Gamma'].astype(float)
optchar_df['Vega'] = optchar_df['Vega'].astype(float)
optchar_df['MaxLong'] = optchar_df['MaxLong'].astype(int)
optchar_df['MaxShort'] = optchar_df['MaxShort'].astype(int)
cost = dict(zip(optchar_df['Option'], optchar_df['Cost']))
delta = dict(zip(optchar_df['Option'], optchar_df['Delta']))
gamma = dict(zip(optchar_df['Option'], optchar_df['Gamma']))
vega = dict(zip(optchar_df['Option'], optchar_df['Vega']))
maxlong = dict(zip(optchar_df['Option'], optchar_df['MaxLong']))
maxshort = dict(zip(optchar_df['Option'], optchar_df['MaxShort']))
assetref_df = assetref_df.set_index('Unnamed: 0')
if not set(option_ids).issubset(set(assetref_df.index)):
    missing = set(option_ids) - set(assetref_df.index)
    raise KeyError(f'Option(s) missing in Option_AssetReferenceMatrix.csv: {missing}')
A = {}
for opt in option_ids:
    A[opt] = {}
    for asset in asset_ids:
        val = assetref_df.loc[opt, asset]
        try:
            A[opt][asset] = int(val)
        except Exception:
            raise ValueError(f"Invalid value for A[{opt}][{asset}]: '{val}'")
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[opt] for opt in option_ids], ub=[maxlong[opt] for opt in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs_pos_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs_neg_{opt}')
for greek in greek_names:
    if greek == 'Delta':
        greek_vec = delta
    elif greek == 'Gamma':
        greek_vec = gamma
    elif greek == 'Vega':
        greek_vec = vega
    else:
        raise ValueError(f'Unknown Greek: {greek}')
    greek_contrib = {}
    for opt in option_ids:
        asset_sum = sum((A[opt][asset] for asset in asset_ids))
        greek_contrib[opt] = greek_vec[opt] * asset_sum
    expr = greek_initial[greek] + gp.quicksum((greek_contrib[opt] * x_vars[opt] for opt in option_ids))
    m.addConstr(expr <= greek_tolerance[greek], name=f'{greek}_upper')
    m.addConstr(expr >= -greek_tolerance[greek], name=f'{greek}_lower')
m.optimize()