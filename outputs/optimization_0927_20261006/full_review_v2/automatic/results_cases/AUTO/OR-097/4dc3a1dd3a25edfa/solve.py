import gurobipy as gp
import pandas as pd
import numpy as np
import re
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_char_df = pd.read_csv(opt_char_path, sep=',', dtype=str, keep_default_na=False)
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
asset_ref_df = pd.read_csv(asset_ref_path, sep=',', dtype=str, keep_default_na=False)
option_ids = opt_char_df['Option'].str.strip()
if option_ids.duplicated().any():
    raise ValueError('Duplicate Option identifiers found in OptionCharacteristics.csv')
option_ids = list(option_ids)
asset_id_cols = [col for col in asset_ref_df.columns if col != 'Unnamed: 0']
asset_ids = [col.strip() for col in asset_id_cols]
cost = dict(zip(option_ids, opt_char_df['Cost'].astype(float)))
delta = dict(zip(option_ids, opt_char_df['Delta'].astype(float)))
gamma = dict(zip(option_ids, opt_char_df['Gamma'].astype(float)))
vega = dict(zip(option_ids, opt_char_df['Vega'].astype(float)))
maxlong = dict(zip(option_ids, opt_char_df['MaxLong'].astype(int)))
maxshort = dict(zip(option_ids, opt_char_df['MaxShort'].astype(int)))
asset_ref_df['Option'] = asset_ref_df['Unnamed: 0'].str.strip()
if not set(asset_ref_df['Option']) == set(option_ids):
    raise ValueError('Mismatch between Option IDs in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv')
A = {}
for (_, row) in asset_ref_df.iterrows():
    opt = row['Option']
    A[opt] = {}
    for asset in asset_ids:
        A[opt][asset] = int(row[asset])
greeks = ['Delta', 'Gamma', 'Vega']
initial_exposure = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
greek_param = {'Delta': delta, 'Gamma': gamma, 'Vega': vega}
m = gp.Model('OptionHedging')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=[maxshort[opt] for opt in option_ids], ub=[maxlong[opt] for opt in option_ids], name='')
z_vars = m.addVars(option_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((cost[opt] * z_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
for opt in option_ids:
    m.addConstr(z_vars[opt] >= x_vars[opt], name=f'abs_pos_{opt}')
    m.addConstr(z_vars[opt] >= -x_vars[opt], name=f'abs_neg_{opt}')
for greek in greeks:
    asset_counts = {opt: sum((A[opt][asset] for asset in asset_ids)) for opt in option_ids}
    expr = gp.LinExpr()
    for opt in option_ids:
        coeff = greek_param[greek][opt] * asset_counts[opt]
        expr.addTerms(coeff, x_vars[opt])
    m.addConstr(initial_exposure[greek] + expr <= tolerance[greek], name=f'{greek}_upper')
    m.addConstr(initial_exposure[greek] + expr >= -tolerance[greek], name=f'{greek}_lower')
m.optimize()