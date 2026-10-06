import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
option_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
option_df = pd.read_csv(option_char_path, sep=',')
asset_ref_df = pd.read_csv(asset_ref_path, sep=',')
options = option_df['Option'].astype(str).tolist()
n_options = len(options)
asset_cols = [col for col in asset_ref_df.columns if col.startswith('Asset_')]
assets = asset_cols
n_assets = len(assets)
greeks = ['Delta', 'Gamma', 'Vega']
cost = option_df.set_index('Option')['Cost'].astype(float).to_dict()
delta = option_df.set_index('Option')['Delta'].astype(float).to_dict()
gamma = option_df.set_index('Option')['Gamma'].astype(float).to_dict()
vega = option_df.set_index('Option')['Vega'].astype(float).to_dict()
maxlong = option_df.set_index('Option')['MaxLong'].astype(int).to_dict()
maxshort = option_df.set_index('Option')['MaxShort'].astype(int).to_dict()
asset_ref_df['Unnamed: 0'] = asset_ref_df['Unnamed: 0'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
if set(option_df['Option']) != set(asset_ref_df['Unnamed: 0']):
    raise ValueError('Option identifiers do not match between OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv')
A = {}
for _, row in asset_ref_df.iterrows():
    opt = row['Unnamed: 0']
    for asset in assets:
        A[opt, asset] = int(row[asset])
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
G_tol = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}

def solve_problem():
    m = gp.Model('option_hedging')
    x = m.addVars(options, vtype=GRB.INTEGER, name='')
    z = m.addVars(options, vtype=GRB.CONTINUOUS, lb=0.0, name='')
    m.setObjective(gp.quicksum((cost[opt] * z[opt] for opt in options)), GRB.MINIMIZE)
    for opt in options:
        m.addConstr(z[opt] >= x[opt], name=f'abs1_{opt}')
        m.addConstr(z[opt] >= -x[opt], name=f'abs2_{opt}')
    for opt in options:
        m.addConstr(x[opt] >= maxshort[opt], name=f'min_{opt}')
        m.addConstr(x[opt] <= maxlong[opt], name=f'max_{opt}')
    for greek in greeks:
        if greek == 'Delta':
            G = delta
        elif greek == 'Gamma':
            G = gamma
        elif greek == 'Vega':
            G = vega
        else:
            raise ValueError(f'Unknown greek: {greek}')
        expr = gp.LinExpr()
        for opt in options:
            coeff = 0.0
            for asset in assets:
                coeff += G[opt] * A[opt, asset]
            expr += coeff * x[opt]
        m.addConstr(G_initial[greek] + expr <= G_tol[greek], name=f'{greek}_upper')
        m.addConstr(G_initial[greek] + expr >= -G_tol[greek], name=f'{greek}_lower')
    m.optimize()
    return m
m = solve_problem()