import gurobipy as gp
import pandas as pd
import numpy as np
opt_char_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
df_opt = pd.read_csv(opt_char_path, sep=',')
asset_ref_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
df_asset = pd.read_csv(asset_ref_path, sep=',')
option_ids = df_opt['Option'].astype(str).tolist()
n_options = len(option_ids)
asset_cols = [col for col in df_asset.columns if col.startswith('Asset_')]
asset_ids = [col for col in asset_cols]
n_assets = len(asset_ids)
greek_names = ['Delta', 'Gamma', 'Vega']
df_asset['Option'] = df_asset['Unnamed: 0'].astype(str).str.strip()
if set(df_asset['Option']) != set(option_ids):
    raise ValueError('Option IDs in OptionCharacteristics.csv and Option_AssetReferenceMatrix.csv do not match.')
df_asset = df_asset.set_index('Option').loc[option_ids]
cost = df_opt.set_index('Option')['Cost'].astype(float).to_dict()
greek_param = {}
for g in greek_names:
    greek_param[g] = df_opt.set_index('Option')[g].astype(float).to_dict()
max_long = df_opt.set_index('Option')['MaxLong'].astype(int).to_dict()
max_short = df_opt.set_index('Option')['MaxShort'].astype(int).to_dict()
A = {}
for i in option_ids:
    A[i] = {}
    for j in asset_ids:
        A[i][j] = int(df_asset.loc[i, j])
initial_greek = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
tolerance_greek = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
m = gp.Model('OptionHedging')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, name='')
z = m.addVars(option_ids, vtype=gp.GRB.INTEGER, name='')
m.setObjective(gp.quicksum((cost[i] * z[i] for i in option_ids)), gp.GRB.MINIMIZE)
for i in option_ids:
    m.addConstr(x[i] >= max_short[i], name=f'minpos_{i}')
    m.addConstr(x[i] <= max_long[i], name=f'maxpos_{i}')
for i in option_ids:
    m.addConstr(z[i] >= x[i], name=f'z_ge_x_{i}')
    m.addConstr(z[i] >= -x[i], name=f'z_ge_negx_{i}')
    m.addConstr(z[i] >= 0, name=f'z_nonneg_{i}')
for g in greek_names:
    greek_expr = gp.LinExpr()
    for i in option_ids:
        for j in asset_ids:
            if A[i][j] != 0:
                greek_expr.add(greek_param[g][i] * A[i][j] * x[i])
    m.addConstr(initial_greek[g] + greek_expr <= tolerance_greek[g], name=f'{g}_upper')
    m.addConstr(initial_greek[g] + greek_expr >= -tolerance_greek[g], name=f'{g}_lower')
m.optimize()