import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
opt_char_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
opt_ref_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
df_opt = pd.read_csv(opt_char_path, sep=',')
df_ref = pd.read_csv(opt_ref_path, sep=',')
df_ref = df_ref.rename(columns={'Unnamed: 0': 'Option'})
df_ref.set_index('Option', inplace=True)
options = df_opt['Option'].tolist()
cost = dict(zip(df_opt['Option'], df_opt['Cost']))
delta = dict(zip(df_opt['Option'], df_opt['Delta']))
gamma = dict(zip(df_opt['Option'], df_opt['Gamma']))
vega = dict(zip(df_opt['Option'], df_opt['Vega']))
max_long = dict(zip(df_opt['Option'], df_opt['MaxLong']))
max_short = dict(zip(df_opt['Option'], df_opt['MaxShort']))
asset_matrix = df_ref[['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4', 'Asset_5', 'Asset_6']].values
asset_count = {opt: int(asset_matrix[idx].sum()) for idx, opt in enumerate(options)}
delta_init = 0.25
gamma_init = 0.08
vega_init = 0.17
tol_delta = 0.06
tol_gamma = 0.05
tol_vega = 0.07
m = gp.Model('OptionHedging')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=lambda opt: max_short[opt], ub=lambda opt: max_long[opt])
u_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((cost[opt] * u_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstrs((u_vars[opt] >= x_vars[opt] for opt in options))
m.addConstrs((u_vars[opt] >= -x_vars[opt] for opt in options))
delta_expr = gp.quicksum((delta[opt] * asset_count[opt] * x_vars[opt] for opt in options))
m.addConstr(delta_init + delta_expr <= tol_delta)
m.addConstr(-(delta_init + delta_expr) <= tol_delta)
gamma_expr = gp.quicksum((gamma[opt] * asset_count[opt] * x_vars[opt] for opt in options))
m.addConstr(gamma_init + gamma_expr <= tol_gamma)
m.addConstr(-(gamma_init + gamma_expr) <= tol_gamma)
vega_expr = gp.quicksum((vega[opt] * asset_count[opt] * x_vars[opt] for opt in options))
m.addConstr(vega_init + vega_expr <= tol_vega)
m.addConstr(-(vega_init + vega_expr) <= tol_vega)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total hedging cost: {m.objVal:.2f}\n')
    print('Option positions (x) and absolute values (u):')
    for opt in options:
        x_val = x_vars[opt].X
        if abs(x_val) > 1e-06:
            print(f'  {opt}: x = {int(round(x_val))}, u = {int(round(u_vars[opt].X))}')
    final_delta = delta_init + sum((delta[opt] * asset_count[opt] * x_vars[opt].X for opt in options))
    final_gamma = gamma_init + sum((gamma[opt] * asset_count[opt] * x_vars[opt].X for opt in options))
    final_vega = vega_init + sum((vega[opt] * asset_count[opt] * x_vars[opt].X for opt in options))
    print('\nFinal exposures after hedging:')
    print(f'  Delta: {final_delta:.4f} (tolerance ±{tol_delta})')
    print(f'  Gamma: {final_gamma:.4f} (tolerance ±{tol_gamma})')
    print(f'  Vega:  {final_vega:.4f} (tolerance ±{tol_vega})')
else:
    print(f'No optimal solution found. Status: {m.status}')