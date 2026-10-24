import pandas as pd
import gurobipy
from gurobipy import GRB
option_char_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv'
asset_ref_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv'
opt_df = pd.read_csv(option_char_path, dtype=str, keep_default_na=False)
asset_df = pd.read_csv(asset_ref_path, dtype=str, keep_default_na=False)
numeric_cols = ['Cost', 'Delta', 'Gamma', 'Vega', 'MaxLong', 'MaxShort']
for col in numeric_cols:
    opt_df[col] = pd.to_numeric(opt_df[col], errors='raise')
opt_ids = opt_df['Option'].tolist()
asset_ids = asset_df['Unnamed: 0'].tolist()
if set(opt_ids) != set(asset_ids):
    missing_in_opt = set(asset_ids) - set(opt_ids)
    missing_in_asset = set(opt_ids) - set(asset_ids)
    msg = []
    if missing_in_opt:
        msg.append(f'Options missing in characteristics: {missing_in_opt}')
    if missing_in_asset:
        msg.append(f'Options missing in asset reference: {missing_in_asset}')
    raise ValueError('; '.join(msg))
opt_df = opt_df.set_index('Option')
asset_df = asset_df.set_index('Unnamed: 0')
asset_cols = ['Asset_1', 'Asset_2', 'Asset_3', 'Asset_4', 'Asset_5', 'Asset_6']
asset_df[asset_cols] = asset_df[asset_cols].apply(pd.to_numeric, errors='raise')
asset_sums = asset_df[asset_cols].sum(axis=1)
Cost = opt_df['Cost'].to_dict()
Delta = opt_df['Delta'].to_dict()
Gamma = opt_df['Gamma'].to_dict()
Vega = opt_df['Vega'].to_dict()
MaxLong = opt_df['MaxLong'].to_dict()
MaxShort = opt_df['MaxShort'].to_dict()
weightDelta = {i: Delta[i] * asset_sums[i] for i in opt_ids}
weightGamma = {i: Gamma[i] * asset_sums[i] for i in opt_ids}
weightVega = {i: Vega[i] * asset_sums[i] for i in opt_ids}
Delta0 = 0.25
Gamma0 = 0.08
Vega0 = 0.17
T_Delta = 0.06
T_Gamma = 0.05
T_Vega = 0.07

def solve_problem():
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(opt_ids, vtype=GRB.INTEGER, name='')
    y = m.addVars(opt_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
    for i in opt_ids:
        x[i].lb = MaxShort[i]
        x[i].ub = MaxLong[i]
    for i in opt_ids:
        m.addConstr(y[i] >= x[i])
        m.addConstr(y[i] >= -x[i])
    expr_delta = gurobipy.quicksum((weightDelta[i] * x[i] for i in opt_ids))
    m.addConstr(Delta0 + expr_delta <= T_Delta)
    m.addConstr(Delta0 + expr_delta >= -T_Delta)
    expr_gamma = gurobipy.quicksum((weightGamma[i] * x[i] for i in opt_ids))
    m.addConstr(Gamma0 + expr_gamma <= T_Gamma)
    m.addConstr(Gamma0 + expr_gamma >= -T_Gamma)
    expr_vega = gurobipy.quicksum((weightVega[i] * x[i] for i in opt_ids))
    m.addConstr(Vega0 + expr_vega <= T_Vega)
    m.addConstr(Vega0 + expr_vega >= -T_Vega)
    m.setObjective(gurobipy.quicksum((Cost[i] * y[i] for i in opt_ids)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()