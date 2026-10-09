import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
params_df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in params_df.columns:
    raise KeyError("Missing 'truck_id' column in parameters.csv")
truck_ids = params_df['truck_id'].tolist()

def col_numeric(df, col):
    if col not in df.columns:
        raise KeyError(f"Missing '{col}' column in parameters.csv")
    return df[col].astype(float).to_dict()
Q_dict = col_numeric(params_df.set_index('truck_id'), 'Q')
S_dict = col_numeric(params_df.set_index('truck_id'), 'S')
C_dict = col_numeric(params_df.set_index('truck_id'), 'C')
demand_cols = ['d1', 'd2', 'd3', 'd4']
for col in demand_cols:
    if col not in params_df.columns:
        raise KeyError(f"Missing '{col}' column in parameters.csv")
periods = [1, 2, 3, 4]
demand_dict = {}
for (idx, dcol) in enumerate(demand_cols):
    demand_dict[periods[idx]] = float(params_df.iloc[0][dcol])
I = truck_ids
T = periods
m = gp.Model('TruckScheduling')
x_vars = m.addVars(I, T, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y_vars = m.addVars(I, T, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(I, T, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S_dict[i] * z_vars[i, t] for i in I for t in T)) + gp.quicksum((C_dict[i] * x_vars[i, t] for i in I for t in T)), gp.GRB.MINIMIZE)
for t in T:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) >= demand_dict[t], name=f'demand_{t}')
for t in T:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in I)), name=f'spare_cap_{t}')
for i in I:
    for t in T:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'cap_{i}_{t}')
for i in I:
    m.addConstr(z_vars[i, 1] == y_vars[i, 1], name=f'startup_init_{i}')
    for t in T[1:]:
        m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def_{i}_{t}')
for i in I:
    for t in [1, 2, 3]:
        m.addConstr(z_vars[i, t] <= y_vars[i, t], name=f'minup1_{i}_{t}')
        m.addConstr(z_vars[i, t] <= y_vars[i, t + 1], name=f'minup2_{i}_{t}')
    m.addConstr(z_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in I:
    for t in [1, 2, 3]:
        if t + 1 in T:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] + y_vars[i, t + 1] <= 1, name=f'mindown_{i}_{t}')
for i in I:
    m.addConstr(y_vars[i, 0] == 0, name=f'init_off_{i}')
    y_vars[i, 0] = 0
for i in I:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()