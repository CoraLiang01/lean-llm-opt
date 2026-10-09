import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing 'truck_id' column in parameters.csv")
truck_ids = df['truck_id'].astype(int).tolist()
n_trucks = len(truck_ids)
periods = [1, 2, 3, 4]
n_periods = len(periods)

def col_to_dict(colname, dtype=float):
    if colname not in df.columns:
        raise KeyError(f"Missing '{colname}' column in parameters.csv")
    return {int(row['truck_id']): dtype(row[colname]) for (_, row) in df.iterrows()}
Q = col_to_dict('Q', int)
S = col_to_dict('S', float)
C = col_to_dict('C', float)
demand = {}
for (t, dcol) in zip(periods, ['d1', 'd2', 'd3', 'd4']):
    if dcol not in df.columns:
        raise KeyError(f"Missing '{dcol}' column in parameters.csv")
    demand[t] = int(df.iloc[0][dcol])
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'buffer_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 1] == y_vars[i, 1], name=f'startup_init_{i}')
    for t in periods[1:]:
        m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def1_{i}_{t}')
        m.addConstr(z_vars[i, t] >= 0, name=f'startup_def2_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 4] == 0, name=f'no_startup_p4_{i}')
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= y_vars[i, t] - z_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t] + y_vars[i, t + 1] <= 1 + y_vars[i, t - 1], name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()