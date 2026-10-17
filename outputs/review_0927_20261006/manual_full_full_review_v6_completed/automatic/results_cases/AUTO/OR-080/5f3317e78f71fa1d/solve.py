import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing required column 'truck_id' in parameters.csv")
truck_ids = df['truck_id'].astype(int).tolist()
n_trucks = len(truck_ids)
periods = [1, 2, 3, 4]

def col_to_dict(colname, dtype):
    if colname not in df.columns:
        raise KeyError(f"Missing required column '{colname}' in parameters.csv")
    return dict(zip(df['truck_id'].astype(int), df[colname].astype(dtype)))
Q = col_to_dict('Q', int)
S = col_to_dict('S', int)
C = col_to_dict('C', float)
demand = {}
for t, dcol in zip(periods, ['d1', 'd2', 'd3', 'd4']):
    if dcol not in df.columns:
        raise KeyError(f"Missing required column '{dcol}' in parameters.csv")
    demand[t] = int(df[dcol].iloc[0])
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            m.addConstr(s_vars[i, t] == y_vars[i, t], name=f'startup_{i}_{t}')
        else:
            m.addConstr(s_vars[i, t] == y_vars[i, t] - y_vars[i, t - 1], name=f'startup_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'minup_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'nostartup4_{i}')
for i in truck_ids:
    for t in [2, 3]:
        m.addConstr(y_vars[i, t - 1] - y_vars[i, t] + y_vars[i, t + 1] <= 1, name=f'mindown_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
m.optimize()