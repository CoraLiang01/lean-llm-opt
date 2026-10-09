import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
df['truck_id'] = df['truck_id'].str.strip()
truck_ids = df['truck_id'].astype(int).tolist()
periods = [1, 2, 3, 4]
Q_dict = dict(zip(df['truck_id'].astype(int), df['Q'].astype(int)))
S_dict = dict(zip(df['truck_id'].astype(int), df['S'].astype(int)))
C_dict = dict(zip(df['truck_id'].astype(int), df['C'].astype(float)))
demand = {}
for t in periods:
    col = f'd{t}'
    if col not in df.columns:
        raise KeyError(f'Missing demand column {col} in parameters.csv')
    demand[t] = int(df.iloc[0][col])
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S_dict[i] * z_vars[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C_dict[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'truck_cap_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            m.addConstr(z_vars[i, t] == y_vars[i, t], name=f'startup_logic_{i}_{t}')
        else:
            m.addConstr(z_vars[i, t] == y_vars[i, t] - y_vars[i, t - 1], name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 4] == 0, name=f'no_startup_p4_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= z_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        shut = m.addVar(vtype=gp.GRB.BINARY, name=f'shut_{i}_{t}')
        m.addConstr(shut >= y_vars[i, t - 1] - y_vars[i, t], name=f'shut_def1_{i}_{t}')
        m.addConstr(shut <= y_vars[i, t - 1], name=f'shut_def2_{i}_{t}')
        m.addConstr(shut <= 1 - y_vars[i, t], name=f'shut_def3_{i}_{t}')
        if t + 1 <= 4:
            m.addConstr(y_vars[i, t + 1] <= 1 - shut, name=f'min_down1_{i}_{t}')
        if t + 2 <= 4:
            m.addConstr(y_vars[i, t + 2] <= 1 - shut, name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()