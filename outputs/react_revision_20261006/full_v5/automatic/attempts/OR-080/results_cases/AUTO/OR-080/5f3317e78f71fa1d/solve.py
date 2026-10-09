import gurobipy as gp
import pandas as pd
import numpy as np
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
required_cols = ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
if df['truck_id'].duplicated().any():
    raise ValueError('Duplicate truck_id found in parameters.csv')
trucks = df['truck_id'].astype(int).tolist()
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].astype(float).to_dict()
S = df.set_index('truck_id')['S'].astype(float).to_dict()
C = df.set_index('truck_id')['C'].astype(float).to_dict()
demand = {1: float(df['d1'].iloc[0]), 2: float(df['d2'].iloc[0]), 3: float(df['d3'].iloc[0]), 4: float(df['d4'].iloc[0])}
m = gp.Model('TruckScheduling')
x_keys = [(i, t) for i in trucks for t in periods]
y_keys = [(i, t) for i in trucks for t in periods]
z_keys = [(i, t) for i in trucks for t in periods]
x = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
z = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in trucks:
    m.addConstr(z[i, 1] >= y[i, 1], name=f'startupdef_{i}_1')
    for t in [2, 3, 4]:
        m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startupdef_{i}_{t}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t] >= z[i, t], name=f'minup1_{i}_{t}')
        m.addConstr(y[i, t + 1] >= z[i, t], name=f'minup2_{i}_{t}')
for i in trucks:
    m.addConstr(z[i, 4] == 0, name=f'nostartup4_{i}')
for i in trucks:
    for t in [2, 3]:
        m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name=f'mindown_{i}_{t}')
for i in trucks:
    for t in [2, 3, 4]:
        m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for i in trucks:
        for t in periods:
            print(f'x[{i},{t}] {x[i, t].VarName} {x[i, t].X}')
    for i in trucks:
        for t in periods:
            print(f'y[{i},{t}] {y[i, t].VarName} {y[i, t].X}')
    for i in trucks:
        for t in periods:
            print(f'z[{i},{t}] {z[i, t].VarName} {z[i, t].X}')
else:
    print(f'Solver status: {m.status}')