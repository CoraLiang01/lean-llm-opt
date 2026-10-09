import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    df[col] = df[col].str.strip()
    if col in ['C']:
        df[col] = df[col].astype(float)
    else:
        df[col] = df[col].astype(int)
truck_ids = df['truck_id'].tolist()
truck_ids = [int(tid) for tid in truck_ids]
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].to_dict()
S = df.set_index('truck_id')['S'].to_dict()
C = df.set_index('truck_id')['C'].to_dict()
demand = {1: int(df.iloc[0]['d1']), 2: int(df.iloc[0]['d2']), 3: int(df.iloc[0]['d3']), 4: int(df.iloc[0]['d4'])}
m = Model('Truck_Scheduling')
y_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
x_vars = m.addVars(truck_ids, periods, vtype=GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(quicksum((S[str(i)] * s_vars[i, t] + C[str(i)] * x_vars[i, t] for i in truck_ids for t in periods)), GRB.MINIMIZE)
for t in periods:
    m.addConstr(quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * quicksum((Q[str(i)] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[str(i)] * y_vars[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            m.addConstr(s_vars[i, 1] == y_vars[i, 1], name=f'startup_logic_{i}_1')
        else:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            m.addConstr(z_vars[i, 1] >= 0 - y_vars[i, 1], name=f'shutdown_logic_{i}_1')
        else:
            m.addConstr(z_vars[i, t] >= y_vars[i, t - 1] - y_vars[i, t], name=f'shutdown_logic_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        if t + 1 in periods:
            m.addConstr(y_vars[i, t + 1] <= 1 - z_vars[i, t], name=f'min_down1_{i}_{t}')
        if t + 2 in periods:
            m.addConstr(y_vars[i, t + 2] <= 1 - z_vars[i, t], name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()