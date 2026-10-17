import gurobipy as gp
import pandas as pd
import numpy as np
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in parameters.csv")
    df[col] = df[col].str.strip()
    if col == 'truck_id':
        df[col] = df[col].astype(int)
    elif col in ['Q', 'S', 'd1', 'd2', 'd3', 'd4']:
        df[col] = df[col].astype(int)
    elif col == 'C':
        df[col] = df[col].astype(float)
truck_ids = df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].to_dict()
S = df.set_index('truck_id')['S'].to_dict()
C = df.set_index('truck_id')['C'].to_dict()
demand = {1: int(df.iloc[0]['d1']), 2: int(df.iloc[0]['d2']), 3: int(df.iloc[0]['d3']), 4: int(df.iloc[0]['d4'])}
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'truck_cap_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_def_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in truck_ids:
    for t in [1, 2]:
        m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1] if t > 1 else 0 - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()