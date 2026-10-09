import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    df[col] = df[col].astype(float if col == 'C' else int)
truck_ids = df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = {int(row['truck_id']): int(row['Q']) for (_, row) in df.iterrows()}
S = {int(row['truck_id']): int(row['S']) for (_, row) in df.iterrows()}
C = {int(row['truck_id']): float(row['C']) for (_, row) in df.iterrows()}
demand = {1: int(df.iloc[0]['d1']), 2: int(df.iloc[0]['d2']), 3: int(df.iloc[0]['d3']), 4: int(df.iloc[0]['d4'])}
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
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
            m.addConstr(z_vars[i, t] == y_vars[i, t], name=f'startup_def_{i}_{t}')
        else:
            m.addConstr(z_vars[i, t] == y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= z_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in truck_ids:
    for t in [1, 2]:
        m.addConstr(y_vars[i, t + 1] <= 1 - (y_vars[i, t - 1] - y_vars[i, t]), name=f'min_down1_{i}_{t}')
        m.addConstr(y_vars[i, t + 2] <= 1 - (y_vars[i, t - 1] - y_vars[i, t]), name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()