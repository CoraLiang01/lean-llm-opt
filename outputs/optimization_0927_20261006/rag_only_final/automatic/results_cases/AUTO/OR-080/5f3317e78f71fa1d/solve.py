import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    df[col] = df[col].astype(float if col == 'C' else int)
truck_ids = df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = {int(row['truck_id']): int(row['Q']) for (_, row) in df.iterrows()}
S = {int(row['truck_id']): int(row['S']) for (_, row) in df.iterrows()}
C = {int(row['truck_id']): float(row['C']) for (_, row) in df.iterrows()}
demand = {1: int(df.iloc[0]['d1']), 2: int(df.iloc[0]['d2']), 3: int(df.iloc[0]['d3']), 4: int(df.iloc[0]['d4'])}
m = Model('truck_scheduling')
y_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
u_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
x_vars = m.addVars(truck_ids, periods, vtype=GRB.CONTINUOUS, lb=0.0, name='')
startup_cost = quicksum((S[i] * u_vars[i, t] for i in truck_ids for t in periods))
transport_cost = quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, GRB.MINIMIZE)
for t in periods:
    m.addConstr(quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(u_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    m.addConstr(u_vars[i, 4] == 0, name=f'no_startup_last_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= u_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y_vars[i, t - 1]
        m.addConstr(z_vars[i, t] >= prev_y - y_vars[i, t], name=f'shutdown_logic_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t + 1 in periods:
            m.addConstr(y_vars[i, t + 1] <= 1 - z_vars[i, t], name=f'min_down1_{i}_{t}')
        if t + 2 in periods:
            m.addConstr(y_vars[i, t + 2] <= 1 - z_vars[i, t], name=f'min_down2_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t + 1 in periods:
            m.addConstr(u_vars[i, t + 1] <= 1 - z_vars[i, t], name=f'no_startup_after_shutdown_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
if set(truck_ids) != set(Q.keys()) or set(truck_ids) != set(S.keys()) or set(truck_ids) != set(C.keys()):
    raise ValueError('Mismatch in truck identifiers between parameter sets.')
if set(periods) != set(demand.keys()):
    raise ValueError('Mismatch in period identifiers between demand and periods.')
m.optimize()