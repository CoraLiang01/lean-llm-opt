import gurobipy as gp
import pandas as pd
import numpy as np
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
truck_ids = df['truck_id'].astype(int).tolist()
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].astype(float).to_dict()
S = df.set_index('truck_id')['S'].astype(float).to_dict()
C = df.set_index('truck_id')['C'].astype(float).to_dict()
demand = {1: float(df.iloc[0]['d1']), 2: float(df.iloc[0]['d2']), 3: float(df.iloc[0]['d3']), 4: float(df.iloc[0]['d4'])}
if set(Q.keys()) != set(truck_ids) or set(S.keys()) != set(truck_ids) or set(C.keys()) != set(truck_ids):
    raise ValueError('Mismatch in truck_id coverage for Q, S, or C.')
if set(demand.keys()) != set(periods):
    raise ValueError('Mismatch in period coverage for demand.')
m = gp.Model('TruckScheduling')
x = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
u = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S[i] * u[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C[i] * x[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_y = 0 if t == 1 else y[i, t - 1]
        m.addConstr(u[i, t] >= y[i, t] - prev_y, name=f'startup_logic_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= u[i, t], name=f'min_up_{i}_{t}')
    m.addConstr(u[i, 4] == 0, name=f'no_startup_4_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        prev_y = 0 if t == 1 else y[i, t - 1]
        m.addConstr(y[i, t + 1] <= 1 - prev_y + y[i, t], name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        prev_x = 0 if t == 2 else x[i, t - 1]
        m.addConstr(x[i, t] - (x[i, t - 1] if t > 1 else 0) <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr((x[i, t - 1] if t > 1 else 0) - x[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')