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
demand = {1: int(df['d1'].iloc[0]), 2: int(df['d2'].iloc[0]), 3: int(df['d3'].iloc[0]), 4: int(df['d4'].iloc[0])}
m = gp.Model('TruckScheduling')
x = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z[i, t] + C[i] * x[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name=f'sparecap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    m.addConstr(z[i, 1] >= y[i, 1], name=f'startupdef_{i}_1')
    for t in [2, 3, 4]:
        m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startupdef_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= z[i, t], name=f'minut_{i}_{t}')
for i in truck_ids:
    m.addConstr(z[i, 4] == 0, name=f'nostart4_{i}')
for i in truck_ids:
    for t in [1, 2]:
        m.addConstr(y[i, t + 2] <= 1 - (y[i, t] - y[i, t + 1]), name=f'mindowntime_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3, 4]:
        if t == 1:
            m.addConstr(x[i, 1] <= 300, name=f'rampup_{i}_1')
            m.addConstr(x[i, 1] >= 0, name=f'rampdown_{i}_1')
        else:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Truck Schedule ---')
    for i in truck_ids:
        print(f'Truck {i}:')
        for t in periods:
            print(f'  Period {t}: y={int(round(y[i, t].X))}, z={int(round(z[i, t].X))}, x={x[i, t].X:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')