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
    y_prev = 0
    for t in periods:
        m.addConstr(u[i, t] >= y[i, t] - y_prev, name=f'startup_logic_{i}_{t}')
        y_prev = y[i, t]
    m.addConstr(u[i, 4] == 0, name=f'no_start_last_{i}')
for i in truck_ids:
    for t in [1, 2, 3]:
        y_prev = 0 if t == 1 else y[i, t - 1]
        m.addConstr(y[i, t + 1] >= y[i, t] - y_prev, name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        y_prev = 0 if t == 1 else y[i, t - 1]
        m.addConstr(y_prev - y[i, t] <= 1 - y[i, t + 1], name=f'min_down_{i}_{t}')
for i in truck_ids:
    x_prev = 0.0
    for t in periods:
        if t == 1:
            x_prev = 0.0
        else:
            m.addConstr(x[i, t] - x_prev <= 300, name=f'ramp_up_{i}_{t}')
            m.addConstr(x_prev - x[i, t] <= 300, name=f'ramp_down_{i}_{t}')
            x_prev = x[i, t]
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nTruck schedule (periods 1-4):')
    for i in truck_ids:
        print(f'Truck {i}:')
        for t in periods:
            y_val = int(round(y[i, t].X))
            u_val = int(round(u[i, t].X))
            x_val = x[i, t].X
            print(f'  Period {t}: Active={y_val}, Startup={u_val}, Transported={x_val:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')