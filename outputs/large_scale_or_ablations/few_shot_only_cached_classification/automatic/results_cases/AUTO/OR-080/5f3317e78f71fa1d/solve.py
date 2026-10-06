import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
trucks = df['truck_id'].astype(int).tolist()
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].astype(float).to_dict()
S = df.set_index('truck_id')['S'].astype(float).to_dict()
C = df.set_index('truck_id')['C'].astype(float).to_dict()
demand = {1: int(df['d1'].iloc[0]), 2: int(df['d2'].iloc[0]), 3: int(df['d3'].iloc[0]), 4: int(df['d4'].iloc[0])}
m = gp.Model('TruckScheduling')
x = m.addVars(trucks, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
z = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
for i in trucks:
    y_prev = 0
    for t in periods:
        m.addConstr(z[i, t] >= y[i, t] - y_prev, name=f'startup_{i}_{t}')
        y_prev = y[i, t]
    m.addConstr(z[i, 4] == 0, name=f'no_startup_last_{i}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= z[i, t], name=f'minup_{i}_{t}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t - 1] - y[i, t] + y[i, t + 1] <= 1, name=f'mindown_{i}_{t}')
for i in trucks:
    x_prev = 0.0
    for t in periods:
        m.addConstr(x[i, t] - x_prev <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x_prev - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
        x_prev = x[i, t]
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Truck Schedule ---')
    for i in trucks:
        print(f'Truck {i}:')
        for t in periods:
            yval = int(round(y[i, t].X))
            zval = int(round(z[i, t].X))
            xval = x[i, t].X
            print(f'  Period {t}: Active={yval}, Startup={zval}, Transported={xval:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')