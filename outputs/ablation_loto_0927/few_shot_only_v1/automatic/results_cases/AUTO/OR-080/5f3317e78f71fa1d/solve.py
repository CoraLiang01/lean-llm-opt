import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
if 'truck_id' not in df.columns:
    raise KeyError("Missing 'truck_id' column in parameters.csv")
trucks = list(df['truck_id'])

def get_col(colname):
    if colname not in df.columns:
        raise KeyError(f"Missing '{colname}' column in parameters.csv")
    return dict(zip(df['truck_id'], df[colname]))
Q = get_col('Q')
S = get_col('S')
C = get_col('C')
periods = [1, 2, 3, 4]
for dcol in ['d1', 'd2', 'd3', 'd4']:
    if dcol not in df.columns:
        raise KeyError(f"Missing '{dcol}' column in parameters.csv")
demand = {1: float(df.iloc[0]['d1']), 2: float(df.iloc[0]['d2']), 3: float(df.iloc[0]['d3']), 4: float(df.iloc[0]['d4'])}
m = gp.Model('TruckScheduling')
x = m.addVars(trucks, periods, vtype=gp.GRB.CONTINUOUS, lb=0, name='')
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
        m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in trucks:
    m.addConstr(z[i, 1] == y[i, 1], name=f'startup1_{i}')
    for t in [2, 3, 4]:
        m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startupdiff_{i}_{t}')
        m.addConstr(z[i, t] <= 1, name=f'startupub_{i}_{t}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= z[i, t], name=f'minup_{i}_{t}')
for i in trucks:
    m.addConstr(z[i, 4] == 0, name=f'nostart4_{i}')
for i in trucks:
    for t in [2, 3]:
        m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name=f'mindown_{i}_{t}')
for i in trucks:
    for t in [2, 3, 4]:
        m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Truck Schedule ---')
    for i in trucks:
        print(f'Truck {i}:')
        for t in periods:
            print(f'  Period {t}: y={int(round(y[i, t].X))}, z={int(round(z[i, t].X))}, x={x[i, t].X:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')