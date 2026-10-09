import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
if 'truck_id' not in df.columns:
    raise KeyError("Missing required column 'truck_id' in parameters.csv")
truck_ids = df['truck_id'].astype(int).tolist()
n_trucks = len(truck_ids)
periods = [1, 2, 3, 4]

def get_numeric_col(colname, dtype):
    if colname not in df.columns:
        raise KeyError(f"Missing required column '{colname}' in parameters.csv")
    return dict(zip(df['truck_id'].astype(int), df[colname].astype(dtype)))
Q = get_numeric_col('Q', int)
S = get_numeric_col('S', int)
C = get_numeric_col('C', float)
demand = {}
for (t, dcol) in zip(periods, ['d1', 'd2', 'd3', 'd4']):
    if dcol not in df.columns:
        raise KeyError(f"Missing required column '{dcol}' in parameters.csv")
    demand[t] = int(df[dcol].iloc[0])
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'truck_cap_{i}_{t}')
for i in truck_ids:
    prev_y = 0
    for t in periods:
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - prev_y, name=f'startup_def_{i}_{t}')
        prev_y = y_vars[i, t]
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t] + y_vars[i, t + 1] >= 2 * s_vars[i, t], name=f'min_up_{i}_{t}')
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_4_{i}')
for i in truck_ids:
    for t in [2, 3]:
        m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'min_down_{i}_{t}')
for i in truck_ids:
    prev_x = 0
    for t in periods:
        if t == 1:
            m.addConstr(x_vars[i, 1] <= 300, name=f'ramp_up_{i}_1')
            m.addConstr(x_vars[i, 1] >= 0, name=f'ramp_down_{i}_1')
        else:
            m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
            m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Truck Schedule ---')
    for i in truck_ids:
        print(f'Truck {i}:')
        for t in periods:
            y = int(round(y_vars[i, t].X))
            s = int(round(s_vars[i, t].X))
            x = x_vars[i, t].X
            print(f'  Period {t}: Active={y}, Startup={s}, Transported={x:.1f} kg')
else:
    print(f'No optimal solution found. Status: {m.status}')