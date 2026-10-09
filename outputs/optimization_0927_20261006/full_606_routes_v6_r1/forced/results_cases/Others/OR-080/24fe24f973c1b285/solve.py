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

def get_numeric_col(colname, dtype):
    if colname not in df.columns:
        raise KeyError(f"Missing required column '{colname}' in parameters.csv")
    return df[colname].astype(dtype).values
Q_arr = get_numeric_col('Q', int)
S_arr = get_numeric_col('S', int)
C_arr = get_numeric_col('C', float)
Q = {truck_ids[i]: Q_arr[i] for i in range(n_trucks)}
S = {truck_ids[i]: S_arr[i] for i in range(n_trucks)}
C = {truck_ids[i]: C_arr[i] for i in range(n_trucks)}
periods = [1, 2, 3, 4]
demand_cols = ['d1', 'd2', 'd3', 'd4']
for col in demand_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in parameters.csv")
demand = {}
for (t, col) in enumerate(demand_cols, start=1):
    val = int(df.iloc[0][col])
    demand[t] = val
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S[i] * z_vars[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 1] == y_vars[i, 1], name=f'startupdef_{i}_1')
    for t in [2, 3, 4]:
        m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startupdef_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(z_vars[i, t] <= y_vars[i, t], name=f'minut_{i}_{t}_a')
        m.addConstr(z_vars[i, t] <= y_vars[i, t + 1], name=f'minut_{i}_{t}_b')
    m.addConstr(z_vars[i, 4] == 0, name=f'nostart4_{i}')
for i in truck_ids:
    for t in [2, 3, 4]:
        if t <= 3:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'mindowntime_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.addConstr(x_vars[i, 1] <= 300, name=f'rampstart_{i}_1')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'zeroifoff_{i}_{t}')
m.optimize()