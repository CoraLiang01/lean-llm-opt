import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
truck_ids = df['truck_id'].astype(int).tolist()
num_trucks = len(truck_ids)
periods = [1, 2, 3, 4]
Q_dict = dict(zip(df['truck_id'].astype(int), df['Q'].astype(float)))
S_dict = dict(zip(df['truck_id'].astype(int), df['S'].astype(float)))
C_dict = dict(zip(df['truck_id'].astype(int), df['C'].astype(float)))
demand_cols = ['d1', 'd2', 'd3', 'd4']
demand = {}
for (t, col) in enumerate(demand_cols, 1):
    val = df.iloc[0][col]
    demand[t] = float(val)
for i in truck_ids:
    if i not in Q_dict or i not in S_dict or i not in C_dict:
        raise ValueError(f'Missing parameter for truck {i}')
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
z_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S_dict[i] * z_vars[i, t] for i in truck_ids for t in periods))
transp_cost = gp.quicksum((C_dict[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transp_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q_dict[i] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q_dict[i] * y_vars[i, t], name=f'cap_{i}_{t}')
for i in truck_ids:
    m.addConstr(z_vars[i, 1] == y_vars[i, 1], name=f'startup_init_{i}')
    for t in [2, 3, 4]:
        m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(z_vars[i, t] <= y_vars[i, t], name=f'minut_{i}_{t}_a')
        m.addConstr(z_vars[i, t] <= y_vars[i, t + 1], name=f'minut_{i}_{t}_b')
    m.addConstr(z_vars[i, 4] == 0, name=f'nostart4_{i}')
for i in truck_ids:
    for t in [2, 3, 4]:
        if t < 4:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'mindown1_{i}_{t}')
        if t < 3:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 2], name=f'mindown2_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(x_vars[i, t + 1] - x_vars[i, t] <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x_vars[i, t] - x_vars[i, t + 1] <= 300, name=f'rampdown_{i}_{t}')
m.optimize()