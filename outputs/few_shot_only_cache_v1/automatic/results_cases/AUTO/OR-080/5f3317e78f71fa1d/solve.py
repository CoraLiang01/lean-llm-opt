import gurobipy as gp
import pandas as pd
import numpy as np
import re
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
param_df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']
for col in required_cols:
    if col not in param_df.columns:
        raise KeyError(f"Missing required column '{col}' in parameters.csv")
truck_ids = param_df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = {row['truck_id']: float(row['Q']) for _, row in param_df.iterrows()}
S = {row['truck_id']: float(row['S']) for _, row in param_df.iterrows()}
C = {row['truck_id']: float(row['C']) for _, row in param_df.iterrows()}
demand = {}
for t in periods:
    col = f'd{t}'
    demand[t] = float(param_df.iloc[0][col])
m = gp.Model('TruckScheduling')
x_vars = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
s_vars = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
startup_cost = gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'spare_capacity_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'truck_cap_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 1] == y_vars[i, 1], name=f'startup_init_{i}')
    for t in periods[1:]:
        m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_lb_{i}_{t}')
        m.addConstr(s_vars[i, t] <= 1 - y_vars[i, t - 1], name=f'startup_ub1_{i}_{t}')
        m.addConstr(s_vars[i, t] <= y_vars[i, t], name=f'startup_ub2_{i}_{t}')
for i in truck_ids:
    m.addConstr(s_vars[i, 4] == 0, name=f'no_startup_last_{i}')
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'min_up_{i}_{t}')
for i in truck_ids:
    for t in [2, 3]:
        m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t] - y_vars[i, t + 1], name=f'min_down_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'ramp_down_{i}_{t}')
m.optimize()