import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
truck_ids = sorted(df['truck_id'].astype(int).unique())
if truck_ids != list(range(1, 11)):
    raise ValueError('Truck IDs in CSV do not match required set 1..10.')
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].astype(int).to_dict()
S = df.set_index('truck_id')['S'].astype(int).to_dict()
C = df.set_index('truck_id')['C'].astype(float).to_dict()
demand_cols = ['d1', 'd2', 'd3', 'd4']
demand = {}
for t, col in enumerate(demand_cols, 1):
    val = df.iloc[0][col]
    if not np.all(df[col] == val):
        raise ValueError(f'Demand column {col} is not constant across trucks.')
    demand[t] = int(val)
m = Model('truck_scheduling')
on = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
startup = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
x = m.addVars(truck_ids, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
for t in periods:
    m.addConstr(quicksum((x[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(quicksum((x[i, t] for i in truck_ids)) <= 0.9 * quicksum((Q[i] * on[i, t] for i in truck_ids)), name=f'buffer_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * on[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
for i in truck_ids:
    for t in periods:
        prev_on = 0 if t == 1 else on[i, t - 1]
        m.addConstr(startup[i, t] >= on[i, t] - prev_on, name=f'startupdef_{i}_{t}')
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(on[i, t + 1] >= startup[i, t], name=f'minut_{i}_{t}')
for i in truck_ids:
    m.addConstr(startup[i, 4] == 0, name=f'nostart4_{i}')
for i in truck_ids:
    for t in [2, 3]:
        m.addConstr(on[i, t - 1] - on[i, t] + on[i, t + 1] <= 1, name=f'mindowntime_{i}_{t}')
for i in truck_ids:
    for t in [2, 3, 4]:
        prev_x = 0 if t == 1 else x[i, t - 1]
        m.addConstr(x[i, t] - prev_x <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(prev_x - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
startup_cost = quicksum((S[i] * startup[i, t] for i in truck_ids for t in periods))
transport_cost = quicksum((C[i] * x[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, GRB.MINIMIZE)
m.optimize()