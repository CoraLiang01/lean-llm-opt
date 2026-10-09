import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
truck_ids = sorted(df['truck_id'].astype(int).unique())
if len(truck_ids) != 10 or not all((tid in truck_ids for tid in range(1, 11))):
    raise ValueError('Expected exactly 10 trucks with IDs 1..10 in parameters.csv')
periods = [1, 2, 3, 4]
Q = {int(row['truck_id']): int(row['Q']) for (_, row) in df.iterrows()}
S = {int(row['truck_id']): int(row['S']) for (_, row) in df.iterrows()}
C = {int(row['truck_id']): float(row['C']) for (_, row) in df.iterrows()}
demand_cols = ['d1', 'd2', 'd3', 'd4']
demands = [int(df.iloc[0][col]) for col in demand_cols]
d = {t: demands[t - 1] for t in periods}
m = Model('TruckScheduling')
on = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
startup = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
shut = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
w = m.addVars(truck_ids, periods, lb=0.0, vtype=GRB.CONTINUOUS, name='')
for t in periods:
    m.addConstr(quicksum((w[i, t] for i in truck_ids)) >= d[t], name=f'demand_{t}')
    m.addConstr(quicksum((w[i, t] for i in truck_ids)) <= 0.9 * quicksum((Q[i] * on[i, t] for i in truck_ids)), name=f'sparecap_{t}')
for i in truck_ids:
    for t in periods:
        m.addConstr(w[i, t] <= Q[i] * on[i, t], name=f'cap_{i}_{t}')
        m.addConstr(w[i, t] >= 0, name=f'nonnegw_{i}_{t}')
for i in truck_ids:
    for t in periods:
        on_prev = 0 if t == 1 else on[i, t - 1]
        m.addConstr(startup[i, t] >= on[i, t] - on_prev, name=f'startupdef_{i}_{t}')
        m.addConstr(shut[i, t] >= on_prev - on[i, t], name=f'shutdef_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t <= 3:
            m.addConstr(startup[i, t] <= on[i, t], name=f'minut_on1_{i}_{t}')
            m.addConstr(startup[i, t] <= on[i, t + 1], name=f'minut_on2_{i}_{t}')
        if t == 4:
            m.addConstr(startup[i, t] == 0, name=f'nostartup4_{i}')
for i in truck_ids:
    for t in periods:
        if t <= 2:
            m.addConstr(shut[i, t] <= 1 - on[i, t + 1], name=f'mindown1_{i}_{t}')
            m.addConstr(shut[i, t] <= 1 - on[i, t + 2], name=f'mindown2_{i}_{t}')
for i in truck_ids:
    for t in periods:
        if t == 1:
            w_prev = 0.0
        else:
            w_prev = w[i, t - 1]
        if t >= 2:
            m.addConstr(w[i, t] - w_prev <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(w_prev - w[i, t] <= 300, name=f'rampdown_{i}_{t}')
startup_cost = quicksum((S[i] * startup[i, t] for i in truck_ids for t in periods))
transport_cost = quicksum((C[i] * w[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, GRB.MINIMIZE)
m.optimize()