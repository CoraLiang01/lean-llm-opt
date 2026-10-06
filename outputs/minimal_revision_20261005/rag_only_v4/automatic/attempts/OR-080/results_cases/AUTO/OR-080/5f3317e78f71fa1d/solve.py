import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',')
if df['truck_id'].isnull().any():
    raise ValueError('Missing truck_id in parameters.csv')
truck_ids = df['truck_id'].astype(int).tolist()
if len(set(truck_ids)) != len(truck_ids):
    raise ValueError('Duplicate truck_id in parameters.csv')
if len(truck_ids) != 10:
    raise ValueError('Expected 10 trucks as per query, got %d' % len(truck_ids))
periods = [1, 2, 3, 4]
Q = df.set_index('truck_id')['Q'].astype(int).to_dict()
S = df.set_index('truck_id')['S'].astype(int).to_dict()
C = df.set_index('truck_id')['C'].astype(float).to_dict()
demand = {}
for t in periods:
    col = f'd{t}'
    if col not in df.columns:
        raise ValueError(f'Missing demand column {col} in parameters.csv')
    vals = df[col].unique()
    if len(vals) != 1:
        raise ValueError(f'Demand column {col} has multiple values: {vals}')
    demand[t] = int(vals[0])

def solve_problem():
    m = gp.Model('truck_scheduling')
    I = truck_ids
    T = periods
    on = m.addVars(I, T, vtype=GRB.BINARY, name='')
    startup = m.addVars(I, T, vtype=GRB.BINARY, name='')
    shut = m.addVars(I, T, vtype=GRB.BINARY, name='')
    w = m.addVars(I, T, lb=0.0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((S[i] * startup[i, t] for i in I for t in T)) + gp.quicksum((C[i] * w[i, t] for i in I for t in T)), GRB.MINIMIZE)
    for t in T:
        m.addConstr(gp.quicksum((w[i, t] for i in I)) >= demand[t], name='demand_%d' % t)
    for t in T:
        m.addConstr(gp.quicksum((w[i, t] for i in I)) <= 0.9 * gp.quicksum((Q[i] * on[i, t] for i in I)), name='buffer_%d' % t)
    for i in I:
        for t in T:
            m.addConstr(w[i, t] <= Q[i] * on[i, t], name='cap_%d_%d' % (i, t))
            m.addConstr(w[i, t] >= 0, name='nonneg_%d_%d' % (i, t))
    for i in I:
        for t in T:
            prev_on = 0 if t == 1 else on[i, t - 1]
            m.addConstr(startup[i, t] >= on[i, t] - prev_on, name='startupdef_%d_%d' % (i, t))
    for i in I:
        m.addConstr(startup[i, 4] == 0, name='nostartup4_%d' % i)
    for i in I:
        for t in [1, 2, 3]:
            m.addConstr(on[i, t + 1] >= startup[i, t], name='minup_%d_%d' % (i, t))
    for i in I:
        for t in T:
            prev_on = 0 if t == 1 else on[i, t - 1]
            m.addConstr(shut[i, t] >= prev_on - on[i, t], name='shutdef_%d_%d' % (i, t))
    for i in I:
        for t in [1, 2]:
            if t + 1 in T:
                m.addConstr(on[i, t + 1] <= 1 - shut[i, t], name='mindown1_%d_%d' % (i, t))
            if t + 2 in T:
                m.addConstr(on[i, t + 2] <= 1 - shut[i, t], name='mindown2_%d_%d' % (i, t))
    for i in I:
        for t in T:
            prev_w = 0 if t == 1 else w[i, t - 1]
            m.addConstr(w[i, t] - prev_w <= 300, name='rampup_%d_%d' % (i, t))
            m.addConstr(w[i, t] - prev_w >= -300, name='rampdown_%d_%d' % (i, t))
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)