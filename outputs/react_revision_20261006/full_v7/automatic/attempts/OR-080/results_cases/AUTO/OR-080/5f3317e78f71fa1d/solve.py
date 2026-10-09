import gurobipy as gp
import pandas as pd
import numpy as np
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
    df[col] = pd.to_numeric(df[col], errors='raise')
truck_ids = df['truck_id'].tolist()
truck_ids = [int(i) for i in truck_ids]
periods = [1, 2, 3, 4]
Q = dict(zip(df['truck_id'].astype(int), df['Q']))
S = dict(zip(df['truck_id'].astype(int), df['S']))
C = dict(zip(df['truck_id'].astype(int), df['C']))
demand = {1: int(df['d1'].iloc[0]), 2: int(df['d2'].iloc[0]), 3: int(df['d3'].iloc[0]), 4: int(df['d4'].iloc[0])}
if set(truck_ids) != set(Q.keys()) or set(truck_ids) != set(S.keys()) or set(truck_ids) != set(C.keys()):
    raise ValueError('Mismatch in truck_id coverage among Q, S, C.')

def solve_problem():
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in truck_ids for t in periods]
    y_keys = [(i, t) for i in truck_ids for t in periods]
    s_keys = [(i, t) for i in truck_ids for t in periods]
    x_vars = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    s_vars = m.addVars(s_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * s_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
    for i in truck_ids:
        for t in periods:
            m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
            m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in truck_ids:
        m.addConstr(s_vars[i, 1] == y_vars[i, 1], name=f'startup_def_{i}_1')
        for t in [2, 3, 4]:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_def_{i}_{t}')
            m.addConstr(s_vars[i, t] >= 0, name=f'startup_nonneg_{i}_{t}')
    for i in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'minup_{i}_{t}')
        m.addConstr(s_vars[i, 4] == 0, name=f'nostartup4_{i}')
    for i in truck_ids:
        for t in [2, 3]:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'mindown_{i}_{t}')
    for i in truck_ids:
        for t in [2, 3, 4]:
            m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.4f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')