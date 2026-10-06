import gurobipy as gp
import pandas as pd
import numpy as np
import math

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',')
    required_cols = {'truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4'}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f'Missing columns in parameters.csv: {missing}')
    trucks = list(df['truck_id'])
    periods = [1, 2, 3, 4]
    Q = dict(zip(df['truck_id'], df['Q']))
    S = dict(zip(df['truck_id'], df['S']))
    C = dict(zip(df['truck_id'], df['C']))
    d_cols = ['d1', 'd2', 'd3', 'd4']
    for col in d_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f'Demand column {col} is not numeric.')
    demand = {t: float(df.iloc[0][f'd{t}']) for t in periods}
    for i in trucks:
        if i not in Q or i not in S or i not in C:
            raise ValueError(f'Missing Q, S, or C for truck {i}')
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in trucks for t in periods]
    y_keys = [(i, t) for i in trucks for t in periods]
    u_keys = [(i, t) for i in trucks for t in periods]
    x = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    u = m.addVars(u_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * u[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
            m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in trucks:
        m.addConstr(u[i, 1] == y[i, 1], name=f'startupdef_{i}_1')
        for t in [2, 3, 4]:
            m.addConstr(u[i, t] >= y[i, t] - y[i, t - 1], name=f'startupdef_{i}_{t}')
            m.addConstr(u[i, t] >= 0, name=f'startupnonneg_{i}_{t}')
    for i in trucks:
        m.addConstr(u[i, 4] == 0, name=f'nostartup4_{i}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y[i, t + 1] >= u[i, t], name=f'minup_{i}_{t}')
    for i in trucks:
        for t in [2, 3]:
            m.addConstr(y[i, t - 1] - y[i, t] - (1 - y[i, t + 1]) <= 1, name=f'mindown_{i}_{t}')
            m.addConstr(y[i, t + 1] <= 1 - (y[i, t - 1] - y[i, t]), name=f'mindown2_{i}_{t}')
    for i in trucks:
        for t in [2, 3, 4]:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')