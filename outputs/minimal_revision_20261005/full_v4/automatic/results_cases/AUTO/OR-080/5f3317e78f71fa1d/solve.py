import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',')
    trucks = df['truck_id'].astype(int).tolist()
    periods = [1, 2, 3, 4]
    Q = df.set_index('truck_id')['Q'].astype(float).to_dict()
    S = df.set_index('truck_id')['S'].astype(float).to_dict()
    C = df.set_index('truck_id')['C'].astype(float).to_dict()
    d_cols = ['d1', 'd2', 'd3', 'd4']
    demand = {}
    for (t, col) in enumerate(d_cols, 1):
        val = df.iloc[0][col]
        demand[t] = float(val)
    if not set(trucks) == set(Q.keys()) == set(S.keys()) == set(C.keys()):
        raise ValueError('Mismatch in truck_id coverage among Q, S, C.')
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in trucks for t in periods]
    y_keys = [(i, t) for i in trucks for t in periods]
    z_keys = [(i, t) for i in trucks for t in periods]
    x = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z[i, t] + C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
            m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in trucks:
        m.addConstr(z[i, 1] == y[i, 1], name=f'startupdef_{i}_1')
        for t in [2, 3, 4]:
            m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startupdef_lb_{i}_{t}')
            m.addConstr(z[i, t] <= y[i, t], name=f'startupdef_ub1_{i}_{t}')
            m.addConstr(z[i, t] <= 1 - y[i, t - 1], name=f'startupdef_ub2_{i}_{t}')
    for i in trucks:
        m.addConstr(z[i, 4] == 0, name=f'nostartup4_{i}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y[i, t + 1] >= z[i, t], name=f'minup_{i}_{t}')
    for i in trucks:
        for t in [1, 2]:
            if t + 1 in periods:
                m.addConstr(y[i, t + 1] <= 1 - (y[i, t - 1] - y[i, t]), name=f'mindown1_{i}_{t}')
            if t + 2 in periods:
                m.addConstr(y[i, t + 2] <= 1 - (y[i, t - 1] - y[i, t]), name=f'mindown2_{i}_{t}')
        t = 3
        if t + 1 in periods:
            m.addConstr(y[i, t + 1] <= 1 - (y[i, t - 1] - y[i, t]), name=f'mindown1_{i}_{t}')
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