import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',')
    required_cols = ['truck_id', 'Q', 'S', 'C']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in parameters.csv")
    truck_ids = list(df['truck_id'])
    if len(truck_ids) != 10:
        raise ValueError(f'Expected 10 trucks, found {len(truck_ids)} in parameters.csv')
    periods = [1, 2, 3, 4]
    demand = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
    Q = dict(zip(df['truck_id'], df['Q']))
    S = dict(zip(df['truck_id'], df['S']))
    C = dict(zip(df['truck_id'], df['C']))
    for i in truck_ids:
        if i not in Q or i not in S or i not in C:
            raise ValueError(f'Missing parameter for truck {i}')
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in truck_ids for t in periods]
    y_keys = [(i, t) for i in truck_ids for t in periods]
    z_keys = [(i, t) for i in truck_ids for t in periods]
    x = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name=f'sparecap_{t}')
    for i in truck_ids:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
            m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in truck_ids:
        m.addConstr(z[i, 1] >= y[i, 1], name=f'startup1_{i}')
        for t in [2, 3, 4]:
            m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startup_{i}_{t}')
    for i in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(z[i, t] <= y[i, t + 1], name=f'minup_{i}_{t}')
        m.addConstr(z[i, 4] == 0, name=f'nostart4_{i}')
    for i in truck_ids:
        for t in [2, 3]:
            m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name=f'mindown_{i}_{t}')
            m.addConstr(z[i, t + 1] <= 1 - (y[i, t - 1] - y[i, t]), name=f'norestart_{i}_{t + 1}')
    for i in truck_ids:
        for t in [2, 3, 4]:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')