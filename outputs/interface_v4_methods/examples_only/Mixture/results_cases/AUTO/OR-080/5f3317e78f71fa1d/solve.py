import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',')
    truck_ids = df['truck_id'].astype(int).tolist()
    n_trucks = len(truck_ids)
    periods = [1, 2, 3, 4]
    Q = df.set_index('truck_id')['Q'].astype(float).to_dict()
    S = df.set_index('truck_id')['S'].astype(float).to_dict()
    C = df.set_index('truck_id')['C'].astype(float).to_dict()
    d = {1: float(df['d1'].iloc[0]), 2: float(df['d2'].iloc[0]), 3: float(df['d3'].iloc[0]), 4: float(df['d4'].iloc[0])}
    m = gp.Model('Truck_Scheduling')
    x = m.addVars(truck_ids, periods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(truck_ids, periods, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= d[t], name=f'demand_{t}')
        m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name=f'spare_cap_{t}')
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
            m.addConstr(y[i, t + 1] >= z[i, t], name=f'min_up_{i}_{t}')
        m.addConstr(z[i, 4] == 0, name=f'no_startup4_{i}')
    for i in truck_ids:
        for t in [2, 3]:
            m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name=f'min_down_{i}_{t}')
    for i in truck_ids:
        m.addConstr(x[i, 1] <= 300, name=f'ramp_up_{i}_1')
        m.addConstr(x[i, 1] >= 0, name=f'ramp_down_{i}_1')
        for t in [2, 3, 4]:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
            m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'ramp_down_{i}_{t}')
    m.optimize()
    return m
m = solve_problem()