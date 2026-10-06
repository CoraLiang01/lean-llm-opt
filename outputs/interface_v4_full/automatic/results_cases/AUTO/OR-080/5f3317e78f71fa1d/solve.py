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
    for t, dcol in zip(periods, d_cols):
        val = df.iloc[0][dcol]
        demand[t] = float(val)
    m = gp.Model('TruckScheduling')
    x = m.addVars(trucks, periods, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
    z = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'spare_cap_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_link_{i}_{t}')
            m.addConstr(x[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in trucks:
        for t in periods:
            y_prev = 0 if t == 1 else y[i, t - 1]
            m.addConstr(z[i, t] >= y[i, t] - y_prev, name=f'startup_lb_{i}_{t}')
            m.addConstr(z[i, t] <= y[i, t], name=f'startup_ub1_{i}_{t}')
            if t == 1:
                m.addConstr(z[i, t] <= 1, name=f'startup_ub2_{i}_{t}')
            else:
                m.addConstr(z[i, t] <= 1 - y[i, t - 1], name=f'startup_ub2_{i}_{t}')
        m.addConstr(z[i, 4] == 0, name=f'no_startup_4_{i}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y[i, t + 1] >= z[i, t], name=f'min_up_{i}_{t}')
    for i in trucks:
        for t in [1, 2]:
            y_prev = 0 if t == 1 else y[i, t - 1]
            m.addConstr(y_prev - y[i, t] <= 1 - y[i, t + 1], name=f'min_down_{i}_{t}')
            m.addConstr(z[i, t + 1] <= y[i, t], name=f'no_restart_{i}_{t + 1}')
    for i in trucks:
        for t in [2, 3, 4]:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'ramp_up_{i}_{t}')
            m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'ramp_down_{i}_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'zero_if_off_{i}_{t}')
    m.optimize()
    return m
m = solve_problem()