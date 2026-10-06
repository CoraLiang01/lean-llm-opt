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
    demand = {t: float(df.iloc[0][d_cols[t - 1]]) for t in periods}
    m = gp.Model('TruckScheduling')
    x = m.addVars(trucks, periods, name='x', lb=0.0)
    y = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='y')
    z = m.addVars(trucks, periods, vtype=gp.GRB.BINARY, name='z')
    m.setObjective(gp.quicksum((S[i] * z[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'spare_cap_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_link_{i}_{t}')
            m.addConstr(x[i, t] >= 0, name=f'x_nonneg_{i}_{t}')
    for i in trucks:
        m.addConstr(z[i, 1] == y[i, 1], name=f'startup_link_{i}_1')
        for t in [2, 3, 4]:
            m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1], name=f'startup_link_{i}_{t}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y[i, t + 1] >= z[i, t], name=f'min_up_{i}_{t}')
    for i in trucks:
        m.addConstr(z[i, 4] == 0, name=f'no_startup_4_{i}')
    for i in trucks:
        for t in [1, 2]:
            m.addConstr(y[i, t] - y[i, t + 1] + y[i, t + 2] <= 1, name=f'min_down_{i}_{t}')
    for i in trucks:
        m.addConstr(x[i, 1] <= 300, name=f'ramp_up_{i}_1')
        for t in [1, 2, 3]:
            m.addConstr(x[i, t + 1] - x[i, t] <= 300, name=f'ramp_up_{i}_{t + 1}')
            m.addConstr(x[i, t] - x[i, t + 1] <= 300, name=f'ramp_down_{i}_{t + 1}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'x_zero_if_off_{i}_{t}')
    m.optimize()
    return m
m = solve_problem()