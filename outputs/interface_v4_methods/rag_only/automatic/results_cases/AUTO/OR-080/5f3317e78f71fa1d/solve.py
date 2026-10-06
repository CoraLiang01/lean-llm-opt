import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',')
    trucks = df['truck_id'].astype(int).tolist()
    n_trucks = len(trucks)
    periods = [1, 2, 3, 4]
    n_periods = len(periods)
    Q = df.set_index('truck_id')['Q'].astype(int).to_dict()
    S = df.set_index('truck_id')['S'].astype(int).to_dict()
    C = df.set_index('truck_id')['C'].astype(float).to_dict()
    d_cols = ['d1', 'd2', 'd3', 'd4']
    demand = {}
    for idx, t in enumerate(periods):
        demand[t] = int(df.iloc[0][d_cols[idx]])
    m = gp.Model('truck_scheduling')
    on = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    startup = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    shut = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    w = m.addVars(trucks, periods, lb=0.0, vtype=GRB.CONTINUOUS, name='')
    on0 = {i: 0 for i in trucks}
    w0 = {i: 0.0 for i in trucks}
    m.setObjective(gp.quicksum((S[i] * startup[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * w[i, t] for i in trucks for t in periods)), GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((w[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((w[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * on[i, t] for i in trucks)), name=f'sparecap_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(w[i, t] <= Q[i] * on[i, t], name=f'cap_{i}_{t}')
            m.addConstr(w[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in trucks:
        for t in periods:
            prev_on = on0[i] if t == 1 else on[i, t - 1]
            m.addConstr(startup[i, t] >= on[i, t] - prev_on, name=f'startupdef_{i}_{t}')
    for i in trucks:
        for t in periods:
            if t < periods[-1]:
                m.addConstr(on[i, t + 1] >= startup[i, t], name=f'minup_{i}_{t}')
            else:
                m.addConstr(startup[i, t] == 0, name=f'nostartup_last_{i}')
    for i in trucks:
        for t in periods:
            prev_on = on0[i] if t == 1 else on[i, t - 1]
            m.addConstr(shut[i, t] >= prev_on - on[i, t], name=f'shutdef_{i}_{t}')
    for i in trucks:
        for t in periods:
            if t <= periods[-2]:
                if t + 1 in periods:
                    m.addConstr(on[i, t + 1] <= 1 - shut[i, t], name=f'mindown1_{i}_{t}')
                if t + 2 in periods:
                    m.addConstr(on[i, t + 2] <= 1 - shut[i, t], name=f'mindown2_{i}_{t}')
    for i in trucks:
        for t in periods:
            if t > 1:
                prev_w = w0[i] if t == 2 else w[i, t - 1]
                m.addConstr(w[i, t] - prev_w <= 300, name=f'rampup_{i}_{t}')
                m.addConstr(prev_w - w[i, t] <= 300, name=f'rampdown_{i}_{t}')
            else:
                m.addConstr(w[i, t] - w0[i] <= 300, name=f'rampup_{i}_{t}')
                m.addConstr(w0[i] - w[i, t] <= 300, name=f'rampdown_{i}_{t}')
    for i in trucks:
        for t in periods:
            m.addConstr(w[i, t] <= Q[i] * on[i, t], name=f'zeroifoff_{i}_{t}')
    for i in trucks:
        m.addConstr(startup[i, periods[-1]] == 0, name=f'nostartup_last2_{i}')
    m.optimize()
    return m
m = solve_problem()