import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
    required_cols = ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    truck_ids = df['truck_id'].tolist()
    if len(truck_ids) != 10:
        raise ValueError(f'Expected 10 trucks, found {len(truck_ids)}')
    periods = [1, 2, 3, 4]
    Q = {}
    S = {}
    C = {}
    for (idx, row) in df.iterrows():
        tid = row['truck_id']
        try:
            Q[tid] = float(row['Q'])
            S[tid] = float(row['S'])
            C[tid] = float(row['C'])
        except Exception as e:
            raise ValueError(f'Non-numeric Q/S/C for truck_id {tid}: {e}')
    try:
        d1 = float(df.iloc[0]['d1'])
        d2 = float(df.iloc[0]['d2'])
        d3 = float(df.iloc[0]['d3'])
        d4 = float(df.iloc[0]['d4'])
    except Exception as e:
        raise ValueError(f'Non-numeric d1-d4 in parameters.csv: {e}')
    demand = {1: d1, 2: d2, 3: d3, 4: d4}
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in truck_ids for t in periods]
    x_vars = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_keys = [(i, t) for i in truck_ids for t in periods]
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    z_keys = [(i, t) for i in truck_ids for t in periods]
    z_vars = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[i] * z_vars[i, t] for i in truck_ids for t in periods)) + gp.quicksum((C[i] * x_vars[i, t] for i in truck_ids for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
    for i in truck_ids:
        for t in periods:
            m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'cap_{i}_{t}')
            m.addConstr(x_vars[i, t] >= 0, name=f'nonneg_{i}_{t}')
    for i in truck_ids:
        m.addConstr(z_vars[i, 1] == y_vars[i, 1], name=f'startup1_{i}')
        for t in [2, 3, 4]:
            m.addConstr(z_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startup_lb_{i}_{t}')
            m.addConstr(z_vars[i, t] <= y_vars[i, t], name=f'startup_ub1_{i}_{t}')
            m.addConstr(z_vars[i, t] <= 1 - y_vars[i, t - 1], name=f'startup_ub2_{i}_{t}')
    for i in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[i, t + 1] >= z_vars[i, t], name=f'minup_{i}_{t}')
        m.addConstr(z_vars[i, 4] == 0, name=f'nostart4_{i}')
    for i in truck_ids:
        m.addConstr(y_vars[i, 1] - y_vars[i, 2] <= 1 - y_vars[i, 3], name=f'mindown1a_{i}')
        m.addConstr(y_vars[i, 1] - y_vars[i, 2] <= 1 - y_vars[i, 4], name=f'mindown1b_{i}')
        m.addConstr(y_vars[i, 2] - y_vars[i, 3] <= 1 - y_vars[i, 4], name=f'mindown2a_{i}')
        m.addConstr(y_vars[i, 3] - y_vars[i, 4] <= 1 - 0, name=f'mindown3a_{i}')
    for i in truck_ids:
        for t in [2, 3, 4]:
            if t == 2:
                prev = 0
                m.addConstr(x_vars[i, t] - 0 <= 300, name=f'rampup_{i}_{t}')
                m.addConstr(0 - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
            else:
                m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
                m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')