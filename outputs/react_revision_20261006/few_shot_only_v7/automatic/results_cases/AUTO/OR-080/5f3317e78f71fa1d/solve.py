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
            raise KeyError(f'Missing required column: {col}')
    truck_ids = list(df['truck_id'].unique())
    if len(truck_ids) < 10:
        raise ValueError('Fewer than 10 trucks found in parameters.csv')
    truck_ids = truck_ids[:10]
    df = df[df['truck_id'].isin(truck_ids)].copy()
    Q = {}
    S = {}
    C = {}
    for (idx, row) in df.iterrows():
        tid = row['truck_id']
        Q[tid] = float(row['Q'])
        S[tid] = float(row['S'])
        C[tid] = float(row['C'])
    periods = [1, 2, 3, 4]
    demand = {1: 1500.0, 2: 2000.0, 3: 1800.0, 4: 1000.0}
    x_keys = [(tid, t) for tid in truck_ids for t in periods]
    y_keys = [(tid, t) for tid in truck_ids for t in periods]
    z_keys = [(tid, t) for tid in truck_ids for t in periods]
    m = gp.Model('TruckScheduling')
    x_vars = m.addVars(x_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    z_vars = m.addVars(z_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((S[tid] * z_vars[tid, t] for tid in truck_ids for t in periods)) + gp.quicksum((C[tid] * x_vars[tid, t] for tid in truck_ids for t in periods)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[tid, t] for tid in truck_ids)) >= demand[t], name=f'demand_{t}')
        m.addConstr(gp.quicksum((x_vars[tid, t] for tid in truck_ids)) <= 0.9 * gp.quicksum((Q[tid] * y_vars[tid, t] for tid in truck_ids)), name=f'sparecap_{t}')
    for tid in truck_ids:
        for t in periods:
            m.addConstr(x_vars[tid, t] <= Q[tid] * y_vars[tid, t], name=f'cap_{tid}_{t}')
            m.addConstr(x_vars[tid, t] >= 0.0, name=f'nonneg_{tid}_{t}')
    for tid in truck_ids:
        m.addConstr(z_vars[tid, 1] >= y_vars[tid, 1], name=f'startup1_{tid}')
        for t in [2, 3, 4]:
            m.addConstr(z_vars[tid, t] >= y_vars[tid, t] - y_vars[tid, t - 1], name=f'startup_{tid}_{t}')
    for tid in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[tid, t + 1] >= z_vars[tid, t], name=f'minup_{tid}_{t}')
        m.addConstr(z_vars[tid, 4] == 0, name=f'nostart4_{tid}')
    for tid in truck_ids:
        for t in [1, 2]:
            m.addConstr(y_vars[tid, t] - y_vars[tid, t + 1] <= 1 - y_vars[tid, t + 2], name=f'mindown_{tid}_{t}')
    for tid in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(x_vars[tid, t + 1] - x_vars[tid, t] <= 300, name=f'rampup_{tid}_{t}')
            m.addConstr(x_vars[tid, t] - x_vars[tid, t + 1] <= 300, name=f'rampdown_{tid}_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')