import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
    truck_ids = df['truck_id'].astype(int).tolist()
    truck_ids_set = set(truck_ids)
    if len(truck_ids) != 10 or sorted(truck_ids) != list(range(1, 11)):
        raise ValueError('Expected exactly 10 trucks with IDs 1..10 in parameters.csv')
    Q = df.set_index('truck_id')['Q'].astype(int).to_dict()
    S = df.set_index('truck_id')['S'].astype(int).to_dict()
    C = df.set_index('truck_id')['C'].astype(float).to_dict()
    periods = [1, 2, 3, 4]
    d_cols = ['d1', 'd2', 'd3', 'd4']
    d_vals = []
    for (idx, dcol) in enumerate(d_cols):
        vals = df[dcol].astype(int).unique()
        if len(vals) != 1:
            raise ValueError(f'Demand column {dcol} has inconsistent values across trucks')
        d_vals.append(vals[0])
    demand = {t: d_vals[t - 1] for t in periods}
    m = gp.Model('TruckScheduling')
    x_keys = [(i, t) for i in truck_ids for t in periods]
    x_vars = m.addVars(x_keys, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_keys = [(i, t) for i in truck_ids for t in periods]
    y_vars = m.addVars(y_keys, vtype=gp.GRB.BINARY, name='')
    s_keys = [(i, t) for i in truck_ids for t in periods]
    s_vars = m.addVars(s_keys, vtype=gp.GRB.BINARY, name='')
    startup_cost = gp.quicksum((S[str(i)] * s_vars[i, t] for i in truck_ids for t in periods))
    transport_cost = gp.quicksum((C[str(i)] * x_vars[i, t] for i in truck_ids for t in periods))
    m.setObjective(startup_cost + transport_cost, gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) >= demand[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[str(i)] * y_vars[i, t] for i in truck_ids)), name=f'sparecap_{t}')
    for i in truck_ids:
        for t in periods:
            m.addConstr(x_vars[i, t] <= Q[str(i)] * y_vars[i, t], name=f'cap_{i}_{t}')
    for i in truck_ids:
        m.addConstr(s_vars[i, 1] == y_vars[i, 1], name=f'startup1_{i}')
        for t in [2, 3, 4]:
            m.addConstr(s_vars[i, t] >= y_vars[i, t] - y_vars[i, t - 1], name=f'startupdef_{i}_{t}')
    for i in truck_ids:
        for t in [1, 2, 3]:
            m.addConstr(y_vars[i, t + 1] >= s_vars[i, t], name=f'minup_{i}_{t}')
        m.addConstr(s_vars[i, 4] == 0, name=f'nostart4_{i}')
    for i in truck_ids:
        for t in [2, 3]:
            m.addConstr(y_vars[i, t - 1] - y_vars[i, t] <= 1 - y_vars[i, t + 1], name=f'mindown_{i}_{t}')
        m.addConstr(y_vars[i, 3] - y_vars[i, 4] <= 1 - y_vars[i, 4], name=f'mindown34_{i}')
    for i in truck_ids:
        for t in [2, 3, 4]:
            m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'rampup_{i}_{t}')
            m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'rampdown_{i}_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')