import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
param_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
df = pd.read_csv(param_path, sep=',', dtype=str, keep_default_na=False)
for col in ['truck_id', 'Q', 'S', 'C', 'd1', 'd2', 'd3', 'd4']:
    df[col] = df[col].astype(float if col == 'C' else int)
truck_ids = df['truck_id'].tolist()
periods = [1, 2, 3, 4]
Q = {int(row['truck_id']): int(row['Q']) for (_, row) in df.iterrows()}
S = {int(row['truck_id']): int(row['S']) for (_, row) in df.iterrows()}
C = {int(row['truck_id']): float(row['C']) for (_, row) in df.iterrows()}
demand = {1: int(df.iloc[0]['d1']), 2: int(df.iloc[0]['d2']), 3: int(df.iloc[0]['d3']), 4: int(df.iloc[0]['d4'])}
if set(truck_ids) != set(Q.keys()) or set(truck_ids) != set(S.keys()) or set(truck_ids) != set(C.keys()):
    raise ValueError('Mismatch in truck_id coverage in parameters.')
m = gp.Model('truck_scheduling')
on_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
startup_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
shut_vars = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
w_vars = m.addVars(truck_ids, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
on0 = {i: 0 for i in truck_ids}
w0 = {i: 0 for i in truck_ids}
startup_cost = gp.quicksum((S[i] * startup_vars[i, t] for i in truck_ids for t in periods))
transport_cost = gp.quicksum((C[i] * w_vars[i, t] for i in truck_ids for t in periods))
m.setObjective(startup_cost + transport_cost, GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((w_vars[i, t] for i in truck_ids)) >= demand[t], name='demand_%d' % t)
for t in periods:
    m.addConstr(gp.quicksum((w_vars[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * on_vars[i, t] for i in truck_ids)), name='sparecap_%d' % t)
for i in truck_ids:
    for t in periods:
        m.addConstr(w_vars[i, t] <= Q[i] * on_vars[i, t], name='cap_%d_%d' % (i, t))
for i in truck_ids:
    for t in periods:
        prev_on = on0[i] if t == 1 else on_vars[i, t - 1]
        m.addConstr(startup_vars[i, t] >= on_vars[i, t] - prev_on, name='startup_logic_%d_%d' % (i, t))
for i in truck_ids:
    for t in periods:
        prev_on = on0[i] if t == 1 else on_vars[i, t - 1]
        m.addConstr(shut_vars[i, t] >= prev_on - on_vars[i, t], name='shut_logic_%d_%d' % (i, t))
for i in truck_ids:
    m.addConstr(startup_vars[i, 4] == 0, name='no_startup_4_%d' % i)
    for t in [1, 2, 3]:
        if t + 1 in periods:
            m.addConstr(on_vars[i, t + 1] >= startup_vars[i, t], name='min_up_%d_%d' % (i, t))
for i in truck_ids:
    for t in [1, 2, 3]:
        if t + 1 in periods:
            m.addConstr(on_vars[i, t] + on_vars[i, t + 1] <= 2 - shut_vars[i, t], name='min_down_%d_%d' % (i, t))
for i in truck_ids:
    for t in [2, 3, 4]:
        prev_w = w0[i] if t == 1 else w_vars[i, t - 1] if t > 1 else w0[i]
        m.addConstr(w_vars[i, t] - (w_vars[i, t - 1] if t > 1 else w0[i]) <= 300, name='ramp_up_%d_%d' % (i, t))
        m.addConstr(w_vars[i, t] - (w_vars[i, t - 1] if t > 1 else w0[i]) >= -300, name='ramp_down_%d_%d' % (i, t))
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print('Solver status:', m.Status)