import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    Q = {}
    S = {}
    C = {}
    for (_, row) in df.iterrows():
        t = int(row['truck_id'])
        Q[t] = float(row['Q'])
        S[t] = float(row['S'])
        C[t] = float(row['C'])
    d = {1: float(df.iloc[0]['d1']), 2: float(df.iloc[0]['d2']), 3: float(df.iloc[0]['d3']), 4: float(df.iloc[0]['d4'])}
    T = list(range(1, 11))
    P = list(range(1, 5))
    m = gurobipy.Model('TruckScheduling')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    s = m.addVars([(t, p) for t in T for p in P], vtype=GRB.BINARY, name='')
    w = m.addVars([(t, p) for t in T for p in P], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((S[t] * s[t, p] + C[t] * w[t, p] for t in T for p in P))
    m.setObjective(obj, GRB.MINIMIZE)
    for t in T:
        m.addConstr(s[t, 1] >= y[t, 1], name=f'startup_start_{t}')
        for p in range(2, 5):
            m.addConstr(s[t, p] >= y[t, p] - y[t, p - 1], name=f'startup_def_{t}_{p}')
        for p in P:
            m.addConstr(s[t, p] <= y[t, p], name=f'startup_leq_{t}_{p}')
        for p in range(2, 5):
            m.addConstr(s[t, p] <= 1 - y[t, p - 1], name=f'startup_leq2_{t}_{p}')
    for t in T:
        m.addConstr(s[t, 4] == 0, name=f'no_start_p4_{t}')
    for t in T:
        for p in range(1, 4):
            m.addConstr(s[t, p] <= y[t, p + 1], name=f'two_consec_{t}_{p}')
    for t in T:
        for p in range(2, 4):
            m.addConstr(y[t, p - 1] - y[t, p] <= 1 - y[t, p + 1], name=f'shutdown_{t}_{p}')
    for t in T:
        for p in P:
            m.addConstr(w[t, p] <= Q[t] * y[t, p], name=f'capacity_{t}_{p}')
    for t in T:
        m.addConstr(w[t, 1] <= 300, name=f'w1_limit_{t}')
        for p in range(2, 5):
            m.addConstr(w[t, p] - w[t, p - 1] <= 300, name=f'w_inc_{t}_{p}')
            m.addConstr(w[t, p - 1] - w[t, p] <= 300, name=f'w_dec_{t}_{p}')
    for p in P:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in T)) >= d[p], name=f'demand_{p}')
    for p in P:
        m.addConstr(gurobipy.quicksum((w[t, p] for t in T)) <= 0.9 * gurobipy.quicksum((Q[t] * y[t, p] for t in T)), name=f'spare_{p}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()