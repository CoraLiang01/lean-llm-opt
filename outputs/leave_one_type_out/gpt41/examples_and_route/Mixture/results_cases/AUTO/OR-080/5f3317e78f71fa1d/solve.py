import gurobipy as gp
from gurobipy import GRB
T = [1, 2, 3, 4]
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
d = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
for k in K:
    if k not in Q or k not in S or k not in C:
        raise ValueError(f'Missing truck parameter for k={k}')
for t in T:
    if t not in d:
        raise ValueError(f'Missing demand for t={t}')
m = gp.Model('Truck_Scheduling')
x = m.addVars(K, T, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(K, range(0, 5), vtype=GRB.BINARY, name='')
u = m.addVars(K, T, vtype=GRB.BINARY, name='')
for k in K:
    m.addConstr(y[k, 0] == 0, name=f'y0_{k}')
m.setObjective(gp.quicksum((S[k] * u[k, t] for k in K for t in T)) + gp.quicksum((C[k] * x[k, t] for k in K for t in T)), GRB.MINIMIZE)
for t in T:
    m.addConstr(gp.quicksum((x[k, t] for k in K)) >= d[t], name=f'demand_{t}')
for t in T:
    m.addConstr(gp.quicksum((x[k, t] for k in K)) <= 0.9 * gp.quicksum((Q[k] * y[k, t] for k in K)), name=f'sparecap_{t}')
for k in K:
    for t in T:
        m.addConstr(x[k, t] <= Q[k] * y[k, t], name=f'cap_{k}_{t}')
        m.addConstr(x[k, t] >= 0, name=f'xnonneg_{k}_{t}')
for k in K:
    for t in T:
        m.addConstr(u[k, t] >= y[k, t] - y[k, t - 1], name=f'u_lb_{k}_{t}')
        m.addConstr(u[k, t] <= 1 - y[k, t - 1], name=f'u_ub1_{k}_{t}')
        m.addConstr(u[k, t] <= y[k, t], name=f'u_ub2_{k}_{t}')
    m.addConstr(u[k, 4] == 0, name=f'u4zero_{k}')
for k in K:
    for t in [1, 2, 3]:
        m.addConstr(y[k, t + 1] >= y[k, t] - u[k, t], name=f'minup_{k}_{t}')
for k in K:
    for t in [2, 3]:
        m.addConstr(y[k, t - 1] - y[k, t] <= 1 - y[k, t + 1], name=f'mindown1_{k}_{t}')
    m.addConstr(y[k, 1] - y[k, 2] <= 1 - y[k, 4], name=f'mindown2_{k}_2')
for k in K:
    for t in T:
        m.addConstr(x[k, t] - (0 if t == 1 else x[k, t - 1]) <= 300, name=f'deltaplus_{k}_{t}')
        m.addConstr((0 if t == 1 else x[k, t - 1]) - x[k, t] <= 300, name=f'deltaminus_{k}_{t}')
        if t == 1:
            pass
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')