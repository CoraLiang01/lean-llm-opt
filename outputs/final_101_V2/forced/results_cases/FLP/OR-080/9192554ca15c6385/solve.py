import gurobipy as gp
from gurobipy import GRB
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
T = [1, 2, 3, 4]
Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
d = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
for i in I:
    if i not in Q or i not in S or i not in C:
        raise ValueError(f'Missing parameter for truck {i}')
for t in T:
    if t not in d:
        raise ValueError(f'Missing demand for period {t}')
m = gp.Model('Truck_Scheduling')
x = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(I, T, vtype=GRB.BINARY, name='')
z = m.addVars(I, T, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((S[i] * z[i, t] for i in I for t in T)) + gp.quicksum((C[i] * x[i, t] for i in I for t in T)), GRB.MINIMIZE)
for t in T:
    m.addConstr(gp.quicksum((x[i, t] for i in I)) >= d[t], name=f'demand_{t}')
for t in T:
    m.addConstr(gp.quicksum((x[i, t] for i in I)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in I)), name=f'sparecap_{t}')
for i in I:
    for t in T:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
for i in I:
    m.addConstr(z[i, 1] >= y[i, 1], name=f'startup1_{i}')
    m.addConstr(z[i, 4] == 0, name=f'nostart4_{i}')
    for t in [2, 3, 4]:
        m.addConstr(z[i, t] >= y[i, t] - y[i, t - 1] if t > 1 else z[i, t] >= y[i, t], name=f'startup_{i}_{t}')
for i in I:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= z[i, t], name=f'minup_{i}_{t}')
for i in I:
    m.addConstr(y[i, 1] - y[i, 2] <= 1 - y[i, 3], name=f'mindown_{i}_2')
    m.addConstr(y[i, 2] - y[i, 3] <= 1 - y[i, 4], name=f'mindown_{i}_3')
for i in I:
    m.addConstr(y[i, 1] - y[i, 2] + y[i, 4] <= 1, name=f'norestart_{i}_2')
for i in I:
    for t in [1, 2, 3, 4]:
        x_prev = 0 if t == 1 else x[i, t - 1]
        m.addConstr(x[i, t] - x_prev <= 300, name=f'rampup_{i}_{t}')
        m.addConstr(x_prev - x[i, t] <= 300, name=f'rampdown_{i}_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')