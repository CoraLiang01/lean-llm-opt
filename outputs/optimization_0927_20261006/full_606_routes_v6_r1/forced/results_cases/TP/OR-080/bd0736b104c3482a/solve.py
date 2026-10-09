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
y_vars = m.addVars(I, T, vtype=GRB.BINARY, name='')
u_vars = m.addVars(I, T, vtype=GRB.BINARY, name='')
z_vars = m.addVars(I, T, vtype=GRB.BINARY, name='')
x_vars = m.addVars(I, T, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((S[i] * u_vars[i, t] for i in I for t in T)) + gp.quicksum((C[i] * x_vars[i, t] for i in I for t in T)), GRB.MINIMIZE)
for i in I:
    m.addConstr(y_vars[i, 1] == u_vars[i, 1], name=f'startup_logic1_{i}')
    for t in [2, 3, 4]:
        m.addConstr(y_vars[i, t] - y_vars[i, t - 1] == u_vars[i, t] - z_vars[i, t], name=f'startup_logic2_{i}_{t}')
for i in I:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[i, t + 1] >= u_vars[i, t], name=f'min_up_{i}_{t}')
    m.addConstr(u_vars[i, 4] == 0, name=f'no_startup_4_{i}')
for i in I:
    for t in [1, 2]:
        m.addConstr(y_vars[i, t + 1] <= 1 - z_vars[i, t], name=f'min_down1_{i}_{t}')
        m.addConstr(y_vars[i, t + 2] <= 1 - z_vars[i, t], name=f'min_down2_{i}_{t}')
    m.addConstr(y_vars[i, 4] <= 1 - z_vars[i, 3], name=f'min_down3_{i}')
for i in I:
    for t in T:
        m.addConstr(x_vars[i, t] <= Q[i] * y_vars[i, t], name=f'inactive_zero_{i}_{t}')
for i in I:
    m.addConstr(x_vars[i, 1] - 0 <= 300, name=f'loadchg1_{i}')
    m.addConstr(0 - x_vars[i, 1] <= 300, name=f'loadchg2_{i}')
    for t in [2, 3, 4]:
        m.addConstr(x_vars[i, t] - x_vars[i, t - 1] <= 300, name=f'loadchg3_{i}_{t}')
        m.addConstr(x_vars[i, t - 1] - x_vars[i, t] <= 300, name=f'loadchg4_{i}_{t}')
for t in T:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) >= d[t], name=f'demand_{t}')
for t in T:
    m.addConstr(gp.quicksum((x_vars[i, t] for i in I)) <= 0.9 * gp.quicksum((Q[i] * y_vars[i, t] for i in I)), name=f'sparecap_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')