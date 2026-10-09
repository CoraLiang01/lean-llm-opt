import gurobipy as gp
from gurobipy import GRB
trucks = [{'truck_id': 1, 'Q': 1000, 'S': 500, 'C': 2.0}, {'truck_id': 2, 'Q': 800, 'S': 300, 'C': 3.0}, {'truck_id': 3, 'Q': 1200, 'S': 400, 'C': 2.5}, {'truck_id': 4, 'Q': 600, 'S': 250, 'C': 3.0}, {'truck_id': 5, 'Q': 900, 'S': 450, 'C': 2.2}, {'truck_id': 6, 'Q': 700, 'S': 280, 'C': 2.8}, {'truck_id': 7, 'Q': 1100, 'S': 420, 'C': 2.4}, {'truck_id': 8, 'Q': 500, 'S': 200, 'C': 3.2}, {'truck_id': 9, 'Q': 1000, 'S': 480, 'C': 2.1}, {'truck_id': 10, 'Q': 650, 'S': 260, 'C': 2.9}]
truck_ids = [t['truck_id'] for t in trucks]
Q = {t['truck_id']: t['Q'] for t in trucks}
S = {t['truck_id']: t['S'] for t in trucks}
C = {t['truck_id']: t['C'] for t in trucks}
T = [1, 2, 3, 4]
K = truck_ids
demand = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
m = gp.Model('Truck_Scheduling')
y_vars = m.addVars(K, T, vtype=GRB.BINARY, name='')
z_vars = m.addVars(K, T, vtype=GRB.BINARY, name='')
x_vars = m.addVars(K, T, lb=0, vtype=GRB.CONTINUOUS, name='')
y0 = {k: 0 for k in K}
m.setObjective(gp.quicksum((S[k] * z_vars[k, t] + C[k] * x_vars[k, t] for k in K for t in T)), GRB.MINIMIZE)
for k in K:
    for t in T:
        m.addConstr(z_vars[k, t] >= y_vars[k, t] - (y0[k] if t == 1 else y_vars[k, t - 1]), name='')
        m.addConstr(z_vars[k, t] <= 1 - (y0[k] if t == 1 else y_vars[k, t - 1]), name='')
        m.addConstr(z_vars[k, t] <= y_vars[k, t], name='')
    m.addConstr(z_vars[k, 4] == 0, name='')
for k in K:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[k, t + 1] >= z_vars[k, t], name='')
for k in K:
    for t in [1, 2]:
        if t + 2 in T:
            m.addConstr(y_vars[k, t + 1] + y_vars[k, t + 2] <= 1 + y_vars[k, t], name='')
for k in K:
    for t in T:
        m.addConstr(x_vars[k, t] <= Q[k] * y_vars[k, t], name='')
        m.addConstr(x_vars[k, t] >= 0, name='')
for k in K:
    m.addConstr(x_vars[k, 1] <= 300, name='')
    for t in [2, 3, 4]:
        m.addConstr(x_vars[k, t] - x_vars[k, t - 1] <= 300, name='')
        m.addConstr(x_vars[k, t - 1] - x_vars[k, t] <= 300, name='')
for t in T:
    m.addConstr(gp.quicksum((x_vars[k, t] for k in K)) >= demand[t], name='')
for t in T:
    m.addConstr(gp.quicksum((x_vars[k, t] for k in K)) <= 0.9 * gp.quicksum((Q[k] * y_vars[k, t] for k in K)), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')