import gurobipy as gp
from gurobipy import GRB
trucks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
periods = [1, 2, 3, 4]
Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
demand = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
m = gp.Model('Truck_Scheduling')
y = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
u = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
z = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
x = m.addVars(trucks, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
r = m.addVars(trucks, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((S[i] * u[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), GRB.MINIMIZE)
for i in trucks:
    y0 = 0
    m.addConstr(y[i, 1] == u[i, 1], name=f'startup1_{i}')
    for t in [2, 3, 4]:
        m.addConstr(y[i, t] - y[i, t - 1] == u[i, t] - z[i, t], name=f'startup2_{i}_{t}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(u[i, t] <= y[i, t + 1], name=f'minup_{i}_{t}')
    m.addConstr(u[i, 4] == 0, name=f'minup_last_{i}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(z[i, t] <= 1 - y[i, t + 1], name=f'mindown1_{i}_{t}')
    for t in [1, 2]:
        m.addConstr(z[i, t] <= 1 - y[i, t + 2], name=f'mindown2_{i}_{t}')
    m.addConstr(z[i, 4] == 0, name=f'mindown_last_{i}')
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'inactive_{i}_{t}')
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i], name=f'cap_{i}_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
for i in trucks:
    x_prev = 0
    for t in periods:
        m.addConstr(r[i, t] >= x[i, t] - x_prev, name=f'ramp1_{i}_{t}')
        m.addConstr(r[i, t] >= x_prev - x[i, t], name=f'ramp2_{i}_{t}')
        m.addConstr(r[i, t] <= 300, name=f'ramp3_{i}_{t}')
        x_prev = x[i, t]
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')