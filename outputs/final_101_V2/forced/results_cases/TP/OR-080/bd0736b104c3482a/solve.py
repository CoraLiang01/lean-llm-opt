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
x = m.addVars(trucks, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((S[i] * u[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), GRB.MINIMIZE)
for i in trucks:
    m.addConstr(y[i, 1] == u[i, 1], name=f'startup_init_{i}')
    for t in [2, 3, 4]:
        m.addConstr(y[i, t] - y[i, t - 1] <= u[i, t], name=f'startup_logic1_{i}_{t}')
        m.addConstr(u[i, t] <= y[i, t], name=f'startup_logic2_{i}_{t}')
    m.addConstr(u[i, 4] == 0, name=f'no_startup_t4_{i}')
for i in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= u[i, t], name=f'min_up_{i}_{t}')
for i in trucks:
    m.addConstr(y[i, 1] - y[i, 2] <= 1 - y[i, 3], name=f'min_down1_{i}_2')
    m.addConstr(y[i, 1] - y[i, 2] <= 1 - y[i, 4], name=f'min_down2_{i}_2')
    m.addConstr(y[i, 2] - y[i, 3] <= 1 - y[i, 4], name=f'min_down_{i}_3')
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
        m.addConstr(x[i, t] >= 0, name=f'x_nonneg_{i}_{t}')
for i in trucks:
    for t in [2, 3, 4]:
        m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'loadchg1_{i}_{t}')
        m.addConstr(x[i, t - 1] - x[i, t] <= 300, name=f'loadchg2_{i}_{t}')
    m.addConstr(x[i, 1] - 0 <= 300, name=f'loadchg1_{i}_1')
    m.addConstr(0 - x[i, 1] <= 300, name=f'loadchg2_{i}_1')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')