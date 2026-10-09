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
y0 = {i: 0 for i in trucks}
x0 = {i: 0.0 for i in trucks}
m.setObjective(gp.quicksum((S[i] * u[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), GRB.MINIMIZE)
for i in trucks:
    for t in periods:
        if t == 1:
            m.addConstr(u[i, 1] >= y[i, 1] - y0[i], name='startup_lb_%d_1' % i)
        else:
            m.addConstr(u[i, t] >= y[i, t] - y[i, t - 1], name='startup_lb_%d_%d' % (i, t))
        m.addConstr(u[i, t] <= 1, name='startup_ub_%d_%d' % (i, t))
    m.addConstr(u[i, 4] == 0, name='no_startup_4_%d' % i)
for i in trucks:
    for t in [1, 2, 3]:
        if t == 1:
            m.addConstr(y[i, 2] >= y[i, 1] - u[i, 1], name='min_up_%d_1' % i)
        elif t == 2:
            m.addConstr(y[i, 3] >= y[i, 2] - u[i, 2], name='min_up_%d_2' % i)
        elif t == 3:
            m.addConstr(y[i, 4] >= y[i, 3] - u[i, 3], name='min_up_%d_3' % i)
    m.addConstr(u[i, 4] == 0, name='no_startup_4b_%d' % i)
for i in trucks:
    for t in [2, 3]:
        if t + 1 in periods:
            m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name='min_down_%d_%d' % (i, t))
    m.addConstr(y[i, 2] - y[i, 3] <= 1 - y[i, 4], name='min_down2_%d' % i)
    m.addConstr(y[i, 3] - y[i, 4] <= 1, name='min_down3_%d' % i)
for i in trucks:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name='cap_link_%d_%d' % (i, t))
        m.addConstr(x[i, t] >= 0, name='x_nonneg_%d_%d' % (i, t))
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= demand[t], name='demand_%d' % t)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name='spare_cap_%d' % t)
for i in trucks:
    for t in periods:
        prev_x = x0[i] if t == 1 else x[i, t - 1]
        m.addConstr(x[i, t] - prev_x <= 300, name='ramp_up_%d_%d' % (i, t))
        m.addConstr(prev_x - x[i, t] <= 300, name='ramp_down_%d_%d' % (i, t))
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')