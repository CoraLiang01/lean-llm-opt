import gurobipy as gp
from gurobipy import GRB
trucks = [{'truck_id': 1, 'Q': 1000, 'S': 500, 'C': 2.0}, {'truck_id': 2, 'Q': 800, 'S': 300, 'C': 3.0}, {'truck_id': 3, 'Q': 1200, 'S': 400, 'C': 2.5}, {'truck_id': 4, 'Q': 600, 'S': 250, 'C': 3.0}, {'truck_id': 5, 'Q': 900, 'S': 450, 'C': 2.2}, {'truck_id': 6, 'Q': 700, 'S': 280, 'C': 2.8}, {'truck_id': 7, 'Q': 1100, 'S': 420, 'C': 2.4}, {'truck_id': 8, 'Q': 500, 'S': 200, 'C': 3.2}, {'truck_id': 9, 'Q': 1000, 'S': 480, 'C': 2.1}, {'truck_id': 10, 'Q': 650, 'S': 260, 'C': 2.9}]
truck_ids = [t['truck_id'] for t in trucks]
Q = {t['truck_id']: t['Q'] for t in trucks}
S = {t['truck_id']: t['S'] for t in trucks}
C = {t['truck_id']: t['C'] for t in trucks}
periods = [1, 2, 3, 4]
demands = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
m = gp.Model('Truck_Scheduling')
y = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
u = m.addVars(truck_ids, periods, vtype=GRB.BINARY, name='')
x = m.addVars(truck_ids, periods, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((S[i] * u[i, t] + C[i] * x[i, t] for i in truck_ids for t in periods)), GRB.MINIMIZE)
for i in truck_ids:
    m.addConstr(u[i, 1] == y[i, 1], name='startup1_%d' % i)
    for t in [2, 3, 4]:
        m.addConstr(u[i, t] >= y[i, t] - y[i, t - 1], name='startup2_%d_%d' % (i, t))
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t + 1] >= u[i, t], name='minup_%d_%d' % (i, t))
    m.addConstr(u[i, 4] == 0, name='nostart4_%d' % i)
for i in truck_ids:
    for t in [1, 2, 3]:
        m.addConstr(y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1], name='mindown_%d_%d' % (i, t))
        if t == 1:
            m.addConstr(0 - y[i, 1] <= 1 - y[i, 2], name='mindown0_%d' % i)
for i in truck_ids:
    for t in periods:
        m.addConstr(x[i, t] <= Q[i] * y[i, t], name='cap_%d_%d' % (i, t))
        m.addConstr(x[i, t] >= 0, name='nonneg_%d_%d' % (i, t))
for i in truck_ids:
    for t in [2, 3, 4]:
        m.addConstr(x[i, t] - x[i, t - 1] <= 300, name='loadchg1_%d_%d' % (i, t))
        m.addConstr(x[i, t - 1] - x[i, t] <= 300, name='loadchg2_%d_%d' % (i, t))
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) >= demands[t], name='demand_%d' % t)
for t in periods:
    m.addConstr(gp.quicksum((x[i, t] for i in truck_ids)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in truck_ids)), name='buffer_%d' % t)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')