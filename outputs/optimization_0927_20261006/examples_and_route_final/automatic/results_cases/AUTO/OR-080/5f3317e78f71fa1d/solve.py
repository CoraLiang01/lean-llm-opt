import gurobipy as gp
from gurobipy import GRB
trucks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
periods = [1, 2, 3, 4]
Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
demand = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
if set(Q.keys()) != set(trucks) or set(S.keys()) != set(trucks) or set(C.keys()) != set(trucks):
    raise ValueError('Parameter keys do not match truck set')
if set(demand.keys()) != set(periods):
    raise ValueError('Demand keys do not match period set')
m = gp.Model('Truck_Scheduling')
y_vars = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
z_vars = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
x_vars = m.addVars(trucks, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
y0 = {k: 0 for k in trucks}
x0 = {k: 0 for k in trucks}
m.setObjective(gp.quicksum((S[k] * z_vars[k, t] for k in trucks for t in periods)) + gp.quicksum((C[k] * x_vars[k, t] for k in trucks for t in periods)), GRB.MINIMIZE)
for k in trucks:
    for t in periods:
        m.addConstr(z_vars[k, t] >= y_vars[k, t] - (y_vars[k, periods[periods.index(t) - 1]] if t > 1 else y0[k]), name=f'startup_logic_{k}_{t}')
for k in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[k, t + 1] >= z_vars[k, t], name=f'min_up_{k}_{t}')
    m.addConstr(z_vars[k, 4] == 0, name=f'no_startup_4_{k}')
for k in trucks:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[k, t + 1] <= y_vars[k, t], name=f'min_down_{k}_{t}')
for k in trucks:
    for t in periods:
        m.addConstr(x_vars[k, t] <= Q[k] * y_vars[k, t], name=f'inactive_zero_{k}_{t}')
for k in trucks:
    for t in periods:
        m.addConstr(x_vars[k, t] <= Q[k], name=f'cap_{k}_{t}')
for k in trucks:
    for t in [2, 3, 4]:
        prev_t = t - 1
        m.addConstr(x_vars[k, t] - (x_vars[k, prev_t] if prev_t in periods else x0[k]) <= 300, name=f'load_inc_{k}_{t}')
        m.addConstr((x_vars[k, prev_t] if prev_t in periods else x0[k]) - x_vars[k, t] <= 300, name=f'load_dec_{k}_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[k, t] for k in trucks)) >= demand[t], name=f'demand_{t}')
for t in periods:
    m.addConstr(gp.quicksum((x_vars[k, t] for k in trucks)) <= 0.9 * gp.quicksum((Q[k] * y_vars[k, t] for k in trucks)), name=f'spare_buffer_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')