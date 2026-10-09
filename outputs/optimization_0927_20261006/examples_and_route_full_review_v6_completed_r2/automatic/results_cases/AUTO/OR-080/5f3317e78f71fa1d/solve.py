import gurobipy as gp
from gurobipy import GRB
T = [1, 2, 3, 4]
K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
d = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
m = gp.Model('Truck_Scheduling')
y_vars = m.addVars(K, T, vtype=GRB.BINARY, name='')
u_vars = m.addVars(K, T, vtype=GRB.BINARY, name='')
x_vars = m.addVars(K, T, lb=0, vtype=GRB.CONTINUOUS, name='')
y0 = {k: 0 for k in K}
x0 = {k: 0 for k in K}
m.setObjective(gp.quicksum((S[k] * u_vars[k, t] for k in K for t in T)) + gp.quicksum((C[k] * x_vars[k, t] for k in K for t in T)), GRB.MINIMIZE)
for t in T:
    m.addConstr(gp.quicksum((x_vars[k, t] for k in K)) >= d[t], name=f'demand_lb_{t}')
    m.addConstr(gp.quicksum((x_vars[k, t] for k in K)) <= 0.9 * gp.quicksum((Q[k] * y_vars[k, t] for k in K)), name=f'demand_ub_{t}')
for k in K:
    for t in T:
        m.addConstr(x_vars[k, t] <= Q[k] * y_vars[k, t], name=f'cap_{k}_{t}')
        m.addConstr(x_vars[k, t] >= 0, name=f'nonneg_{k}_{t}')
for k in K:
    for t in T:
        if t == 1:
            prev_y = y0[k]
        else:
            prev_y = y_vars[k, t - 1]
        m.addConstr(u_vars[k, t] >= y_vars[k, t] - prev_y, name=f'startup_lb_{k}_{t}')
        m.addConstr(u_vars[k, t] <= 1 - prev_y, name=f'startup_ub1_{k}_{t}')
        m.addConstr(u_vars[k, t] <= y_vars[k, t], name=f'startup_ub2_{k}_{t}')
    m.addConstr(u_vars[k, 4] == 0, name=f'no_startup_4_{k}')
for k in K:
    for t in [1, 2, 3]:
        m.addConstr(y_vars[k, t + 1] >= u_vars[k, t], name=f'min_up_{k}_{t}')
for k in K:
    for t in [2, 3]:
        prev_y = y_vars[k, t - 1]
        curr_y = y_vars[k, t]
        if t + 1 in T:
            m.addConstr(prev_y - curr_y <= 1 - y_vars[k, t + 1], name=f'min_down1_{k}_{t}')
        if t + 2 in T:
            m.addConstr(prev_y - curr_y <= 1 - y_vars[k, t + 2], name=f'min_down2_{k}_{t}')
for k in K:
    for t in [3, 4]:
        if t - 1 == 1:
            y_tm1 = y_vars[k, 1]
        else:
            y_tm1 = y_vars[k, t - 1]
        if t - 2 == 0:
            y_tm2 = y0[k]
        else:
            y_tm2 = y_vars[k, t - 2]
        m.addConstr(u_vars[k, t] + y_tm1 - y_tm2 <= 1, name=f'norestart_{k}_{t}')
for k in K:
    for t in [2, 3, 4]:
        m.addConstr(x_vars[k, t] - x_vars[k, t - 1] <= 300, name=f'ramp_up_{k}_{t}')
        m.addConstr(x_vars[k, t - 1] - x_vars[k, t] <= 300, name=f'ramp_down_{k}_{t}')
    m.addConstr(x_vars[k, 1] - x0[k] <= 300, name=f'ramp_up_{k}_1')
    m.addConstr(x0[k] - x_vars[k, 1] <= 300, name=f'ramp_down_{k}_1')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')