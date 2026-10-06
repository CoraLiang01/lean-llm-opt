import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    trucks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    periods = [1, 2, 3, 4]
    Q = {1: 1000, 2: 800, 3: 1200, 4: 600, 5: 900, 6: 700, 7: 1100, 8: 500, 9: 1000, 10: 650}
    S = {1: 500, 2: 300, 3: 400, 4: 250, 5: 450, 6: 280, 7: 420, 8: 200, 9: 480, 10: 260}
    C = {1: 2.0, 2: 3.0, 3: 2.5, 4: 3.0, 5: 2.2, 6: 2.8, 7: 2.4, 8: 3.2, 9: 2.1, 10: 2.9}
    d = {1: 1500, 2: 2000, 3: 1800, 4: 1000}
    for i in trucks:
        if i not in Q or i not in S or i not in C:
            raise ValueError(f'Missing parameter for truck {i}')
    for t in periods:
        if t not in d:
            raise ValueError(f'Missing demand for period {t}')
    m = gp.Model('Truck_Scheduling')
    y = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    u = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    z = m.addVars(trucks, periods, vtype=GRB.BINARY, name='')
    x = m.addVars(trucks, periods, lb=0, vtype=GRB.CONTINUOUS, name='')
    y0 = {i: 0 for i in trucks}
    m.setObjective(gp.quicksum((S[i] * u[i, t] for i in trucks for t in periods)) + gp.quicksum((C[i] * x[i, t] for i in trucks for t in periods)), GRB.MINIMIZE)
    for i in trucks:
        for t in periods:
            if t == 1:
                m.addConstr(y[i, 1] - y0[i] == u[i, 1] - z[i, 1], name=f'state_{i}_1')
            else:
                m.addConstr(y[i, t] - y[i, t - 1] == u[i, t] - z[i, t], name=f'state_{i}_{t}')
    for i in trucks:
        for t in [1, 2, 3]:
            m.addConstr(y[i, t + 1] >= u[i, t], name=f'minup_{i}_{t}')
    for i in trucks:
        m.addConstr(u[i, 4] == 0, name=f'nostart4_{i}')
    for i in trucks:
        for t in [1, 2]:
            if t + 1 in periods:
                m.addConstr(y[i, t + 1] <= 1 - z[i, t], name=f'mindown1_{i}_{t}')
            if t + 2 in periods:
                m.addConstr(y[i, t + 2] <= 1 - z[i, t], name=f'mindown2_{i}_{t}')
        m.addConstr(y[i, 4] <= 1 - z[i, 3], name=f'mindown3_{i}')
        m.addConstr(z[i, 4] == 0, name=f'noshutdown4_{i}')
    for i in trucks:
        for t in periods:
            m.addConstr(x[i, t] <= Q[i] * y[i, t], name=f'cap_{i}_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) >= d[t], name=f'demand_{t}')
    for t in periods:
        m.addConstr(gp.quicksum((x[i, t] for i in trucks)) <= 0.9 * gp.quicksum((Q[i] * y[i, t] for i in trucks)), name=f'sparecap_{t}')
    for i in trucks:
        m.addConstr(x[i, 1] <= 300, name=f'loadchg1a_{i}')
        m.addConstr(x[i, 1] >= 0, name=f'loadchg1b_{i}')
        for t in [2, 3, 4]:
            m.addConstr(x[i, t] - x[i, t - 1] <= 300, name=f'loadchg2a_{i}_{t}')
            m.addConstr(x[i, t] - x[i, t - 1] >= -300, name=f'loadchg2b_{i}_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()