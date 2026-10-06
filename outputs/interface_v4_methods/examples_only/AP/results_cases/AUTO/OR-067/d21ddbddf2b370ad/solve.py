import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    managers = ['MA', 'MB', 'MC']
    projects = ['P1', 'P2', 'P3']
    cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
    for i in managers:
        if i not in cost or not isinstance(cost[i], dict):
            raise ValueError(f'Missing cost data for manager {i}')
        for j in projects:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for manager {i}, project {j}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = {}
    for i in managers:
        x[i] = {}
        for j in projects:
            x[i][j] = m.addVar(vtype=GRB.BINARY, name='', obj=0)
    obj = gp.LinExpr()
    for i in managers:
        for j in projects:
            obj += cost[i][j] * x[i][j]
    m.setObjective(obj, GRB.MINIMIZE)
    for i in managers:
        m.addConstr(gp.quicksum((x[i][j] for j in projects)) == 1, name='')
    for j in projects:
        m.addConstr(gp.quicksum((x[i][j] for i in managers)) == 1, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in managers:
            for j in projects:
                print(f'x_{i}_{j} {x[i][j].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()