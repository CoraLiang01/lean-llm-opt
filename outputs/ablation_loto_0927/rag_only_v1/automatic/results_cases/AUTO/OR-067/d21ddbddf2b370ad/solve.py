import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    managers = ['MA', 'MB', 'MC']
    projects = ['P1', 'P2', 'P3']
    cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
    for i in managers:
        if i not in cost:
            raise ValueError(f'Missing cost data for manager {i}')
        for j in projects:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for manager {i}, project {j}')
    m = gp.Model()
    x = {}
    for i in managers:
        x[i] = {}
        for j in projects:
            x[i][j] = m.addVar(vtype=GRB.BINARY, name='x', lb=0, ub=1)
    for i in managers:
        m.addConstr(gp.quicksum((x[i][j] for j in projects)) == 1, name='cm')
    for j in projects:
        m.addConstr(gp.quicksum((x[i][j] for i in managers)) == 1, name='cp')
    obj = gp.quicksum((cost[i][j] * x[i][j] for i in managers for j in projects))
    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in managers:
            for j in projects:
                v = x[i][j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_assignment()