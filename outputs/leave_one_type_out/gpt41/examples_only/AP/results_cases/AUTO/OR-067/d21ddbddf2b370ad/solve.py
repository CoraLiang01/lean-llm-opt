import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    managers = ['MA', 'MB', 'MC']
    projects = ['P1', 'P2', 'P3']
    cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
    for m in managers:
        if m not in cost or not isinstance(cost[m], dict):
            raise ValueError(f'Missing cost data for manager {m}')
        for p in projects:
            if p not in cost[m]:
                raise ValueError(f'Missing cost data for manager {m}, project {p}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for ma in managers:
        m.addConstr(gp.quicksum((x[ma, p] for p in projects)) == 1, name='cm')
    for pr in projects:
        m.addConstr(gp.quicksum((x[mgr, pr] for mgr in managers)) == 1, name='cp')
    obj = gp.quicksum((cost[ma][pr] * x[ma, pr] for ma in managers for pr in projects))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for ma in managers:
            for pr in projects:
                v = x[ma, pr]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_assignment()