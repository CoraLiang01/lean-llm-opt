import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7']
    projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7']
    cost_matrix = [[2972, 2727, 2795, 2922, 1302, 2489, 1533], [1094, 2158, 2990, 1844, 2887, 2021, 2288], [2133, 1675, 2422, 2639, 1033, 2261, 1695], [1951, 2309, 2070, 2802, 2328, 1313, 2434], [1269, 2153, 1296, 2685, 2627, 1610, 1641], [1220, 1192, 2907, 2622, 2595, 1261, 2384], [1286, 1659, 1179, 1348, 1420, 2862, 1959]]
    c = {}
    for i, m in enumerate(managers):
        c[m] = {}
        for j, p in enumerate(projects):
            c[m][p] = cost_matrix[i][j]
    if len(cost_matrix) != len(managers):
        raise ValueError('Cost matrix row count does not match number of managers.')
    for i, row in enumerate(cost_matrix):
        if len(row) != len(projects):
            raise ValueError(f'Cost matrix row {i} does not match number of projects.')
    for m in managers:
        if m not in c or len(c[m]) != len(projects):
            raise ValueError(f'Missing cost data for manager {m}.')
        for p in projects:
            if p not in c[m]:
                raise ValueError(f'Missing cost data for manager {m}, project {p}.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c[man][proj] * x[man, proj] for man in managers for proj in projects)), GRB.MINIMIZE)
    for man in managers:
        m.addConstr(gp.quicksum((x[man, proj] for proj in projects)) == 1, name='')
    for proj in projects:
        m.addConstr(gp.quicksum((x[man, proj] for man in managers)) == 1, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for man in managers:
            for proj in projects:
                var = x[man, proj]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_assignment()