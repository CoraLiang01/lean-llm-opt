import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7']
    projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7']
    cost_matrix = [[2972, 2727, 2795, 2922, 1302, 2489, 1533], [1094, 2158, 2990, 1844, 2887, 2021, 2288], [2133, 1675, 2422, 2639, 1033, 2261, 1695], [1951, 2309, 2070, 2802, 2328, 1313, 2434], [1269, 2153, 1296, 2685, 2627, 1610, 1641], [1220, 1192, 2907, 2622, 2595, 1261, 2384], [1286, 1659, 1179, 1348, 1420, 2862, 1959]]
    if len(cost_matrix) != len(managers):
        raise ValueError('Cost matrix row count does not match number of managers.')
    for row in cost_matrix:
        if len(row) != len(projects):
            raise ValueError('Cost matrix column count does not match number of projects.')
    c = {}
    for i, manager in enumerate(managers):
        for j, project in enumerate(projects):
            c[manager, project] = cost_matrix[i][j]
    m = gp.Model('assignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    m.setObjective(gp.quicksum((c[manager, project] * x[manager, project] for manager in managers for project in projects)), GRB.MINIMIZE)
    for manager in managers:
        m.addConstr(gp.quicksum((x[manager, project] for project in projects)) == 1, name='mgr')
    for project in projects:
        m.addConstr(gp.quicksum((x[manager, project] for manager in managers)) == 1, name='prj')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for manager in managers:
            for project in projects:
                var = x[manager, project]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()