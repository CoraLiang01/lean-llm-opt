import gurobipy as gp
from gurobipy import GRB

def solve_assignment():
    managers = [1, 2, 3, 4, 5, 6, 7]
    projects = [1, 2, 3, 4, 5, 6, 7]
    cost_matrix = [[2972, 2727, 2795, 2922, 1302, 2489, 1533], [1094, 2158, 2990, 1844, 2887, 2021, 2288], [2133, 1675, 2422, 2639, 1033, 2261, 1695], [1951, 2309, 2070, 2802, 2328, 1313, 2434], [1269, 2153, 1296, 2685, 2627, 1610, 1641], [1220, 1192, 2907, 2622, 2595, 1261, 2384], [1286, 1659, 1179, 1348, 1420, 2862, 1959]]
    if len(cost_matrix) != len(managers):
        raise ValueError('Cost matrix row count does not match number of managers.')
    for row in cost_matrix:
        if len(row) != len(projects):
            raise ValueError('Cost matrix column count does not match number of projects.')
    c = {}
    for (i_idx, i) in enumerate(managers):
        for (j_idx, j) in enumerate(projects):
            c[i, j] = cost_matrix[i_idx][j_idx]
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for i in managers:
        m.addConstr(gp.quicksum((x[i, j] for j in projects)) == 1, name='cm')
    for j in projects:
        m.addConstr(gp.quicksum((x[i, j] for i in managers)) == 1, name='cp')
    m.setObjective(gp.quicksum((c[i, j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in managers:
            for j in projects:
                var = x[i, j]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_assignment()