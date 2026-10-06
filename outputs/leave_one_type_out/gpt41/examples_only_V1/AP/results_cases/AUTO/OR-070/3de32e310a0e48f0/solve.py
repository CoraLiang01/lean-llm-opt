from gurobipy import Model, GRB
managers = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']
projects = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
cost_matrix = [[2972, 2727, 2795, 2922, 1302, 2489, 1533], [1094, 2158, 2990, 1844, 2887, 2021, 2288], [2133, 1675, 2422, 2639, 1033, 2261, 1695], [1951, 2309, 2070, 2802, 2328, 1313, 2434], [1269, 2153, 1296, 2685, 2627, 1610, 1641], [1220, 1192, 2907, 2622, 2595, 1261, 2384], [1286, 1659, 1179, 1348, 1420, 2862, 1959]]
if len(cost_matrix) != len(managers):
    raise ValueError('Cost matrix row count does not match number of managers.')
for row in cost_matrix:
    if len(row) != len(projects):
        raise ValueError('Cost matrix column count does not match number of projects.')
cost = {}
for i, m in enumerate(managers):
    for j, p in enumerate(projects):
        cost[m, p] = cost_matrix[i][j]
m = Model()
m.setParam('MIPGap', 0.0001)
x = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
for mngr in managers:
    m.addConstr(sum((x[mngr, p] for p in projects)) == 1, name='')
for prj in projects:
    m.addConstr(sum((x[m, prj] for m in managers)) == 1, name='')
m.setObjective(sum((cost[mngr, prj] * x[mngr, prj] for mngr in managers for prj in projects)), GRB.MINIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for mngr in managers:
        for prj in projects:
            var = x[mngr, prj]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')