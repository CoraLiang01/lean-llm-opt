from gurobipy import Model, GRB, quicksum
managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7']
projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7']
cost_matrix = [[2972, 2727, 2795, 2922, 1302, 2489, 1533], [1094, 2158, 2990, 1844, 2887, 2021, 2288], [2133, 1675, 2422, 2639, 1033, 2261, 1695], [1951, 2309, 2070, 2802, 2328, 1313, 2434], [1269, 2153, 1296, 2685, 2627, 1610, 1641], [1220, 1192, 2907, 2622, 2595, 1261, 2384], [1286, 1659, 1179, 1348, 1420, 2862, 1959]]
cost = {}
for (i, m) in enumerate(managers):
    for (j, p) in enumerate(projects):
        cost[m, p] = cost_matrix[i][j]
if len(cost_matrix) != len(managers):
    raise ValueError('Cost matrix row count does not match number of managers.')
for row in cost_matrix:
    if len(row) != len(projects):
        raise ValueError('Cost matrix column count does not match number of projects.')
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f'Missing cost entry for ({m}, {p})')

def build_assignment_model():
    model = Model()
    model.Params.MIPGap = 0.0001
    x_vars = model.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for m in managers:
        model.addConstr(quicksum((x_vars[m, p] for p in projects)) == 1, name='')
    for p in projects:
        model.addConstr(quicksum((x_vars[m, p] for m in managers)) == 1, name='')
    model.setObjective(quicksum((cost[m, p] * x_vars[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    return model
m = build_assignment_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')