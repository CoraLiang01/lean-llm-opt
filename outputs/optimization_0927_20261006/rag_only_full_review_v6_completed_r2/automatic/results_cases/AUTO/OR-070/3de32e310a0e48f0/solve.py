from gurobipy import Model, GRB, quicksum
managers = [1, 2, 3, 4, 5, 6, 7]
projects = [1, 2, 3, 4, 5, 6, 7]
cost_matrix = {1: {1: 2972, 2: 2727, 3: 2795, 4: 2922, 5: 1302, 6: 2489, 7: 1533}, 2: {1: 1094, 2: 2158, 3: 2990, 4: 1844, 5: 2887, 6: 2021, 7: 2288}, 3: {1: 2133, 2: 1675, 3: 2422, 4: 2639, 5: 1033, 6: 2261, 7: 1695}, 4: {1: 1951, 2: 2309, 3: 2070, 4: 2802, 5: 2328, 6: 1313, 7: 2434}, 5: {1: 1269, 2: 2153, 3: 1296, 4: 2685, 5: 2627, 6: 1610, 7: 1641}, 6: {1: 1220, 2: 1192, 3: 2907, 4: 2622, 5: 2595, 6: 1261, 7: 2384}, 7: {1: 1286, 2: 1659, 3: 1179, 4: 1348, 5: 1420, 6: 2862, 7: 1959}}
if set(cost_matrix.keys()) != set(managers):
    raise ValueError('Manager keys in cost_matrix do not match managers set.')
for i in managers:
    if set(cost_matrix[i].keys()) != set(projects):
        raise ValueError(f'Project keys in cost_matrix for manager {i} do not match projects set.')

def build_assignment_model():
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    for j in projects:
        m.addConstr(quicksum((x_vars[i, j] for i in managers)) == 1, name='pj')
    for i in managers:
        m.addConstr(quicksum((x_vars[i, j] for j in projects)) <= 1, name='mg')
    m.setObjective(quicksum((cost_matrix[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in managers:
            for j in projects:
                v = x_vars[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_assignment_model()