from gurobipy import Model, GRB, quicksum
managers = ['MA', 'MB', 'MC', 'MD', 'ME', 'MF']
projects = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
cost = {'MA': {'P1': 2216, 'P2': 1911, 'P3': 1661, 'P4': 2122, 'P5': 1442, 'P6': 1442}, 'MB': {'P1': 1100, 'P2': 1271, 'P3': 2764, 'P4': 2557, 'P5': 1036, 'P6': 1036}, 'MC': {'P1': 2827, 'P2': 2784, 'P3': 2206, 'P4': 2216, 'P5': 2677, 'P6': 2677}, 'MD': {'P1': 2627, 'P2': 1273, 'P3': 2610, 'P4': 1957, 'P5': 1594, 'P6': 1594}, 'ME': {'P1': 3359, 'P2': 1003, 'P3': 2554, 'P4': 1706, 'P5': 2065, 'P6': 2065}, 'MF': {'P1': 1579, 'P2': 2289, 'P3': 2368, 'P4': 1922, 'P5': 2740, 'P6': 2740}}
if set(cost.keys()) != set(managers):
    raise ValueError('Cost matrix manager keys do not match managers set.')
for i in managers:
    if set(cost[i].keys()) != set(projects):
        raise ValueError(f'Cost matrix project keys for manager {i} do not match projects set.')

def build_assignment_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    for i in managers:
        m.addConstr(quicksum((x_vars[i, j] for j in projects)) == 1, name='mgr')
    for j in projects:
        m.addConstr(quicksum((x_vars[i, j] for i in managers)) == 1, name='prj')
    m.setObjective(quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.update()
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