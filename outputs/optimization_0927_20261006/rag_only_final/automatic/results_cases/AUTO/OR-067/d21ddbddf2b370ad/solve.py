from gurobipy import Model, GRB, quicksum
managers = ['MA', 'MB', 'MC']
projects = ['P1', 'P2', 'P3']
cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
for i in managers:
    if i not in cost:
        raise ValueError(f'Missing cost data for manager {i}')
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for manager {i}, project {j}')

def build_assignment_model():
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, lb=0, ub=1, name='')
    for i in managers:
        m.addConstr(quicksum((x_vars[i, j] for j in projects)) == 1, name='mgr_%s' % i)
    for j in projects:
        m.addConstr(quicksum((x_vars[i, j] for i in managers)) == 1, name='prj_%s' % j)
    m.setObjective(quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    return m
m = build_assignment_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')