from gurobipy import Model, GRB
managers = ['MA', 'MB', 'MC']
projects = ['P1', 'P2', 'P3']
cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
if set(cost.keys()) != set(managers):
    raise ValueError('Cost matrix manager keys do not match managers set.')
for i in managers:
    if set(cost[i].keys()) != set(projects):
        raise ValueError(f'Cost matrix for manager {i} does not cover all projects.')
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(sum((cost[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
for i in managers:
    m.addConstr(sum((x[i, j] for j in projects)) == 1, name='')
for j in projects:
    m.addConstr(sum((x[i, j] for i in managers)) == 1, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in managers:
        for j in projects:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')