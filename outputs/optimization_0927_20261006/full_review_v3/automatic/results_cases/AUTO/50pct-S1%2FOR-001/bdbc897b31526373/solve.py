import gurobipy as gp
from gurobipy import GRB
managers = ['MA', 'MB', 'MC']
projects = ['P1', 'P2', 'P3']
cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
for i in managers:
    if i not in cost:
        raise ValueError(f'Missing cost data for manager {i}')
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for manager {i}, project {j}')
m = gp.Model('manager_project_assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')