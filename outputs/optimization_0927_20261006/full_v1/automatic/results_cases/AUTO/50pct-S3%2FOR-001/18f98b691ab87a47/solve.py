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

def build_and_solve():
    model = gp.Model('manager_project_assignment')
    x_vars = model.addVars(managers, projects, vtype=GRB.BINARY, name='')
    model.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
    model.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
    model.Params.MIPGap = 0.0001
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = build_and_solve()