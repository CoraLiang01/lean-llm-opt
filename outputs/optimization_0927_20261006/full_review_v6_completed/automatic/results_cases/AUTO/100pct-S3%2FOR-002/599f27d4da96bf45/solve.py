import gurobipy as gp
from gurobipy import GRB
managers = ['MA', 'MB', 'MC', 'MD', 'ME', 'MF']
projects = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
cost = {'MA': {'P1': 2216, 'P2': 1911, 'P3': 1661, 'P4': 2122, 'P5': 1442, 'P6': 1442}, 'MB': {'P1': 1100, 'P2': 1271, 'P3': 2764, 'P4': 2557, 'P5': 1036, 'P6': 1036}, 'MC': {'P1': 2827, 'P2': 2784, 'P3': 2206, 'P4': 2216, 'P5': 2677, 'P6': 2677}, 'MD': {'P1': 2627, 'P2': 1273, 'P3': 2610, 'P4': 1957, 'P5': 1594, 'P6': 1594}, 'ME': {'P1': 3359, 'P2': 1003, 'P3': 2554, 'P4': 1706, 'P5': 2065, 'P6': 2065}, 'MF': {'P1': 1579, 'P2': 2289, 'P3': 2368, 'P4': 1922, 'P5': 2740, 'P6': 2740}}
for i in managers:
    if i not in cost or not isinstance(cost[i], dict):
        raise ValueError(f'Missing cost data for manager {i}')
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for manager {i}, project {j}')
m = gp.Model('Manager_Project_Assignment')
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