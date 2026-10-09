import gurobipy as gp
from gurobipy import GRB
teams = ['M1', 'M2', 'M3', 'M4']
jobs = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8']
capacity = {'M1': 13, 'M2': 12, 'M3': 12, 'M4': 12}
cost = {'M1': {'J1': 8, 'J2': 7, 'J3': 25, 'J4': 24, 'J5': 27, 'J6': 26, 'J7': 28, 'J8': 29}, 'M2': {'J1': 23, 'J2': 24, 'J3': 6, 'J4': 9, 'J5': 25, 'J6': 27, 'J7': 26, 'J8': 28}, 'M3': {'J1': 27, 'J2': 26, 'J3': 24, 'J4': 25, 'J5': 5, 'J6': 8, 'J7': 23, 'J8': 24}, 'M4': {'J1': 25, 'J2': 27, 'J3': 26, 'J4': 24, 'J5': 23, 'J6': 25, 'J7': 6, 'J8': 7}}
resource = {'M1': {'J1': 5, 'J2': 6, 'J3': 8, 'J4': 7, 'J5': 9, 'J6': 8, 'J7': 7, 'J8': 7}, 'M2': {'J1': 8, 'J2': 7, 'J3': 4, 'J4': 7, 'J5': 8, 'J6': 9, 'J7': 8, 'J8': 7}, 'M3': {'J1': 9, 'J2': 8, 'J3': 7, 'J4': 8, 'J5': 6, 'J6': 5, 'J7': 8, 'J8': 7}, 'M4': {'J1': 8, 'J2': 8, 'J3': 7, 'J4': 8, 'J5': 8, 'J6': 7, 'J7': 5, 'J8': 6}}
for i in teams:
    if i not in capacity:
        raise ValueError(f'Missing capacity for team {i}')
    if i not in cost or i not in resource:
        raise ValueError(f'Missing cost or resource data for team {i}')
    for j in jobs:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for team {i}, job {j}')
        if j not in resource[i]:
            raise ValueError(f'Missing resource for team {i}, job {j}')
m = gp.Model('GeneralizedAssignment')
x_vars = m.addVars(teams, jobs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in teams for j in jobs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in teams)) == 1 for j in jobs), name='')
m.addConstrs((gp.quicksum((resource[i][j] * x_vars[i, j] for j in jobs)) <= capacity[i] for i in teams), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')