import gurobipy as gp
from gurobipy import GRB
workstations = ['W1', 'W2', 'W3', 'W4']
jobs = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9']
workstation_capacity = {'W1': 15, 'W2': 14, 'W3': 16, 'W4': 13}
assignment_costs = {'W1': {'J1': 6, 'J2': 8, 'J3': 18, 'J4': 20, 'J5': 21, 'J6': 19, 'J7': 23, 'J8': 22, 'J9': 24}, 'W2': {'J1': 19, 'J2': 18, 'J3': 7, 'J4': 6, 'J5': 20, 'J6': 22, 'J7': 21, 'J8': 23, 'J9': 25}, 'W3': {'J1': 22, 'J2': 21, 'J3': 20, 'J4': 19, 'J5': 5, 'J6': 7, 'J7': 18, 'J8': 20, 'J9': 21}, 'W4': {'J1': 21, 'J2': 22, 'J3': 23, 'J4': 20, 'J5': 19, 'J6': 18, 'J7': 6, 'J8': 8, 'J9': 7}}
assignment_resources = {'W1': {'J1': 4, 'J2': 5, 'J3': 7, 'J4': 8, 'J5': 8, 'J6': 7, 'J7': 8, 'J8': 9, 'J9': 8}, 'W2': {'J1': 7, 'J2': 8, 'J3': 4, 'J4': 5, 'J5': 8, 'J6': 8, 'J7': 7, 'J8': 8, 'J9': 9}, 'W3': {'J1': 8, 'J2': 7, 'J3': 8, 'J4': 7, 'J5': 5, 'J6': 4, 'J7': 7, 'J8': 8, 'J9': 7}, 'W4': {'J1': 8, 'J2': 8, 'J3': 9, 'J4': 8, 'J5': 7, 'J6': 7, 'J7': 4, 'J8': 5, 'J9': 4}}
for i in workstations:
    if i not in workstation_capacity:
        raise ValueError(f'Missing capacity for workstation {i}')
    if i not in assignment_costs or i not in assignment_resources:
        raise ValueError(f'Missing assignment data for workstation {i}')
    for j in jobs:
        if j not in assignment_costs[i]:
            raise ValueError(f'Missing cost for ({i},{j})')
        if j not in assignment_resources[i]:
            raise ValueError(f'Missing resource for ({i},{j})')
m = gp.Model('GeneralizedAssignment')
x = m.addVars(workstations, jobs, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((assignment_costs[i][j] * x[i, j] for i in workstations for j in jobs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in workstations)) == 1 for j in jobs), name='')
m.addConstrs((gp.quicksum((assignment_resources[i][j] * x[i, j] for j in jobs)) <= workstation_capacity[i] for i in workstations), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')