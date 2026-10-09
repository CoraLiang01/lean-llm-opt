import gurobipy as gp
from gurobipy import GRB
managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7']
projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7']
cost = {'Manager 1': {'Project 1': 2972, 'Project 2': 2727, 'Project 3': 2795, 'Project 4': 2922, 'Project 5': 1302, 'Project 6': 2489, 'Project 7': 1533}, 'Manager 2': {'Project 1': 1094, 'Project 2': 2158, 'Project 3': 2990, 'Project 4': 1844, 'Project 5': 2887, 'Project 6': 2021, 'Project 7': 2288}, 'Manager 3': {'Project 1': 2133, 'Project 2': 1675, 'Project 3': 2422, 'Project 4': 2639, 'Project 5': 1033, 'Project 6': 2261, 'Project 7': 1695}, 'Manager 4': {'Project 1': 1951, 'Project 2': 2309, 'Project 3': 2070, 'Project 4': 2802, 'Project 5': 2328, 'Project 6': 1313, 'Project 7': 2434}, 'Manager 5': {'Project 1': 1269, 'Project 2': 2153, 'Project 3': 1296, 'Project 4': 2685, 'Project 5': 2627, 'Project 6': 1610, 'Project 7': 1641}, 'Manager 6': {'Project 1': 1220, 'Project 2': 1192, 'Project 3': 2907, 'Project 4': 2622, 'Project 5': 2595, 'Project 6': 1261, 'Project 7': 2384}, 'Manager 7': {'Project 1': 1286, 'Project 2': 1659, 'Project 3': 1179, 'Project 4': 1348, 'Project 5': 1420, 'Project 6': 2862, 'Project 7': 1959}}
for m in managers:
    if m not in cost:
        raise ValueError(f'Missing cost data for manager {m}')
    for p in projects:
        if p not in cost[m]:
            raise ValueError(f'Missing cost data for manager {m}, project {p}')

def build_assignment_model():
    model = gp.Model('Manager_Project_Assignment')
    x_vars = model.addVars(managers, projects, vtype=GRB.BINARY, lb=0, name='')
    model.setObjective(gp.quicksum((cost[m][p] * x_vars[m, p] for m in managers for p in projects)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum((x_vars[m, p] for m in managers)) == 1 for p in projects), name='')
    model.addConstrs((gp.quicksum((x_vars[m, p] for p in projects)) == 1 for m in managers), name='')
    return model
m = build_assignment_model()
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')