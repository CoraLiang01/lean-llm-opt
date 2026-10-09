import gurobipy as gp
from gurobipy import GRB
managers = ['MA', 'MB', 'MC']
projects = ['P1', 'P2', 'P3']
cost = {'MA': {'P1': 3000, 'P2': 3200, 'P3': 3100}, 'MB': {'P1': 2800, 'P2': 3300, 'P3': 2900}, 'MC': {'P1': 2900, 'P2': 3100, 'P3': 3000}}
for m in managers:
    if m not in cost or not isinstance(cost[m], dict):
        raise ValueError(f'Missing cost data for manager {m}')
    for p in projects:
        if p not in cost[m]:
            raise ValueError(f'Missing cost data for manager {m}, project {p}')
m = gp.Model('Manager_Project_Assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr][prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
m.addConstrs((gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')