LEGACY_OBSERVATION = 'Unnamed: 0,P1,P2,P3,P4,P5,P6\nMA,2216,1911,1661,2122,1442,1442\nMB,1100,1271,2764,2557,1036,1036\nMC,2827,2784,2206,2216,2677,2677\nMD,2627,1273,2610,1957,1594,1594\nME,3359,1003,2554,1706,2065,2065\nMF,1579,2289,2368,1922,2740,2740'
LEGACY_RECORDS = [{'source': '', 'values': {'Unnamed: 0': 'MA', 'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442'}}, {'source': '', 'values': {'Unnamed: 0': 'MB', 'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036'}}, {'source': '', 'values': {'Unnamed: 0': 'MC', 'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677'}}, {'source': '', 'values': {'Unnamed: 0': 'MD', 'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594'}}, {'source': '', 'values': {'Unnamed: 0': 'ME', 'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065'}}, {'source': '', 'values': {'Unnamed: 0': 'MF', 'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
managers = []
projects = []
cost = {}
for rec in records:
    vals = rec['values']
    manager = vals['Unnamed: 0']
    managers.append(manager)
    for proj in vals:
        if proj != 'Unnamed: 0':
            if proj not in projects:
                projects.append(proj)
            if manager not in cost:
                cost[manager] = {}
            cost[manager][proj] = int(vals[proj])
if len(managers) != len(projects):
    raise ValueError('Number of managers and projects must be equal for assignment.')
for i in managers:
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost coefficient for manager {i}, project {j}.')
m = gp.Model('manager_project_assignment')
x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')