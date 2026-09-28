LEGACY_OBSERVATION = '{"values": {"Unnamed: 0": "MA", "P1": "2216", "P2": "1911", "P3": "1661", "P4": "2122", "P5": "1442", "P6": "1442"}}\n{"values": {"Unnamed: 0": "MB", "P1": "1100", "P2": "1271", "P3": "2764", "P4": "2557", "P5": "1036", "P6": "1036"}}\n{"values": {"Unnamed: 0": "MC", "P1": "2827", "P2": "2784", "P3": "2206", "P4": "2216", "P5": "2677", "P6": "2677"}}\n{"values": {"Unnamed: 0": "MD", "P1": "2627", "P2": "1273", "P3": "2610", "P4": "1957", "P5": "1594", "P6": "1594"}}\n{"values": {"Unnamed: 0": "ME", "P1": "3359", "P2": "1003", "P3": "2554", "P4": "1706", "P5": "2065", "P6": "2065"}}\n{"values": {"Unnamed: 0": "MF", "P1": "1579", "P2": "2289", "P3": "2368", "P4": "1922", "P5": "2740", "P6": "2740"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Unnamed: 0': 'MA', 'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442'}}, {'source': '', 'values': {'Unnamed: 0': 'MB', 'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036'}}, {'source': '', 'values': {'Unnamed: 0': 'MC', 'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677'}}, {'source': '', 'values': {'Unnamed: 0': 'MD', 'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594'}}, {'source': '', 'values': {'Unnamed: 0': 'ME', 'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065'}}, {'source': '', 'values': {'Unnamed: 0': 'MF', 'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740'}}]
import gurobipy as gp
from gurobipy import GRB
managers = []
projects = []
cost = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    manager = vals['Unnamed: 0']
    managers.append(manager)
    if not projects:
        projects = [k for k in vals if k != 'Unnamed: 0']
    for proj in projects:
        if manager not in cost:
            cost[manager] = {}
        cost[manager][proj] = int(vals[proj])
if set(cost.keys()) != set(managers):
    raise ValueError('Manager keys mismatch in cost data')
for m in managers:
    if set(cost[m].keys()) != set(projects):
        raise ValueError(f'Project keys mismatch in cost data for manager {m}')
m = gp.Model('ManagerProjectAssignment')
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