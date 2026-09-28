LEGACY_OBSERVATION = '{"values": {"Unnamed: 0": "MA", "P1": "3000", "P2": "3200", "P3": "3100"}}\n{"values": {"Unnamed: 0": "MB", "P1": "2800", "P2": "3300", "P3": "2900"}}\n{"values": {"Unnamed: 0": "MC", "P1": "2900", "P2": "3100", "P3": "3000"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
managers = []
projects = []
cost = {}
for rec in records:
    vals = rec['values']
    m = vals['Unnamed: 0']
    if m not in managers:
        managers.append(m)
    for p in vals:
        if p == 'Unnamed: 0':
            continue
        if p not in projects:
            projects.append(p)
        if m not in cost:
            cost[m] = {}
        cost[m][p] = int(vals[p])
for m in managers:
    if m not in cost or any((p not in cost[m] for p in projects)):
        raise ValueError(f'Missing cost data for manager {m}')
m = gp.Model('assignment')
x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr][prj] * x[mgr, prj] for mgr in managers for prj in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[mgr, prj] for prj in projects)) == 1 for mgr in managers), name='')
m.addConstrs((gp.quicksum((x[mgr, prj] for mgr in managers)) == 1 for prj in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')