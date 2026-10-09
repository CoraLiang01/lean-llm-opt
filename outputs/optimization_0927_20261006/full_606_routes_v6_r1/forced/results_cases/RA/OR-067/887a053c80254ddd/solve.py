LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MA",\n      "P1": "3000",\n      "P2": "3200",\n      "P3": "3100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MB",\n      "P1": "2800",\n      "P2": "3300",\n      "P3": "2900"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MC",\n      "P1": "2900",\n      "P2": "3100",\n      "P3": "3000"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
managers = []
projects = []
cost = {}
for rec in records:
    m = rec['values']['Unnamed: 0']
    if m not in managers:
        managers.append(m)
    for p in rec['values']:
        if p == 'Unnamed: 0':
            continue
        if p not in projects:
            projects.append(p)
        cost.setdefault(m, {})[p] = int(rec['values'][p])
for m in managers:
    for p in projects:
        if p not in cost[m]:
            raise ValueError(f'Missing cost for manager {m}, project {p}')
m = gp.Model('manager_project_assignment')
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