LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MA",\n      "P1": "3000",\n      "P2": "3200",\n      "P3": "3100"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MB",\n      "P1": "2800",\n      "P2": "3300",\n      "P3": "2900"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MC",\n      "P1": "2900",\n      "P2": "3100",\n      "P3": "3000"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
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
    cost[manager] = {}
    for k, v in vals.items():
        if k == 'Unnamed: 0':
            continue
        if k not in projects:
            projects.append(k)
        cost[manager][k] = float(v)
for i in managers:
    if i not in cost:
        raise ValueError(f'Missing cost row for manager {i}')
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost entry for manager {i}, project {j}')
eligible_pairs = [(i, j) for i in managers for j in projects]
m = gp.Model('AP_Manager_Project')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i, j in eligible_pairs)), GRB.MINIMIZE)
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