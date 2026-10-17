LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MA",\n      "P1": "2216",\n      "P2": "1911",\n      "P3": "1661",\n      "P4": "2122",\n      "P5": "1442",\n      "P6": "1442"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MB",\n      "P1": "1100",\n      "P2": "1271",\n      "P3": "2764",\n      "P4": "2557",\n      "P5": "1036",\n      "P6": "1036"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MC",\n      "P1": "2827",\n      "P2": "2784",\n      "P3": "2206",\n      "P4": "2216",\n      "P5": "2677",\n      "P6": "2677"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MD",\n      "P1": "2627",\n      "P2": "1273",\n      "P3": "2610",\n      "P4": "1957",\n      "P5": "1594",\n      "P6": "1594"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "ME",\n      "P1": "3359",\n      "P2": "1003",\n      "P3": "2554",\n      "P4": "1706",\n      "P5": "2065",\n      "P6": "2065"\n    }\n  },\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv",\n    "values": {\n      "Unnamed: 0": "MF",\n      "P1": "1579",\n      "P2": "2289",\n      "P3": "2368",\n      "P4": "1922",\n      "P5": "2740",\n      "P6": "2740"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MD', 'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'ME', 'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MF', 'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '2216', 'P2': '1911', 'P3': '1661', 'P4': '2122', 'P5': '1442', 'P6': '1442'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '1100', 'P2': '1271', 'P3': '2764', 'P4': '2557', 'P5': '1036', 'P6': '1036'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2827', 'P2': '2784', 'P3': '2206', 'P4': '2216', 'P5': '2677', 'P6': '2677'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MD', 'P1': '2627', 'P2': '1273', 'P3': '2610', 'P4': '1957', 'P5': '1594', 'P6': '1594'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'ME', 'P1': '3359', 'P2': '1003', 'P3': '2554', 'P4': '1706', 'P5': '2065', 'P6': '2065'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MF', 'P1': '1579', 'P2': '2289', 'P3': '2368', 'P4': '1922', 'P5': '2740', 'P6': '2740'}}]
managers = []
projects = []
cost = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    manager = vals['Unnamed: 0']
    managers.append(manager)
    if not projects:
        projects = [k for k in vals if k != 'Unnamed: 0']
    cost[manager] = {pj: int(vals[pj]) for pj in projects}
for i in managers:
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for manager {i}, project {j}')
eligible_pairs = [(i, j) for i in managers for j in projects]
m = gp.Model('Original_RAG_AP')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
for i, j in [(i, j) for i in managers for j in projects if (i, j) not in eligible_pairs]:
    m.addConstr(x_vars[i, j] == 0)
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i, j in eligible_pairs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')