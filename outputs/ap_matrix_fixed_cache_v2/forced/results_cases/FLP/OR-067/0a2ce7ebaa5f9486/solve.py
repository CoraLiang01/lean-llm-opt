LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv", "values": {"Unnamed: 0": "MA", "P1": "3000", "P2": "3200", "P3": "3100"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv", "values": {"Unnamed: 0": "MB", "P1": "2800", "P2": "3300", "P3": "2900"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv", "values": {"Unnamed: 0": "MC", "P1": "2900", "P2": "3100", "P3": "3000"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
import gurobipy as gp
from gurobipy import GRB
records = [r for r in LEGACY_RECORDS if r['source'] == '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv']
if not records:
    raise ValueError('No records found for manager_project_costs.csv')
managers = []
projects = []
cost = {}
for rec in records:
    vals = rec['values']
    manager = vals['Unnamed: 0']
    managers.append(manager)
    if not projects:
        projects = [k for k in vals if k != 'Unnamed: 0']
    cost[manager] = {}
    for proj in projects:
        try:
            cost[manager][proj] = float(vals[proj])
        except Exception as e:
            raise ValueError(f'Invalid cost for manager {manager}, project {proj}: {vals[proj]}') from e
if len(managers) != len(projects):
    raise ValueError('Number of managers and projects must be equal for one-to-one assignment.')
for manager in managers:
    for proj in projects:
        if proj not in cost[manager]:
            raise ValueError(f'Missing cost for manager {manager}, project {proj}')
m = gp.Model('Original_RAG_Assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
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