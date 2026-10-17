SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
import gurobipy as gp
from gurobipy import GRB
SOURCE_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MA', 'P1': '3000', 'P2': '3200', 'P3': '3100'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MB', 'P1': '2800', 'P2': '3300', 'P3': '2900'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', 'values': {'Unnamed: 0': 'MC', 'P1': '2900', 'P2': '3100', 'P3': '3000'}}]
managers = []
projects = []
for rec in SOURCE_RECORDS:
    manager = rec['values']['Unnamed: 0']
    managers.append(manager)
    for proj in rec['values']:
        if proj != 'Unnamed: 0' and proj not in projects:
            projects.append(proj)
cost = {}
for rec in SOURCE_RECORDS:
    manager = rec['values']['Unnamed: 0']
    cost[manager] = {}
    for proj in projects:
        val = rec['values'][proj]
        if val == '' or val is None:
            raise ValueError(f'Missing cost for manager {manager}, project {proj}')
        cost[manager][proj] = float(val)
eligible_pairs = {(i, j): True for i in managers for j in projects}
pair_mask = eligible_pairs
m = gp.Model('manager_project_assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
for (i, j), eligible in pair_mask.items():
    if not eligible:
        m.addConstr(x_vars[i, j] == 0)
m.addConstrs((gp.quicksum((x_vars[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for i in managers)) == 1 for j in projects), name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')