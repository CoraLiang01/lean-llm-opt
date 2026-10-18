LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 1", "Project 1 Cost": "2972", "Project 2 Cost": "2727", "Project 3 Cost": "2795", "Project 4 Cost": "2922", "Project 5 Cost": "1302", "Project 6 Cost": "2489", "Project 7 Cost": "1533"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 2", "Project 1 Cost": "1094", "Project 2 Cost": "2158", "Project 3 Cost": "2990", "Project 4 Cost": "1844", "Project 5 Cost": "2887", "Project 6 Cost": "2021", "Project 7 Cost": "2288"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 3", "Project 1 Cost": "2133", "Project 2 Cost": "1675", "Project 3 Cost": "2422", "Project 4 Cost": "2639", "Project 5 Cost": "1033", "Project 6 Cost": "2261", "Project 7 Cost": "1695"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 4", "Project 1 Cost": "1951", "Project 2 Cost": "2309", "Project 3 Cost": "2070", "Project 4 Cost": "2802", "Project 5 Cost": "2328", "Project 6 Cost": "1313", "Project 7 Cost": "2434"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 5", "Project 1 Cost": "1269", "Project 2 Cost": "2153", "Project 3 Cost": "1296", "Project 4 Cost": "2685", "Project 5 Cost": "2627", "Project 6 Cost": "1610", "Project 7 Cost": "1641"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 6", "Project 1 Cost": "1220", "Project 2 Cost": "1192", "Project 3 Cost": "2907", "Project 4 Cost": "2622", "Project 5 Cost": "2595", "Project 6 Cost": "1261", "Project 7 Cost": "2384"}}, {"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv", "values": {"Manager": "Manager 7", "Project 1 Cost": "1286", "Project 2 Cost": "1659", "Project 3 Cost": "1179", "Project 4 Cost": "1348", "Project 5 Cost": "1420", "Project 6 Cost": "2862", "Project 7 Cost": "1959"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 1', 'Project 1 Cost': '2972', 'Project 2 Cost': '2727', 'Project 3 Cost': '2795', 'Project 4 Cost': '2922', 'Project 5 Cost': '1302', 'Project 6 Cost': '2489', 'Project 7 Cost': '1533'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 2', 'Project 1 Cost': '1094', 'Project 2 Cost': '2158', 'Project 3 Cost': '2990', 'Project 4 Cost': '1844', 'Project 5 Cost': '2887', 'Project 6 Cost': '2021', 'Project 7 Cost': '2288'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 3', 'Project 1 Cost': '2133', 'Project 2 Cost': '1675', 'Project 3 Cost': '2422', 'Project 4 Cost': '2639', 'Project 5 Cost': '1033', 'Project 6 Cost': '2261', 'Project 7 Cost': '1695'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 4', 'Project 1 Cost': '1951', 'Project 2 Cost': '2309', 'Project 3 Cost': '2070', 'Project 4 Cost': '2802', 'Project 5 Cost': '2328', 'Project 6 Cost': '1313', 'Project 7 Cost': '2434'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 5', 'Project 1 Cost': '1269', 'Project 2 Cost': '2153', 'Project 3 Cost': '1296', 'Project 4 Cost': '2685', 'Project 5 Cost': '2627', 'Project 6 Cost': '1610', 'Project 7 Cost': '1641'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 6', 'Project 1 Cost': '1220', 'Project 2 Cost': '1192', 'Project 3 Cost': '2907', 'Project 4 Cost': '2622', 'Project 5 Cost': '2595', 'Project 6 Cost': '1261', 'Project 7 Cost': '2384'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv', 'values': {'Manager': 'Manager 7', 'Project 1 Cost': '1286', 'Project 2 Cost': '1659', 'Project 3 Cost': '1179', 'Project 4 Cost': '1348', 'Project 5 Cost': '1420', 'Project 6 Cost': '2862', 'Project 7 Cost': '1959'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
managers = []
projects = []
cost = {}
for rec in records:
    vals = rec['values']
    manager = vals['Manager']
    if manager not in managers:
        managers.append(manager)
    if not projects:
        projects = [k.replace(' Cost', '') for k in vals if k.endswith('Cost')]
    cost[manager] = {}
    for proj in projects:
        field = f'{proj} Cost'
        if field not in vals:
            raise ValueError(f'Missing cost for {manager}, {proj}')
        cost[manager][proj] = int(vals[field])
if len(managers) != len(projects):
    raise ValueError('Number of managers and projects must be equal for assignment.')
for m in managers:
    for p in projects:
        if p not in cost[m]:
            raise ValueError(f'Missing cost for {m}, {p}')
m = gp.Model('Manager_Project_Assignment')
x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager][project] * x_vars[manager, project] for manager in managers for project in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[manager, project] for manager in managers)) == 1 for project in projects), name='')
m.addConstrs((gp.quicksum((x_vars[manager, project] for project in projects)) <= 1 for manager in managers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')