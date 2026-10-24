CSVQA_DATA = {'query': 'A set of managers and a set of construction projects are given. Assigning manager i to project j incurs a cost c_ij, reflecting the manager\'s suitability, experience, and expertise for that project. The cost matrix is provided in the CSV file "manager_project_costs.csv", where each row corresponds to a manager and each column corresponds to a project.\n\nThe task is to determine a minimum-cost one-to-one assignment of managers to projects. Each manager must be assigned to exactly one project, and each project must be assigned to exactly one manager. The objective is to minimize the total assignment cost over all manager-project pairs.', 'relationships': [], 'tables': [{'columns': ['Manager', 'Project 1 Cost', 'Project 2 Cost', 'Project 3 Cost', 'Project 4 Cost', 'Project 5 Cost', 'Project 6 Cost', 'Project 7 Cost', 'Project 8 Cost', 'Project 9 Cost', 'Project 10 Cost', 'Project 11 Cost'], 'file_name': 'manager_project_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Manager': 'Manager 1', 'Project 1 Cost': '708', 'Project 10 Cost': '3167', 'Project 11 Cost': '1711', 'Project 2 Cost': '1948', 'Project 3 Cost': '2424', 'Project 4 Cost': '1068', 'Project 5 Cost': '729', 'Project 6 Cost': '199', 'Project 7 Cost': '1651', 'Project 8 Cost': '3174', 'Project 9 Cost': '3211'}}, {'data_row_number': 1, 'values': {'Manager': 'Manager 2', 'Project 1 Cost': '1700', 'Project 10 Cost': '2603', 'Project 11 Cost': '1822', 'Project 2 Cost': '2670', 'Project 3 Cost': '1883', 'Project 4 Cost': '2534', 'Project 5 Cost': '1429', 'Project 6 Cost': '1173', 'Project 7 Cost': '777', 'Project 8 Cost': '248', 'Project 9 Cost': '1704'}}, {'data_row_number': 2, 'values': {'Manager': 'Manager 3', 'Project 1 Cost': '160', 'Project 10 Cost': '2502', 'Project 11 Cost': '2595', 'Project 2 Cost': '755', 'Project 3 Cost': '3477', 'Project 4 Cost': '3122', 'Project 5 Cost': '2968', 'Project 6 Cost': '3023', 'Project 7 Cost': '1417', 'Project 8 Cost': '254', 'Project 9 Cost': '3175'}}, {'data_row_number': 3, 'values': {'Manager': 'Manager 4', 'Project 1 Cost': '2213', 'Project 10 Cost': '1954', 'Project 11 Cost': '1805', 'Project 2 Cost': '1008', 'Project 3 Cost': '411', 'Project 4 Cost': '1199', 'Project 5 Cost': '418', 'Project 6 Cost': '1000', 'Project 7 Cost': '3148', 'Project 8 Cost': '1724', 'Project 9 Cost': '1984'}}, {'data_row_number': 4, 'values': {'Manager': 'Manager 5', 'Project 1 Cost': '198', 'Project 10 Cost': '270', 'Project 11 Cost': '1893', 'Project 2 Cost': '1721', 'Project 3 Cost': '1318', 'Project 4 Cost': '3194', 'Project 5 Cost': '3036', 'Project 6 Cost': '2938', 'Project 7 Cost': '3298', 'Project 8 Cost': '3332', 'Project 9 Cost': '1806'}}, {'data_row_number': 5, 'values': {'Manager': 'Manager 6', 'Project 1 Cost': '2375', 'Project 10 Cost': '2696', 'Project 11 Cost': '2217', 'Project 2 Cost': '1804', 'Project 3 Cost': '3174', 'Project 4 Cost': '1607', 'Project 5 Cost': '2168', 'Project 6 Cost': '1642', 'Project 7 Cost': '970', 'Project 8 Cost': '3433', 'Project 9 Cost': '1528'}}, {'data_row_number': 6, 'values': {'Manager': 'Manager 7', 'Project 1 Cost': '2400', 'Project 10 Cost': '2762', 'Project 11 Cost': '577', 'Project 2 Cost': '211', 'Project 3 Cost': '1172', 'Project 4 Cost': '425', 'Project 5 Cost': '1222', 'Project 6 Cost': '287', 'Project 7 Cost': '653', 'Project 8 Cost': '1466', 'Project 9 Cost': '479'}}, {'data_row_number': 7, 'values': {'Manager': 'Manager 8', 'Project 1 Cost': '272', 'Project 10 Cost': '3260', 'Project 11 Cost': '2981', 'Project 2 Cost': '2574', 'Project 3 Cost': '413', 'Project 4 Cost': '202', 'Project 5 Cost': '1220', 'Project 6 Cost': '2392', 'Project 7 Cost': '410', 'Project 8 Cost': '2250', 'Project 9 Cost': '2272'}}, {'data_row_number': 8, 'values': {'Manager': 'Manager 9', 'Project 1 Cost': '2844', 'Project 10 Cost': '114', 'Project 11 Cost': '2161', 'Project 2 Cost': '2775', 'Project 3 Cost': '357', 'Project 4 Cost': '2601', 'Project 5 Cost': '1627', 'Project 6 Cost': '125', 'Project 7 Cost': '1029', 'Project 8 Cost': '1354', 'Project 9 Cost': '2280'}}, {'data_row_number': 9, 'values': {'Manager': 'Manager 10', 'Project 1 Cost': '1222', 'Project 10 Cost': '1873', 'Project 11 Cost': '1185', 'Project 2 Cost': '296', 'Project 3 Cost': '3375', 'Project 4 Cost': '352', 'Project 5 Cost': '2167', 'Project 6 Cost': '2202', 'Project 7 Cost': '3139', 'Project 8 Cost': '2526', 'Project 9 Cost': '767'}}, {'data_row_number': 10, 'values': {'Manager': 'Manager 11', 'Project 1 Cost': '2661', 'Project 10 Cost': '3003', 'Project 11 Cost': '2161', 'Project 2 Cost': '887', 'Project 3 Cost': '455', 'Project 4 Cost': '2552', 'Project 5 Cost': '1067', 'Project 6 Cost': '552', 'Project 7 Cost': '2991', 'Project 8 Cost': '1727', 'Project 9 Cost': '1639'}}], 'table_id': 'file_0_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for tbl in data.get('tables', []):
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    managers = []
    costs = {}
    projects_set = set()
    for rec in records:
        values = rec.get('values', {})
        manager = values.get('Manager')
        if manager is None:
            raise ValueError("Record missing 'Manager' field")
        managers.append(manager)
        costs[manager] = {}
        for col, val in values.items():
            if col.startswith('Project ') and col.endswith(' Cost'):
                parts = col.split()
                if len(parts) != 3:
                    continue
                try:
                    proj_num = int(parts[1])
                except ValueError:
                    continue
                project_id = f'Project {proj_num}'
                try:
                    cost_val = int(val)
                except ValueError:
                    raise ValueError(f'Non-integer cost value for {manager}, {project_id}')
                costs[manager][project_id] = cost_val
                projects_set.add(project_id)
    if not managers or not projects_set:
        raise ValueError('No managers or projects found in data')
    for m in managers:
        if len(costs[m]) != len(projects_set):
            missing = projects_set - costs[m].keys()
            raise ValueError(f'Missing cost entries for manager {m}: {missing}')
    projects = sorted(projects_set, key=lambda p: int(p.split()[1]))
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((costs[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')
